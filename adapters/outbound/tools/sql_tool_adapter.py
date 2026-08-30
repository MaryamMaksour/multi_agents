from domain.ports.database_port import DatabasePort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.cache_port import CachePort


from domain.exceptions import UnknownToolError, ToolExecutionError
from libs.agent_core.sql_validation import validate_identifier, validate_readonly_query


from typing import Any
import base64
import json
import re
import uuid
import zlib


MAX_OFFSET = 5000

_COLUMN_LINE_RE = re.compile(r'^"?([A-Za-z_][A-Za-z0-9_]*)"?\s+\S')
    


def _encode_cursor(payload: dict) -> str:
    s = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    c = zlib.compress(s, level=9)
    return base64.urlsafe_b64encode(c).rstrip(b"=").decode("ascii")


def _decode_cursor(cursor: str, max_bytes: int = 65536) -> dict:
    pad = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode(cursor + pad)
    decompressor = zlib.decompressobj()
    s = decompressor.decompress(raw, max_bytes)
    if decompressor.unconsumed_tail:
        raise ValueError("Cursor payload too large.")
    return json.loads(s.decode("utf-8"))

def _extract_column_names(table_schema: dict) -> set:
    columns_field = table_schema.get("columns")
    if isinstance(columns_field, (set, frozenset)):
        text = next(iter(columns_field), "")
    elif isinstance(columns_field, str):
        text = columns_field
    else:
        text = str(columns_field or "")

    names = set()
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        m = _COLUMN_LINE_RE.match(line)
        if m:
            names.add(m.group(1).lower())
    return names

class SqlToolAdapter():

    def __init__(self, db: DatabasePort, embeddings: EmbeddingPort, cache: CachePort,
                        allowed_tables: list[str], schema: dict, filters: dict, 
                        lsit_values: dict, dist_op: str, vector_ttl_seconds: int  ):

        self._db = db
        self._embeddings = embeddings
        self._cache = cache
        self._allowed_tables = {table.lower() for table in allowed_tables}
        self._schema = schema
        self._filters = filters
        self._lsit_values = lsit_values
        self._dist_op = dist_op 
        self._vector_ttl_seconds = vector_ttl_seconds

        # dispatcher: match the tool name with the method name
        self._handlers = {
            "get_table_schema": self._get_table_schema,
            "get_filter": self._get_filter,
            "get_lsit_values": self._get_lsit_values,
            "db_execute": self._db_execute,
            "get_table_records": self._get_table_records,
            "embed_query_tool": self._embed_query_tool,
            "execute_next_cursor": self._execute_next_cursor,
        }




    def get_tool_schemas(self) -> list[dict]:
        """Describe the SQL tools in the format the LLM adapter passes on.

        Names here must match the _handlers keys exactly - call_tool dispatches
        on them, so a mismatch is an UnknownToolError at runtime rather than a
        startup failure. (That includes "get_lsit_values": the spelling is the
        contract until the handler key is renamed too.)

        The descriptions carry the rules that _db_execute enforces by
        returning {"error": ...}. Stating them here turns a wasted round trip
        into a query the model gets right the first time - the error path stays
        as the guarantee, this is just the shorter route to it.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_table_schema",
                    "description": (
                        "Return the columns and types of the given tables. Call this "
                        "before writing any query - never guess a column name. Tables "
                        "outside this agent's scope come back with a message instead "
                        "of a schema."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Table names to describe. Ask for all of them at once.",
                            },
                        },
                        "required": ["tables"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_filter",
                    "description": (
                        "Return how each column may be filtered: an exact comparison, a "
                        "date range, a text match, or a vector distance. Call this for "
                        "every column you intend to put in a WHERE clause. It decides "
                        "the retrieval strategy for you - do not infer one from the "
                        "column's name or type."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns you intend to filter on.",
                            },
                            "table_name": {
                                "type": "string",
                                "description": "The table those columns belong to.",
                            },
                        },
                        "required": ["columns", "table_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_lsit_values",
                    "description": (
                        "Return the distinct values stored in one column. Use it before "
                        "filtering on a column whose values you have not seen, so you "
                        "match what is actually stored rather than what the question "
                        "called it. Above 20 distinct values it returns a sample and a "
                        "count instead of the full list."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name."},
                            "column": {"type": "string", "description": "Column name."},
                        },
                        "required": ["table", "column"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "embed_query_tool",
                    "description": (
                        "Turn a piece of text into a vector and return a short token "
                        "standing for it. Required before any semantic filter: pass the "
                        "token as a parameter to db_execute and it is resolved to the "
                        "vector at query time. Embed each concept separately rather than "
                        "the whole question at once, and never paste a vector into SQL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The phrase to embed, e.g. one column's search term.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_execute",
                    "description": (
                        "Run one read-only SELECT and return a page of rows. Rules the "
                        "call is rejected for: it must be a SELECT over allowed tables "
                        "only, with no stacked statements; the text must contain both "
                        "'LIMIT $n' and 'OFFSET $n'; params must end with the limit and "
                        "the offset in that order; limit is at most 100 and offset at "
                        f"most {MAX_OFFSET}. Any parameter that is a token from "
                        "embed_query_tool is resolved to its vector before the query "
                        "runs. Combine every filter in a single WHERE clause - numeric "
                        "and date comparisons alongside vector distances - rather than "
                        "querying twice and intersecting the results yourself."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The SELECT, with positional placeholders $1, $2, ... "
                                    "and ending in LIMIT $n OFFSET $n."
                                ),
                            },
                            "params": {
                                "type": "array",
                                "items": {},
                                "description": (
                                    "Values for the placeholders, in order, ending with "
                                    "[..., limit, offset]. A vector token may be used "
                                    "wherever a vector is expected."
                                ),
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Starting row. Use 0 - paging is done with the returned cursor.",
                            },
                            "count_query": {
                                "type": "string",
                                "description": (
                                    "A SELECT count(*) over the same tables and the same "
                                    "WHERE clause, without LIMIT or OFFSET. It is what "
                                    "makes has_more meaningful."
                                ),
                            },
                            "count_params": {
                                "type": "array",
                                "items": {},
                                "description": "Values for count_query, without the limit and offset.",
                            },
                        },
                        "required": ["query", "params", "offset", "count_query", "count_params"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_next_cursor",
                    "description": (
                        "Fetch the next page of a previous db_execute. Pass the "
                        "next_cursor value exactly as returned - it already carries the "
                        "query and the offset, so do not rebuild the query or send it "
                        "again. Use this whenever has_more was true and you still need "
                        "more rows."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cursor": {
                                "type": "string",
                                "description": "The next_cursor from the previous result, unmodified.",
                            },
                        },
                        "required": ["cursor"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_table_records",
                    "description": (
                        "Return a few whole rows from one table that read as closest in "
                        "meaning to a phrase, as text. Use it to see what a table's rows "
                        "actually look like before writing a query against it. It is not "
                        "a substitute for db_execute: it cannot filter, count, or page."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The phrase to match rows against."},
                            "table_name": {"type": "string", "description": "Table to sample from."},
                            "mx": {
                                "type": "integer",
                                "description": "How many rows to return. Clamped to between 3 and 6.",
                            },
                        },
                        "required": ["query", "table_name"],
                    },
                },
            },
        ]


    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Invoke the named tool with the given arguments and return its result.

        Raises:
            UnknownToolError: if tool_name is not a recognized tool.
            ToolExecutionError: if the tool itself fails while executing.
        """

        handler = self._handlers.get(tool_name)

        if handler is None:
            raise UnknownToolError(f"Unknown tool: {tool_name}")

        try:
            return await handler(**args)
        except Exception as e:
            raise ToolExecutionError(f"Error {e} while executing tool: {tool_name}") from e


    async def _get_table_schema(self, tables: list[str]) -> Any:
            
            schemas = {}
     
            for table in tables:
                table_key = table.lower()
                if table_key not in self._allowed_tables:
                    schemas[table_key] = "Not allowed to use this table, or error with table name"
                else:
                    schemas[table_key] = self._schema[table_key]

            return schemas


    async def _get_filter(self, columns: list[str], table_name: str) -> Any:

            table_key = table_name.lower()

            if table_key not in self._allowed_tables:
                 return "Not allowed to use this table, or error with table name"
            
            filters = {}
   
            for col in columns:
                filters[col] = self._filters[table_key].get(col, "column not found")
 

            return filters


    async def _get_lsit_values(self, table: str, column: str) -> Any:
        table_id = validate_identifier((table or "").lower())
        if table_id not in self._allowed_tables:
            return {"error": f"Unknown table: {table}. use one of the tables in the schema only {sorted(self._allowed_tables)}"}

        column_id = validate_identifier((column or "").lower())
        real_columns = _extract_column_names(self._schema.get(table_id, {}))
        if column_id not in real_columns:
            return {"error": f"Unknown column: {column}. use one of the columns in table {table_id} schema only"}

        sql = f"SELECT DISTINCT {column_id} FROM {table_id}"
        rows = await self._db.fetch(sql)

        if not rows:
            return f"all values in column {column_id} is Null"

        values = [r[column_id] for r in rows]
        if len(values) > 20:
            return f"we have multy value {len(values)}, here are some of them {values[:10]}"

        return f"in column {column_id} in table {table_id} we have this list {values}"
            

    async def _db_execute(self, query: str, params: list, offset: int,
                       count_query: str, count_params: list,
                       cursor: str = "") -> Any:
        if cursor:
            state = _decode_cursor(cursor)
            offset = state["offset"]
            query = state["query"]
            params = list(state.get("resolved_params", params))
            if params:
                params[-1] = offset
            count_query = state.get("count_query", count_query)
            count_params = state.get("count_params", count_params)
        else:
            offset = 0

        query_err = validate_readonly_query(query, self._allowed_tables)
        if query_err:
            return {"error": query_err}
        count_query_err = validate_readonly_query(count_query, self._allowed_tables)
        if count_query_err:
            return {"error": count_query_err}

        query_check = query.lower()
        if "limit $" not in query_check:
            return {"error": "limit $n should be in the query, params = [..., limit, offset]"}
        if "offset $" not in query_check:
            return {"error": "offset $n should be in the query, params = [..., limit, offset]"}

        if len(params) >= 2:
            if int(params[-2]) > 100:
                return {"error": "limit should be less than 100"}
            if int(params[-1]) > MAX_OFFSET:
                return {"error": f"offset should be less than {MAX_OFFSET}"}
        else:
            return {"error": "params must include limit and offset as the last two values"}

        resolved_params = []
        for p in params:
            if isinstance(p, str) and p.startswith("vec_"):
                p = await self._cache.get(p)
            resolved_params.append(p)

        resolved_count_params = []
        for p in count_params:
            if isinstance(p, str) and p.startswith("vec_"):
                p = await self._cache.get(p)
            resolved_count_params.append(p)

        data = await self._db.fetch(query, *resolved_params)
        row_count = len(data)

        total_rows = await self._db.fetch(count_query, *resolved_count_params)
        total = total_rows[0][list(total_rows[0].keys())[0]] if total_rows else 0

        next_offset = offset + row_count
        has_more = next_offset < total
        next_cursor = ""
        if has_more:
            next_cursor = _encode_cursor({
                "offset": next_offset,
                "resolved_params": resolved_params,
                "query": query,
                "count_query": count_query,
                "count_params": resolved_count_params,
            })

        return {"rows": data, "row_count": total, "has_more": has_more, "next_cursor": next_cursor}


    async def _get_table_records(self, query: str, table_name: str, mx: int = 5) -> Any:
        table_key = table_name.lower()
        if table_key not in self._allowed_tables:
            return {"error": f"Unknown table: {table_name}. use one of {sorted(self._allowed_tables)}"}

        mx = max(3, min(int(mx), 6))
        vec = await self._embeddings.embed(query)

        sql = f"""
            SELECT row_txt
            FROM {validate_identifier(table_key)}
            ORDER BY embedding {self._dist_op} $1::vector
            LIMIT $2
        """
        rows = await self._db.fetch(sql, vec, mx)
        items = [r["row_txt"] for r in rows]
        return {"rows": items, "row_count": len(items), "has_more": False, "next_cursor": ""}


    async def _embed_query_tool(self, query: str) -> Any:
        embed = await self._embeddings.embed(query)
        token = f"vec_{uuid.uuid4().hex[:12]}"
        await self._cache.set(token, embed, ttl=self._vector_ttl_seconds)
        return {"vector_token": token}


    async def _execute_next_cursor(self, cursor: str) -> Any:
        return await self._db_execute(
            query="", params=[], offset=0,
            count_query="", count_params=[], cursor=cursor,
        )


                    

        
