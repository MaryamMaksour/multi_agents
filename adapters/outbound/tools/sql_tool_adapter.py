from domain.ports.database_port import DatabasePort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.cache_port import CachePort


from domain.exceptions import UnknownToolError, ToolExecutionError
from libs.agent_core.pgvector import to_vector_literal
from libs.agent_core.sql_validation import validate_identifier, validate_readonly_query
from libs.agent_core.logging_setup import log_event


from typing import Any
import base64
import inspect
import json
import logging
import re
import uuid
import zlib

logger = logging.getLogger(__name__)


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
                        dist_op: str, vector_ttl_seconds: int,
                        lsit_values: dict | None = None):

        self._db = db
        self._embeddings = embeddings
        self._cache = cache
        self._allowed_tables = {table.lower() for table in allowed_tables}
        self._schema = schema
        self._filters = filters
        # Accepted and ignored. Nothing has ever read it - values are read
        # from the database by _get_list_values, and the enum values the model
        # is shown come from the startup probe through `filters`. Kept as an
        # optional argument so the callers that still pass it are not a
        # breaking change, and defaulted so new ones need not know about it.
        self._lsit_values = lsit_values or {}
        self._dist_op = dist_op 
        self._vector_ttl_seconds = vector_ttl_seconds

        # dispatcher: match the tool name with the method name
        self._handlers = {
            "get_table_schema": self._get_table_schema,
            "get_filter": self._get_filter,
            "get_list_values": self._get_list_values,
            # The old misspelling, kept as an alias rather than deleted. Two
            # reasons: a conversation window from before this rename replays
            # tool calls by name, and a model that has read the word "list" a
            # million times writes it correctly often enough that accepting
            # both is worth one dictionary entry either way.
            "get_lsit_values": self._get_list_values,
            "db_execute": self._db_execute,
            "get_table_records": self._get_table_records,
            "embed_query_tool": self._embed_query_tool,
            "execute_next_cursor": self._execute_next_cursor,
        }




    def get_tool_schemas(self) -> list[dict]:
        """Describe the SQL tools in the format the LLM adapter passes on.

        Names here must match a _handlers key exactly - call_tool dispatches
        on them, so a mismatch is an UnknownToolError at runtime rather than a
        startup failure.

        This declares `get_list_values`, spelled correctly. It was
        `get_lsit_values`, and the misspelling was not cosmetic: the name is
        read by the model, which has seen "list" a great many more times than
        "lsit" and wrote the correct spelling often enough to lose a step to
        UnknownToolError each time. Both names dispatch, so a replayed
        conversation window still works.

        The descriptions carry the rules that _db_execute enforces by
        returning {"error": ...}. Stating them here turns a wasted round trip
        into a query the model gets right the first time - the error path stays
        as the guarantee, this is just the shorter route to it.

        Not everything dispatchable is declared. `get_table_records` is still
        in _handlers and is deliberately absent from this list: it selects
        `row_txt` ordered by an `embedding` column, and neither exists in any
        schema this design produces - embeddings live in `embed_<column>`
        columns beside the column they describe. Declaring it told the model
        about a tool that fails on every call. The handler is left in place
        because whether whole-row search is rebuilt or dropped is a decision
        of its own; see docs/deferred.md.
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
                    "name": "get_list_values",
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
        ]


    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Invoke the named tool with the given arguments and return its result.

        Raises:
            UnknownToolError: if tool_name is not a recognized tool.
            ToolExecutionError: if the tool itself fails while executing.
        """

        handler = self._handlers.get(tool_name)

        if handler is None:
            # A model inventing a tool is worth seeing: it usually means the
            # schema description reads like something it is not.
            log_event(logger, "tool.unknown", level=logging.WARNING,
                      tool=tool_name, known=sorted(self._handlers))
            raise UnknownToolError(f"Unknown tool: {tool_name}")

        # Argument checking is done *before* the call, not by catching
        # TypeError around it.
        #
        # `except TypeError` wrapped the whole handler body, so a TypeError
        # raised deep inside - a bug in this adapter, or an asyncpg row used
        # wrongly - was reported to the model as "wrong arguments". The model
        # would then retry the identical call, be told the same thing, and
        # spend the iteration budget on a message that was never about its
        # arguments. Binding the signature separates the two cases exactly.
        try:
            inspect.signature(handler).bind(**args)
        except TypeError as e:
            log_event(logger, "tool.bad_arguments", level=logging.WARNING,
                      tool=tool_name, given=sorted(args), detail=str(e))
            raise ToolExecutionError(
                f"{tool_name}: wrong arguments ({e}). Given: {sorted(args)}."
            ) from e

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
                # The lookup folds case, the answer does not: keys are the
                # model's own spelling so it can match them to what it asked
                # for, while the classifier keys on Postgres's lowercase
                # names. A model that writes "Genre" was told about "genre".
                filters[col] = self._filters[table_key].get(
                    (col or "").lower(), "column not found"
                )
 

            return filters


    async def _get_list_values(self, table: str, column: str) -> Any:
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

        # Every rejection below returns an error to the model rather than
        # raising, which is right - the model reads it and rewrites the query.
        # It also means a rejected query leaves no trace anywhere, so a run
        # where the model spent four of its twelve steps being told its LIMIT
        # was missing looks identical to a run where it thought for a while.
        # _rejected logs and returns in one line so that cannot drift.
        def _rejected(reason: str, **fields):
            log_event(logger, "sql.rejected", level=logging.WARNING,
                      reason=reason, sql=" ".join(query.split())[:400], **fields)
            return {"error": reason}

        query_err = validate_readonly_query(query, self._allowed_tables)
        if query_err:
            return _rejected(query_err, which="query")
        count_query_err = validate_readonly_query(count_query, self._allowed_tables)
        if count_query_err:
            return _rejected(count_query_err, which="count_query")

        query_check = query.lower()
        if "limit $" not in query_check:
            return _rejected("limit $n should be in the query, params = [..., limit, offset]")
        if "offset $" not in query_check:
            return _rejected("offset $n should be in the query, params = [..., limit, offset]")

        if len(params) >= 2:
            # int() on a value the model chose: it sends "10" as often as 10,
            # and a string that is not a number at all took the turn down with
            # a ValueError from inside the adapter rather than telling the
            # model what was wrong with its call.
            try:
                limit_value, offset_value = int(params[-2]), int(params[-1])
            except (TypeError, ValueError):
                return _rejected(
                    "the last two params must be the limit and the offset, "
                    f"as numbers. Got: {params[-2]!r}, {params[-1]!r}"
                )
            if limit_value > 100:
                return _rejected("limit should be less than 100", limit=limit_value)
            if offset_value > MAX_OFFSET:
                return _rejected(f"offset should be less than {MAX_OFFSET}",
                                 offset=offset_value)
        else:
            return _rejected("params must include limit and offset as the last two values")

        resolved_params = []
        for p in params:
            if isinstance(p, str) and p.startswith("vec_"):
                # The cache holds the embedding as a list of floats; Postgres
                # needs pgvector's text form. Converted here rather than at
                # storage time so the cached value stays a vector rather than
                # a Postgres-shaped string.
                p = to_vector_literal(await self._cache.get(p))
            resolved_params.append(p)

        resolved_count_params = []
        for p in count_params:
            if isinstance(p, str) and p.startswith("vec_"):
                p = to_vector_literal(await self._cache.get(p))
            resolved_count_params.append(p)

        data = await self._db.fetch(query, *resolved_params)
        row_count = len(data)

        total_rows = await self._db.fetch(count_query, *resolved_count_params)
        total = total_rows[0][list(total_rows[0].keys())[0]] if total_rows else 0

        next_offset = offset + row_count

        # Two conditions, and the second one is a bug fix.
        #
        # `next_offset < total` alone is wrong whenever the query is itself an
        # aggregate. "SELECT count(*) AS n FROM books WHERE ... LIMIT $1" and
        # its count_query both return 12, so total is 12, the page holds one
        # row, and has_more comes back true - inviting the model to page
        # through eleven more rows that do not exist. It happens on the most
        # common question this system gets asked.
        #
        # A short page means the end, whatever the count says. That rule holds
        # for aggregates, for a count_query whose WHERE has drifted from the
        # query's, and for a total that changed between the two statements.
        page_limit = int(params[-2]) if len(params) >= 2 else row_count
        has_more = row_count >= page_limit and next_offset < total
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
        vec = to_vector_literal(await self._embeddings.embed(query))

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


                    

        
