"""A model that needs no key, no GPU and no network.

For seeing the system work when the model is the thing you do not have. It
speaks the OpenAI chat-completions shape and drives a fixed, sensible flow -
read the schema, read the filters, run one query, answer - so every layer
below it runs for real: the tool dispatch, the SQL validator, the least-
privilege role, Postgres, the pagination, the delegation hop.

What is genuinely real in the output: the schema comes from introspection,
the filter guidance from the classifier, and the number in the answer is the
count Postgres returned. What is not real is the choosing. This decides
nothing - it follows a script, and the script is written here rather than
reasoned out.

So it is a demonstration and a development harness, never an evaluation. It
cannot tell you whether a model would pick the right agent or write correct
SQL; it can tell you that when one does, everything around it works.

    python3 scripts/demo_model.py                 # serves on 11435

    export QWEN_API_URL=http://host.docker.internal:11435/v1
    export QWEN_API_KEY=demo
    export QWEN_EMBED_MODEL=demo
    docker compose -f deploy/docker-compose.yml up -d --force-recreate
    python3 scripts/first_question.py
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

EMBEDDING_DIM = 1024

# Tool names that only a sub-agent is given. Anything else in the tool list is
# an agent name, which means this request is the orchestrator's.
SQL_TOOLS = {"get_table_schema", "get_filter", "db_execute", "embed_query_tool"}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def calls(name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def tool_results(messages: list[dict]) -> list[dict]:
    """Every tool result in this conversation, decoded, oldest first."""
    out = []
    for message in messages:
        if message.get("role") == "tool":
            try:
                out.append(json.loads(message["content"]))
            except (json.JSONDecodeError, TypeError):
                out.append({})
    return out


def sub_agent_step(messages: list[dict]) -> dict:
    """Schema, then filters, then one counting query, then an answer."""
    seen = tool_results(messages)

    if len(seen) == 0:
        return calls("get_table_schema", tables=["books"])

    if len(seen) == 1:
        return calls("get_filter", columns=["genre", "language", "page_count"],
                     table_name="books")

    if len(seen) == 2:
        # A real query against the seeded schema. The filter guidance the
        # previous step returned is what tells a real model that `genre` is
        # an enum and `page_count` takes comparisons; this just knows.
        return calls(
            "db_execute",
            query=("SELECT count(*) AS matches FROM books "
                   "WHERE language = $1 AND page_count < $2 LIMIT $3 OFFSET $4"),
            params=["English", 400, 1, 0],
            offset=0,
            count_query=("SELECT count(*) FROM books "
                         "WHERE language = $1 AND page_count < $2"),
            count_params=["English", 400],
        )

    rows = seen[2].get("rows") or [{}]
    if "error" in seen[2]:
        return assistant(f"The query was refused: {seen[2]['error']}")

    matches = rows[0].get("matches", "an unknown number of")
    return assistant(
        f"{matches} English-language books are under 400 pages. "
        "(Scripted answer - the number came from Postgres, the sentence did not.)"
    )


def orchestrator_step(messages: list[dict], agents: list[str]) -> dict:
    """Delegate once to the first registered agent, then relay its answer."""
    seen = tool_results(messages)

    if not seen:
        question = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "How many books are in the catalogue?",
        )
        return calls(
            agents[0],
            # A self-contained question, because a sub-agent has no memory of
            # this conversation. Resolving references is the orchestrator's
            # job and this is the shape of doing it.
            query=f"{question} (Answer from the catalogue only.)",
        )

    answer = seen[0].get("answer") or "The agent returned nothing."
    return assistant(f"{answer}\n\n(Relayed from the {agents[0]} agent.)")


def reply_for(body: dict) -> dict:
    offered = [t["function"]["name"] for t in body.get("tools", [])]
    messages = body.get("messages", [])

    if set(offered) & SQL_TOOLS:
        message = sub_agent_step(messages)
    elif offered:
        message = orchestrator_step(messages, offered)
    else:
        message = assistant("No tools were offered, so there is nothing to look up.")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "demo"),
        "choices": [{
            "index": 0, "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def embedding_for(body: dict) -> dict:
    """A stable vector. Not meaningful - nothing in the seeded data is
    embedded either, so there is nothing for it to be near."""
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0,
                  "embedding": [0.1] * EMBEDDING_DIM}],
        "model": body.get("model", "demo"),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.path}  {fmt % args}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = embedding_for(body) if self.path.endswith("/embeddings") else reply_for(body)

        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path.endswith("/models"):
            payload = {"object": "list",
                       "data": [{"id": "demo", "object": "model"}]}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self.send_error(404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    print(f"Scripted model on http://localhost:{args.port}/v1 - decides nothing.\n")
    print("  export QWEN_API_URL=http://host.docker.internal:%d/v1" % args.port)
    print("  export QWEN_API_KEY=demo")
    print("  export QWEN_EMBED_MODEL=demo")
    print("  unset EMBED_API_URL EMBED_API_KEY")
    print("  docker compose -f deploy/docker-compose.yml up -d --force-recreate")
    print("  python3 scripts/first_question.py\n")

    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
