"""An OpenAI-compatible endpoint that returns a scripted conversation.

There is a gap no other test reaches. Unit tests stop at the port; the
integration tests stop at the last step before the model is called; and a
real model needs a key, costs money, and answers differently every run. What
sits in that gap is the agent loop itself - a model asks for a tool, the tool
runs against a real database, the result goes back, the model asks for
another - and every bug found by running this system for the first time lived
exactly there or just under it.

So: a server that speaks the OpenAI chat-completions shape and replies from a
script. It decides nothing. That is the point - what is being tested is the
plumbing between the model and the database, and a fake that made choices
would be testing the fake.

It also records every request, so the things worth asserting about a loop are
observable afterwards: which tools were offered, whether the system prompt
stayed byte-identical across calls (the prefix-caching property), and what the
model was actually shown.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


def tool_call(name: str, **arguments) -> dict:
    """One step of a script: ask for a tool."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def says(text: str) -> dict:
    """The last step of a script: answer."""
    return {"role": "assistant", "content": text}


class ScriptedModel:
    """A running fake endpoint. `base_url` goes straight into QWEN_API_URL.

    `script` is a list of replies, returned in order. A shorter script than
    the loop needs means the loop asked one more time than expected, which is
    itself worth failing on rather than looping forever - so it raises
    instead of repeating the last reply.
    """

    def __init__(self, script: list[dict], embedding_dim: int = 1024):
        self.script = list(script)
        self.embedding_dim = embedding_dim
        self.requests: list[dict] = []
        self._step = 0
        self._server = None
        self._thread = None

    # -- what the tests ask afterwards -----------------------------------

    @property
    def tools_offered(self) -> set[str]:
        if not self.requests:
            return set()
        return {t["function"]["name"] for t in self.requests[0].get("tools", [])}

    @property
    def system_prompts(self) -> list[str]:
        """The first message of every request. Identical across calls is the
        property prefix caching depends on."""
        return [
            r["messages"][0]["content"] for r in self.requests
            if r.get("messages") and r["messages"][0]["role"] == "system"
        ]

    def tool_results_seen(self) -> list[str]:
        """Every tool result the model was shown, once, in order.

        Read from the last request only. Each request carries the whole
        conversation so far, so collecting across all of them returns the
        early results once per remaining call - which looks like the loop
        running the same tool repeatedly.
        """
        if not self.requests:
            return []
        return [
            m["content"] for m in self.requests[-1].get("messages", [])
            if m.get("role") == "tool"
        ]

    # -- the server ------------------------------------------------------

    def _reply(self, body: dict) -> dict:
        self.requests.append(body)
        if self._step >= len(self.script):
            raise AssertionError(
                f"The loop made {self._step + 1} model calls but the script has "
                f"{len(self.script)}. Either the loop is not stopping, or the "
                "script is missing a step."
            )
        message = self.script[self._step]
        self._step += 1
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "scripted"),
            "choices": [{
                "index": 0, "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }],
            # A plausible usage block, including the cached-token shape, so
            # anything that reads those numbers is exercised rather than
            # handed zeros. The second call onward reports most of the prompt
            # as reused, which is what a stable prefix looks like.
            "usage": {
                "prompt_tokens": 1200 + 300 * (self._step - 1),
                "completion_tokens": 40,
                "total_tokens": 1240 + 300 * (self._step - 1),
                "prompt_tokens_details": {
                    "cached_tokens": 0 if self._step == 1 else 1200,
                },
            },
        }

    def _embedding(self, body: dict) -> dict:
        # Deterministic and not all-zero: pgvector's cosine distance is
        # undefined for a zero vector, and a test that silently compared
        # against NaN would pass for the wrong reason.
        vector = [0.1] * self.embedding_dim
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vector}],
            "model": body.get("model", "scripted-embed"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def start(self) -> "ScriptedModel":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence the default stderr log
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    if self.path.endswith("/embeddings"):
                        payload, status = fake._embedding(body), 200
                    else:
                        payload, status = fake._reply(body), 200
                except AssertionError as e:
                    payload, status = {"error": {"message": str(e)}}, 500

                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class ServedApp:
    """A FastAPI app on a real port, in a background thread.

    For the one test where two runtimes have to talk to each other. Nesting
    two TestClients does not work - each runs its own event loop, and the
    pools opened in one are closed from the other - and mocking the delegate's
    HTTP client would skip the thing being tested, which is that the
    orchestrator's request and the sub-agent's response actually fit.

    A real server on a real socket has neither problem, and is closer to the
    deployment than either shortcut.
    """

    def __init__(self, app):
        self.app = app
        self.startup_error: BaseException | None = None
        self._server = None
        self._thread = None

    def __enter__(self) -> str:
        import uvicorn

        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="error")
        self._server = uvicorn.Server(config)

        def serve():
            # The app's lifespan runs in here, so a database that is not
            # there fails on this thread. Kept rather than dropped: the
            # caller decides whether an unreachable service is a skip or a
            # failure, and it cannot decide about an exception it never sees.
            try:
                self._server.run()
            except BaseException as e:
                self.startup_error = e

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()

        deadline = time.time() + 30
        while not self._server.started and time.time() < deadline:
            if not self._thread.is_alive():
                if self.startup_error is not None:
                    raise self.startup_error
                raise RuntimeError(
                    "the sub-agent server stopped before it started, with no error"
                )
            time.sleep(0.05)
        if not self._server.started:
            raise TimeoutError("the sub-agent server did not start")

        host, port = self._server.servers[0].sockets[0].getsockname()[:2]
        return f"http://{host}:{port}"

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)
