"""Which models can this key actually use, and do they support tool calling?

Two questions, and the second is the one that bites. `/v1/models` lists what
the provider offers, not what an account may call - a model can appear there
and answer `403 AccessDenied.Unpurchased`. And a model that answers happily
may still ignore the `tools` parameter and reply in prose, which this system
reads as a final answer: the orchestrator would tell the user "I'll look that
up in the catalogue" and stop.

So each candidate gets one real request with one real tool, and is judged on
whether a tool call comes back.

    python3 scripts/check_model.py                    # a list of likely ones
    python3 scripts/check_model.py qwen-plus qwen-max # or your own

Costs a few tokens per model - one short call each.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY", "")

# Read the same way config.py reads it, so this probes what the service sends.
MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "8192"))

# Ordered cheapest-first, because the cheapest model that can call a tool is
# usually the right one for the sub-agents - they do narrow work with the
# strategy already handed to them by get_filter.
CANDIDATES = [
    "qwen-flash", "qwen-turbo", "qwen3.5-flash", "qwen3.7-flash",
    "qwen-plus", "qwen3.5-plus", "qwen3-14b", "qwen3-32b", "qwen-max",
]

# Deliberately a question the model cannot answer without the tool, so a
# model that merely *can* call tools has a reason to.
PROBE = {
    "messages": [
        {"role": "system", "content": "You look things up. Never guess a number."},
        {"role": "user", "content": "How many books are in the catalogue?"},
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "count_books",
            "description": "Return the number of books in the catalogue.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }],
    # The value the service will actually send, not a small safe one.
    #
    # This was 64, which every model accepts - so the check passed for
    # qwen-max while the service, sending QWEN_MAX_TOKENS, was refused with
    #
    #     400 InternalError.Algo.InvalidParameter:
    #     Range of max_tokens should be [1, 8192]
    #
    # on every question. A check that probes a configuration the system does
    # not use answers a question nobody asked; the point is to find out
    # whether *this deployment* can call *this model*.
    "max_tokens": MAX_TOKENS,
}


def ask(model: str) -> tuple[str, str]:
    """Returns (verdict, detail)."""
    body = json.dumps({**PROBE, "model": model}).encode()
    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions", data=body,
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            message = json.loads(raw)["error"]["message"]
        except Exception:
            message = raw[:120]
        if e.code == 400 and "max_tokens" in message:
            # The account has the model; the request shape is wrong. A
            # different fix entirely, and "no access" would send someone to
            # Model Studio to enable something already enabled.
            return "max_tokens", f"{message[:70]} (sent {MAX_TOKENS})"
        return "no access", f"{e.code} {message[:90]}"
    except urllib.error.URLError as e:
        return "unreachable", str(e)[:90]

    message = payload["choices"][0]["message"]
    if message.get("tool_calls"):
        return "TOOLS OK", message["tool_calls"][0]["function"]["name"]

    # The failure that costs an afternoon: the call succeeded, the model
    # answered in prose, and nothing anywhere says tool calling did not
    # happen.
    return "no tools", (message.get("content") or "")[:80].replace("\n", " ")


EMBED_URL = os.getenv("EMBED_API_URL", "") or BASE_URL
EMBED_KEY = os.getenv("EMBED_API_KEY", "") or API_KEY
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


def check_embedding(model: str) -> None:
    """Whether the embedding endpoint answers, and with how many dimensions.

    The dimension is the part worth checking rather than assuming. It is
    fixed in the schema as `vector(N)`, so a model that returns a different
    width is a migration and a re-embedding of every row - not a restart -
    and the failure otherwise arrives as a Postgres error in the middle of
    someone's question.
    """
    body = json.dumps({"model": model, "input": "a short sentence to embed"}).encode()
    request = urllib.request.Request(
        f"{EMBED_URL.rstrip('/')}/embeddings", data=body,
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {EMBED_KEY}"},
    )
    print(f"\nembeddings: {EMBED_URL}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            detail = json.loads(raw)["error"]["message"]
        except Exception:
            detail = raw[:140]
        print(f"  {model:<28}no access    {e.code} {detail[:80]}")
        return
    except urllib.error.URLError as e:
        print(f"  {model:<28}unreachable  {str(e)[:80]}")
        return

    dimensions = len(payload["data"][0]["embedding"])
    if dimensions == EMBEDDING_DIM:
        print(f"  {model:<28}OK           {dimensions} dimensions")
    else:
        print(
            f"  {model:<28}WRONG WIDTH  {dimensions} dimensions, "
            f"schema expects {EMBEDDING_DIM}\n"
            f"  {'':<28}             this is a migration and a re-embedding "
            "of every row,\n"
            f"  {'':<28}             not a change of variable"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=None)
    parser.add_argument("--embedding", metavar="MODEL",
                        help="also check an embedding model, and its width")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("QWEN_API_KEY is not set in this shell.")

    print(f"endpoint: {BASE_URL}")
    print(f"max_tokens: {MAX_TOKENS}\n")
    width = max(len(m) for m in (args.models or CANDIDATES)) + 2
    print(f"{'model':<{width}}{'verdict':<13}detail")
    print("-" * 78)

    usable = []
    for model in (args.models or CANDIDATES):
        verdict, detail = ask(model)
        if verdict == "TOOLS OK":
            usable.append(model)
        print(f"{model:<{width}}{verdict:<13}{detail}")

    if args.embedding:
        check_embedding(args.embedding)

    print()
    if usable:
        print("Usable, cheapest first:", ", ".join(usable))
        print(f"\n  export QWEN_MODEL={usable[0]}")
        print("  docker compose -f deploy/docker-compose.yml up -d --force-recreate")
    else:
        print(
            "None of these can call a tool with this key.\n"
            "'no access' means the model needs activating in Model Studio.\n"
            "'max_tokens' means the model is available but rejects the "
            "configured QWEN_MAX_TOKENS - lower it; qwen-max caps at 8192.\n"
            "'no tools' means it answered in prose - it cannot drive this system,\n"
            "and would fail silently rather than loudly if used."
        )


if __name__ == "__main__":
    main()
