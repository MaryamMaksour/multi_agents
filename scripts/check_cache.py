"""Measure prompt caching against the real endpoint, with this system's prompt.

Providers cache by matching the prefix of a request, and this system resends
a large fixed prefix on every call in the loop: the tool schemas alone are
about a thousand tokens, and one question is roughly seven calls. Whether
those are being paid for each time is a number, not a judgement - and it
comes back in every response, under `usage.prompt_tokens_details`.

So rather than sending anyone to a billing console: build the prompt this
system actually sends, send it twice, and print what the provider says.

    python3 scripts/check_cache.py

The second request is what matters. A cached count near the first request's
prompt total means the prefix is stable and being reused; near zero means
either the provider does not cache, or something in the prefix changes
between calls and the whole thing is being re-read.

Costs two short calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter  # noqa: E402

BASE_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY", "")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


def real_tool_schemas() -> list[dict]:
    """The schemas the agents actually send, not a stand-in for them.

    The point of the measurement is the size of this specific prefix, so a
    smaller invented one would answer a different question.
    """
    adapter = SqlToolAdapter(
        db=None, embeddings=None, cache=None, allowed_tables=["books"],
        schema={}, filters={}, lsit_values={}, dist_op="<=>", vector_ttl_seconds=1,
    )
    return adapter.get_tool_schemas()


def send(body: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            message = json.loads(raw)["error"]["message"]
        except Exception:
            message = raw[:200]
        sys.exit(f"HTTP {e.code}: {message}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach {BASE_URL}: {e}")


def cached_tokens(usage: dict) -> int | None:
    """Providers report this in more than one place, and some not at all.

    None means "not reported", which is different from zero: zero is a cache
    that exists and missed, None is a provider that says nothing either way.
    """
    for holder in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(holder) or {}
        if "cached_tokens" in details:
            return details["cached_tokens"]
    if "prompt_cache_hit_tokens" in usage:
        return usage["prompt_cache_hit_tokens"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("QWEN_API_KEY is not set in this shell.")

    tools = real_tool_schemas()
    system = (
        "You answer questions about a library's catalogue.\n\n"
        "Before writing any SQL, call the schema tool for the tables you "
        "intend to use, and the filter tool for the columns you intend to "
        "filter on. Do not guess a column name or a filter style."
    )
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "How many books are in the catalogue?"},
        ],
        "tools": tools,
        "max_tokens": 32,
    }

    schema_chars = len(json.dumps(tools))
    print(f"endpoint : {BASE_URL}")
    print(f"model    : {args.model}")
    print(f"prefix   : {len(system)} chars of system prompt + "
          f"{schema_chars} chars of tool schemas\n")

    print(f"{'':<10}{'prompt':>9}{'cached':>9}{'reused':>9}")
    print("-" * 40)

    results = []
    for label in ("first", "second"):
        usage = send(body).get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        cached = cached_tokens(usage)
        results.append((prompt, cached))
        shown = "n/a" if cached is None else str(cached)
        share = "-" if not cached or not prompt else f"{cached / prompt:.0%}"
        print(f"{label:<10}{prompt:>9}{shown:>9}{share:>9}")

    print()
    _, second_cached = results[1]
    if second_cached is None:
        print(
            "This provider does not report cached tokens, so caching cannot be\n"
            "confirmed from here. It may still be happening - implicit caching\n"
            "is usually silent - but the bill is the only evidence."
        )
    elif second_cached == 0:
        print(
            "Nothing was reused on the second, identical request. Either this\n"
            "model does not cache, or its cache needs a longer prefix than this\n"
            "probe sends. Worth re-checking against a real turn, where the\n"
            "prefix is several times larger."
        )
    else:
        prompt, _ = results[1]
        print(
            f"{second_cached} of {prompt} input tokens were reused on an identical\n"
            "second request, so the prefix is stable and the provider is\n"
            "matching it. In a real turn the reused part is larger: the loop\n"
            "resends everything before it on every call."
        )


if __name__ == "__main__":
    main()
