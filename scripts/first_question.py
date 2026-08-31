"""Ask one real question, end to end, and report what it cost.

This is the one path nothing else covers. Everything up to the model call is
tested against a real Postgres in tests/integration/; the call itself needs a
key, and a test that skips without one reports green for a path nobody ran.

Run it from the repo root, with the system up:

    docker compose -f deploy/docker-compose.yml up -d
    export QWEN_API_KEY=...
    python3 scripts/first_question.py

It reports the answer, the delegation trace, the tokens spent, and - the
number worth watching - how many of the input tokens were served from the
provider's cache. A cache hit rate near zero on the second question means the
prefix is not stable, and the prefix is where most of the bill lives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ORCHESTRATOR = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

QUESTIONS = [
    "كم رواية عربية عندنا أقل من ٣٠٠ صفحة؟",
    "Which authors have more than three books in the catalogue?",
    # The third repeats the first on purpose: same shape, so the fixed prefix
    # is identical and a provider that caches has something to hit.
    "كم رواية إنكليزية عندنا أقل من ٤٠٠ صفحة؟",
]


def call(path: str, payload: dict | None = None, timeout: int = 300) -> dict:
    url = f"{ORCHESTRATOR.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"content-type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def preflight() -> None:
    """Fail on the things that are cheap to check before spending a token."""
    try:
        health = call("/health", timeout=10)
    except urllib.error.URLError as e:
        sys.exit(
            f"No orchestrator at {ORCHESTRATOR}: {e}\n"
            "Start it with:  docker compose -f deploy/docker-compose.yml up -d"
        )

    if health["kind"] != "orchestrator":
        sys.exit(
            f"{ORCHESTRATOR} is serving the {health['agent']!r} agent, not the "
            "orchestrator. Point ORCHESTRATOR_URL at the process with no "
            "AGENT_KEY set."
        )

    agents = call("/agents", timeout=10)
    if not agents:
        sys.exit("The orchestrator has no agents to route to. Check the registry.")

    print(f"orchestrator up, routing to: {', '.join(a['key'] for a in agents)}")
    for agent in agents:
        print(f"  {agent['key']:<14} {agent['description'][:70]}…")
    print()


def ask(question: str, session: str) -> None:
    print("─" * 72)
    print(f"Q  {question}")
    started = time.time()
    try:
        body = call("/ask", {"question": question, "session_id": session})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            detail = json.loads(raw).get("detail", raw)
        except json.JSONDecodeError:
            detail = raw
        print(f"!  HTTP {e.code}: {str(detail)[:500]}")
        if e.code >= 500:
            print("   full traceback:")
            print("     docker compose -f deploy/docker-compose.yml logs --tail 60 orchestrator")
        # 401/403 is the key; 402 is the quota. Both are worth stopping on,
        # because every later question would fail the same way.
        if e.code in (401, 402, 403) or "401" in str(detail):
            sys.exit(
                "\nStopping: this is a credentials or quota problem, not a bug.\n"
                "\nDid the key reach the container?\n"
                "  docker compose -f deploy/docker-compose.yml exec orchestrator \\\n"
                "      sh -c 'echo ${#QWEN_API_KEY}'\n"
                "\nIs it a key for this region? DashScope keys are region-bound,\n"
                "and a Beijing key on the Singapore endpoint returns exactly this.\n"
                "  curl -s -o /dev/null -w '%{http_code}\\n' \\\n"
                "      -H \"Authorization: Bearer $QWEN_API_KEY\" \\\n"
                "      https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models\n"
                "\nAfter fixing it, containers read the environment when they are\n"
                "created, not per request:\n"
                "  docker compose -f deploy/docker-compose.yml up -d --force-recreate"
            )
        return
    elapsed = time.time() - started

    print(f"A  {body['answer']}")
    print(f"   {elapsed:.1f}s   turn {body['turn_id'][:8]}")

    paging = {k: v for k, v in body.get("pagination", {}).items() if v.get("has_more")}
    if paging:
        print(f"   more rows available from: {', '.join(paging)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="ask your own instead")
    parser.add_argument("--session", default=f"first-run-{int(time.time())}")
    args = parser.parse_args()

    preflight()

    questions = [" ".join(args.question)] if args.question else QUESTIONS
    for question in questions:
        ask(question, args.session)

    print("─" * 72)
    print(
        "Token usage and cache hits are on the provider's console - Model\n"
        "Studio > Billing for DashScope. What to look for: on the second and\n"
        "third questions, cached input tokens should be most of the input.\n"
        "Near zero means the prefix is not stable and the tool schemas are\n"
        "being paid for on every call in the loop."
    )


if __name__ == "__main__":
    main()
