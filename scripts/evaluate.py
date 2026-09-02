"""Ask the system questions whose answers are facts, and count how often it is right.

Every change made to the prompts so far was judged on a single run of three
questions, read by eye. That is how a fix that helps and a model that varied
came to look the same: asked for Arabic novels, this system answered
correctly once in three attempts, and asked the identical question about
English novels, once in three - with nothing changed between them.

So: questions with answers the database can be asked for directly, each run
several times, reported as a fraction. The expected value comes from running
the SQL, not from a number written down here, so the questions stay right
when the seed data changes.

    python3 scripts/evaluate.py                 # 3 attempts each
    python3 scripts/evaluate.py --runs 5
    python3 scripts/evaluate.py --only arabic_novels_under_300

Costs one full turn per attempt - about seven model calls - so it is not
free. It is cheaper than deciding a prompt change worked because one answer
came out right.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ORCHESTRATOR = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# Where each sub-agent is published on this machine, from the compose file.
# The orchestrator reaches them by service name on the compose network, which
# does not resolve from here - so checking them needs the published port, and
# an agent without one simply is not checked.
SUB_AGENT_PORTS = {"catalog": 8001, "circulation": 8002}
QUESTIONS = Path(__file__).resolve().parent.parent / "tests" / "eval" / "questions.json"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def numbers_in(text: str) -> set[int]:
    """Every integer an answer contains, in either set of digits.

    Answers arrive in Arabic or English and the digits follow, so a check
    that only reads one of them marks correct answers wrong.
    """
    plain = text.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "")
    return {int(n) for n in re.findall(r"\b\d+\b", plain)}


async def expected_values(questions: list[dict]) -> dict:
    """Run each question's SQL, so the expectation is the database's."""
    import asyncpg

    connection = await asyncpg.connect(
        host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", "55432")),
        user=os.getenv("PGUSER", "dev"), password=os.getenv("PGPASSWORD", "dev"),
        database=os.getenv("PGDATABASE", "library_dev"), timeout=5,
    )
    try:
        answers = {}
        for question in questions:
            answers[question["id"]] = {
                "right": await connection.fetchval(question["sql"]),
                "wrong": {
                    await connection.fetchval(sql): why
                    for sql, why in (question.get("wrong") or {}).items()
                },
            }
        return answers
    finally:
        await connection.close()


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def preflight() -> str:
    """Refuse to score anything until the system is actually answering.

    This exists because the first real run of this script reported 0/18 with
    every answer "wrong (said [404, 500])" - which is not a model getting
    questions wrong, it is HTTP status codes being read as numbers in an
    answer. The orchestrator was returning 500 because a sub-agent was
    returning 404, and the harness dutifully scored six questions against a
    system that was not running.

    A measurement tool that reports a broken deployment as a bad model is
    worse than no measurement tool, so this runs first and stops.

    Returns the model name on success.
    """
    base = ORCHESTRATOR.rstrip("/")

    try:
        health = get_json(f"{base}/health")
    except Exception as e:
        sys.exit(
            f"The orchestrator at {base} is not answering /health: {e}\n\n"
            "  docker compose -f deploy/docker-compose.yml up -d --build\n"
            "  curl localhost:8000/health\n"
        )

    if health.get("kind") != "orchestrator":
        sys.exit(
            f"{base} is serving the {health.get('agent')!r} agent, not the "
            "orchestrator, so it has no /ask. AGENT_KEY is set on the wrong "
            "container.\n"
        )

    if "model" not in health:
        # /health has reported the model since the commit that added this
        # check, so its absence means the container is running an older image.
        # --force-recreate recreates a container from the image it already
        # has; only --build makes a new one.
        sys.exit(
            "The orchestrator is running an older image than this checkout - "
            "its /health does not report a model.\n\n"
            "  docker compose -f deploy/docker-compose.yml up -d --build\n\n"
            "--force-recreate alone recreates the container from the image it "
            "already has; --build is what rebuilds it.\n"
        )

    # Every agent it routes to, before asking it six questions that delegate.
    # A sub-agent that is down turns into a 500 at the orchestrator, and the
    # message that comes back names neither the agent nor the reason.
    unreachable = check_sub_agents(health.get("routes_to", []))
    if unreachable:
        sys.exit(
            "The orchestrator is up but cannot reach: "
            + ", ".join(unreachable) + "\n\n"
            "Every question here delegates, so all of them would fail. The "
            "orchestrator turns this into a 500 whose message names neither "
            "the agent nor the reason.\n\n"
            "  docker compose -f deploy/docker-compose.yml ps\n"
            "  docker compose -f deploy/docker-compose.yml logs agent-catalog\n"
        )

    checked = [n for n in health.get("routes_to", []) if n in SUB_AGENT_PORTS]
    print(f"orchestrator ok, agents reachable: {', '.join(checked) or 'none published'}")

    model = health.get("model") or "unset"
    return f"{model} (thinking)" if health.get("thinking") else model


def check_sub_agents(names: list[str]) -> list[str]:
    """Which registered agents are not answering their own /health.

    Ports come from the compose file - the orchestrator reaches them by
    service name on the compose network, which does not resolve from here.
    An agent whose port is not published cannot be checked and is not
    reported as broken.
    """
    unreachable = []
    for index, name in enumerate(names):
        port = SUB_AGENT_PORTS.get(name)
        if port is None:
            continue
        try:
            health = get_json(f"http://localhost:{port}/health", timeout=5)
        except Exception as e:
            unreachable.append(f"{name} (localhost:{port}: {e})")
            continue
        if health.get("kind") != "sub_agent":
            unreachable.append(
                f"{name} (localhost:{port} is a {health.get('kind')}, not a "
                "sub_agent - AGENT_KEY is unset on that container)")
    return unreachable


def ask(question: str, session: str) -> tuple[str, list, float]:
    payload = json.dumps({"question": question, "session_id": session}).encode()
    request = urllib.request.Request(
        f"{ORCHESTRATOR.rstrip('/')}/ask", data=payload,
        headers={"content-type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as e:
        # None as the answer, not a string. A string went through numbers_in()
        # and "[HTTP 500] ... 404 ..." scored as the model answering 404 - so
        # six questions were reported wrong against a system that never
        # answered any of them. An error is a third outcome, not a wrong one.
        return None, [{"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}], \
            time.time() - started
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach the orchestrator at {ORCHESTRATOR}: {e}")

    return body["answer"], body.get("delegated", []), time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3,
                        help="attempts per question (default 3)")
    parser.add_argument("--only", help="run one question by id")
    parser.add_argument("--show-delegated", action="store_true",
                        help="print what the orchestrator asked each agent")
    parser.add_argument("--label", help="a note to print with the score, for "
                                        "keeping two runs apart")
    args = parser.parse_args()

    catalogue = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.only:
        catalogue = [q for q in catalogue if q["id"] == args.only]
        if not catalogue:
            sys.exit(f"No question with id {args.only!r}")

    try:
        truth = asyncio.run(expected_values(catalogue))
    except Exception as e:
        sys.exit(f"Cannot read the expected answers from Postgres: {e}")

    model = preflight()
    print(f"{len(catalogue)} question(s), {args.runs} attempt(s) each, "
          f"against {ORCHESTRATOR}")
    print(f"model: {model}"
          + (f"   [{args.label}]" if args.label else "") + "\n")

    totals = [0, 0]
    errors = 0
    for question in catalogue:
        right = truth[question["id"]]["right"]
        wrong = truth[question["id"]]["wrong"]

        print(f"── {question['id']}  (expects {right})")
        print(f"   {question['ask']}")
        print(f"   tests: {question['tests']}")

        correct = 0
        for attempt in range(args.runs):
            # A fresh session each time. Sharing one would make attempt three
            # a different question from attempt one, which is the opposite of
            # what is being measured.
            answer, delegated, elapsed = ask(
                question["ask"], f"eval-{question['id']}-{attempt}-{int(time.time())}"
            )

            if answer is None:
                # Counted apart from correct and wrong. A run with errors in it
                # has not measured the model at all, and averaging them into a
                # score would hide that.
                errors += 1
                print(f"     {attempt + 1}. {'  ERROR':<10}{elapsed:>6.1f}s  "
                      f"{delegated[0]['error']}")
                continue

            found = numbers_in(answer)

            if right in found:
                verdict, note = "correct", ""
            elif found & set(wrong):
                got = next(iter(found & set(wrong)))
                verdict, note = "  wrong", f"({got}: {wrong[got]})"
            elif not found:
                verdict, note = "no number", f"- {answer[:60].strip()}"
            else:
                verdict, note = "  wrong", f"(said {sorted(found)})"

            correct += verdict == "correct"
            print(f"     {attempt + 1}. {verdict:<10}{elapsed:>6.1f}s  {note}")
            if args.show_delegated:
                for step in delegated:
                    print(f"        → {step['agent']}: {step['question']}")

        totals[0] += correct
        totals[1] += args.runs
        print(f"   {correct}/{args.runs}\n")

    score, out_of = totals
    print("─" * 60)
    print(f"{score}/{out_of} correct   model: {model}"
          + (f"   [{args.label}]" if args.label else ""))

    if errors:
        print(
            f"\n{errors} of those {out_of} attempts errored rather than "
            "answering, so this score does not measure the model.\n"
            "The orchestrator was reachable when this started, so something "
            "failed mid-run - read what:\n\n"
            "    docker compose -f deploy/docker-compose.yml logs --tail=100\n"
        )
        return
    if score < out_of:
        print(
            "\nA question that is sometimes right and sometimes wrong is not a\n"
            "prompt that needs rewording - it is the same prompt producing\n"
            "different SQL.\n\n"
            "To compare a stronger model, change one variable and run this again:\n\n"
            "    QWEN_MODEL=qwen-max docker compose -f deploy/docker-compose.yml \\\n"
            "        up -d --force-recreate\n"
            "    python3 scripts/evaluate.py --runs 3 --label qwen-max\n\n"
            "The model printed above comes from /health, so a container that did\n"
            "not restart is visible rather than scored as the new model.\n\n"
            "For a wrong answer, read the trace rather than guessing:\n\n"
            "    python3 scripts/show_history.py --agent orchestrator --turns 3\n"
        )


if __name__ == "__main__":
    main()
