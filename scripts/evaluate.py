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
        return f"[HTTP {e.code}] {e.read().decode()[:120]}", [], time.time() - started
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

    print(f"{len(catalogue)} question(s), {args.runs} attempt(s) each, "
          f"against {ORCHESTRATOR}\n")

    totals = [0, 0]
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
    print(f"{score}/{out_of} correct")
    if score < out_of:
        print(
            "\nA question that is sometimes right and sometimes wrong is not a\n"
            "prompt that needs rewording - it is the same prompt producing\n"
            "different SQL. Lower QWEN_TEMPERATURE, or use a stronger model for\n"
            "the agent that got it wrong, and run this again."
        )


if __name__ == "__main__":
    main()
