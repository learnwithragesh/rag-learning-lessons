"""
Lesson 5: The full RAG pipeline, end to end.

Part 1: look at the exact prompt the LLM receives
Part 2: answer a set of questions (answerable, unanswerable, and the E-417 trap)

Run from the course root:   python3 05-full-pipeline/lesson.py
Ask your own questions:     python3 05-full-pipeline/ask.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.rag import SYSTEM_PROMPT, answer
from common.vector_store import DEFAULT_INDEX_DIR, VectorStore


def load_store() -> VectorStore:
    if not (DEFAULT_INDEX_DIR / "vectors.npy").exists():
        raise SystemExit("No index found. Run Lesson 4 first:  python3 04-vector-store/lesson.py")
    return VectorStore.load()


def part1_show_prompt(store):
    print("=" * 80)
    print("PART 1: the exact prompt sent to the LLM")
    print("=" * 80)
    result = answer("How long does charging take and when does the robot start charging?",
                    store.search, k=2)
    print("----- SYSTEM -----")
    print(SYSTEM_PROMPT)
    print("----- USER -----")
    print(result["prompt"])
    print("----- LLM ANSWER -----")
    print(result["answer"], "\n")


def part2_questions(store):
    print("=" * 80)
    print("PART 2: questions end to end (k=4)")
    print("=" * 80)
    questions = [
        ("How much is the home office stipend and what can I buy with it?", "answerable"),
        ("Can I carry over unused annual leave?", "answerable"),
        ("Who do I ask for a fully remote arrangement?", "answerable"),
        ("How often must the lidar window be cleaned, and with what?", "answerable"),
        ("What is the CEO's salary?", "NOT in the documents"),
        ("What is the capital of France?", "general knowledge, NOT in the documents"),
        ("The robot shows E-417. What should I do?", "retrieval trap from Lesson 4"),
    ]
    for q, kind in questions:
        r = answer(q, store.search, k=4)
        print(f"\nQ ({kind}): {q}")
        print(f"A: {r['answer']}")
        print("   retrieved:", ", ".join(f"[{i}] {c['section']}" for i, c in enumerate(r["chunks"], 1)))


if __name__ == "__main__":
    store = load_store()
    part1_show_prompt(store)
    part2_questions(store)
