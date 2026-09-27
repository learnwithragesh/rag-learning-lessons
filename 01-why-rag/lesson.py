"""
Lesson 1: See the problem RAG solves.

We ask the LLM questions about a fictional company TWICE:
  A) with no context          -> it must guess (hallucination risk)
  B) with the right paragraph -> "manual RAG", it answers from evidence

Run from the course root:   python3 01-why-rag/lesson.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat

QUESTIONS = [
    {
        "question": "How many days of paid annual leave do Nimbus Robotics employees get?",
        # In Lesson 1 WE pick the relevant text by hand. Later, retrieval does this.
        "context": (
            "Every full-time employee at Nimbus Robotics receives 23 days of paid "
            "annual leave per calendar year. Up to 5 unused days may be carried over "
            "into the next year, but carried-over days expire on March 31."
        ),
    },
    {
        "question": "What does error code E-417 mean on the Zephyr-3 robot and how do I fix it?",
        "context": (
            "E-417: Lidar Calibration Drift. The lidar's measured distances disagree "
            "with known fiducial marker positions by more than 3 centimetres. First "
            "clean the lidar window. If the error persists, run the 'Auto-Calibrate' "
            "routine from the Nimbus Fleet Console, which takes about 4 minutes."
        ),
    },
    {
        "question": "Which days are Core Days at Nimbus Robotics?",
        "context": (
            "Nimbus follows a 'Core Days' hybrid model. All employees must be in the "
            "office on Tuesdays and Thursdays, which are the Core Days."
        ),
    },
]

RAG_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def main():
    for i, item in enumerate(QUESTIONS, 1):
        q = item["question"]
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("=" * 80)

        print("\n[A] WITHOUT context (closed-book):")
        print("   ", chat(q).replace("\n", "\n    "))

        print("\n[B] WITH context (open-book, i.e. manual RAG):")
        prompt = RAG_TEMPLATE.format(context=item["context"], question=q)
        print("   ", chat(prompt).replace("\n", "\n    "))
        print()

    # --- Experiment C: force a short answer and the model hallucinates ---
    print("=" * 80)
    print("[C] Hallucination demo: force the model to answer with just a number")
    print("=" * 80)
    forced = ("Reply with only a number, no words. How many days of paid annual "
              "leave do Nimbus Robotics employees get per year?")
    for attempt in range(1, 4):
        # temperature > 0 adds randomness, like a real chat app
        print(f"  attempt {attempt}: {chat(forced, temperature=0.8)!r}   (true answer: 23)")
    print("  -> wrong numbers stated confidently (sometimes a refusal). Never 23. This is hallucination.\n")

    print("Takeaway: same model, same question. The only difference is the")
    print("evidence we put in the prompt. RAG automates finding that evidence.")


if __name__ == "__main__":
    main()
