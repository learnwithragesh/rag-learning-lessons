"""
Interactive RAG chat over the knowledge base.

    python3 05-full-pipeline/ask.py                      # interactive loop
    python3 05-full-pipeline/ask.py "Who founded Nimbus?"  # one question
    python3 05-full-pipeline/ask.py --show-prompt "..."    # also print the prompt
    python3 05-full-pipeline/ask.py --hybrid "..."         # hybrid retrieval (Lesson 7)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.rag import answer
from common.retrievers import BM25Index, make_hybrid
from common.vector_store import DEFAULT_INDEX_DIR, VectorStore


def run(question: str, retriever, show_prompt: bool) -> None:
    r = answer(question, retriever, k=4)
    if show_prompt:
        print("\n----- PROMPT -----\n" + r["prompt"] + "\n------------------")
    print(f"\n{r['answer']}\n")
    print("Sources:")
    for i, c in enumerate(r["chunks"], 1):
        print(f"  [{i}] {c['source']} > {c['section']}  (score {c['score']:.3f})")
    print()


def main():
    if not (DEFAULT_INDEX_DIR / "vectors.npy").exists():
        raise SystemExit("No index found. Run:  python3 04-vector-store/lesson.py")
    store = VectorStore.load()

    args = sys.argv[1:]
    show_prompt = "--show-prompt" in args
    retriever = make_hybrid(store, BM25Index(store.chunks)) if "--hybrid" in args else store.search
    args = [a for a in args if a not in ("--show-prompt", "--hybrid")]

    if args:
        run(" ".join(args), retriever, show_prompt)
        return
    print("Ask about Nimbus Robotics (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q, retriever, show_prompt)


if __name__ == "__main__":
    main()
