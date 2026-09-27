"""
Lesson 7: Advanced retrieval.

Part 1: BM25 keyword search vs vector search on the E-417 problem
Part 2: Hybrid search (RRF) fixes it
Part 3: Compare retrievers on the main eval set AND a paraphrase-only eval set
Part 4: HyDE (hypothetical document embeddings)
Part 5: LLM reranking (two-stage retrieval)
Part 6: End-to-end: the E-417 question answered with hybrid retrieval

Run from the course root:   python3 07-advanced-retrieval/lesson.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.evaluate import EVAL_PATH, evaluate_retrieval, load_eval_set
from common.rag import answer
from common.retrievers import BM25Index, make_hybrid, make_hyde, make_reranked, tokenize
from common.vector_store import VectorStore

E417 = "The robot says E-417, what should I do?"


def rank_of(results, section_prefix):
    return next((i for i, r in enumerate(results, 1) if r["section"].startswith(section_prefix)), None)


def show(results, n=4):
    for i, r in enumerate(results[:n], 1):
        extra = f" rerank={r['rerank_score']:.0f}" if "rerank_score" in r else ""
        print(f"    #{i} score={r['score']:.3f}{extra}  {r['section']}")


def main():
    store = VectorStore.load()
    bm25 = BM25Index(store.chunks)
    hybrid = make_hybrid(store, bm25)

    print("=" * 80)
    print("PART 1: keyword search (BM25) vs vector search")
    print("=" * 80)
    print(f"  Query tokens for BM25: {tokenize(E417)}")
    print(f"  IDF of 'e-417' = {bm25.idf['e-417']:.2f} (rare, so weighs a lot)"
          f"   IDF of 'robot' = {bm25.idf['robot']:.2f} (common)\n")
    print(f"  VECTOR search for {E417!r}:")
    show(store.search(E417, 4))
    print(f"\n  BM25 search:")
    show(bm25.search(E417, 4))
    q2 = "How much holiday can I take?"
    print(f"\n  ...but BM25 fails on paraphrases. BM25 for {q2!r}:")
    show(bm25.search(q2, 3))
    print("  (no chunk contains 'holiday'; only 'take' matched, in Sick Leave.)")
    print("  VECTOR for the same:")
    show(store.search(q2, 3))

    print("\n" + "=" * 80)
    print("PART 2: hybrid search = vector + BM25 fused with RRF")
    print("=" * 80)
    for q in [E417, q2]:
        print(f"  {q!r}")
        show(hybrid(q, 4))
        print()
    print("  -> hybrid pulls E-417 into the top-4 (BM25's strength) while keeping")
    print("     Annual Leave for the paraphrase (the vector's strength).\n")

    print("=" * 80)
    print("PART 3: all retrievers on two eval sets")
    print("=" * 80)
    retrievers = {"vector": store.search, "bm25": bm25.search, "hybrid (RRF)": hybrid}
    eval_sets = {
        "MAIN eval set (24 answerable, mostly same words as the docs)": load_eval_set(),
        "PARAPHRASE eval set (10 questions, different words than the docs)":
            load_eval_set(EVAL_PATH.parent / "eval_paraphrase.json"),
    }
    for title, eval_set in eval_sets.items():
        print(f"\n  {title}")
        print(f"  {'retriever':<14} {'hit@1':>6} {'hit@3':>6} {'hit@4':>6} {'MRR@4':>6}")
        for name, fn in retrievers.items():
            m1, m3, m4 = (evaluate_retrieval(fn, eval_set, k) for k in (1, 3, 4))
            print(f"  {name:<14} {m1['hit@k']:>6.2f} {m3['hit@k']:>6.2f} {m4['hit@k']:>6.2f} {m4['mrr']:>6.2f}")
            if m4["misses"]:
                print(f"  {'':<14} misses@4: {m4['misses']}")
    print("\n  -> BM25 wins when questions reuse the documents' words and collapses on")
    print("     paraphrases. Vector is the reverse. Hybrid is never the worst: robust.")

    print("\n" + "=" * 80)
    print("PART 4: HyDE (search with a hypothetical answer)")
    print("=" * 80)
    hyde = make_hyde(store)
    for q in [q2, "My robot keeps stopping because the motor gets hot"]:
        print(f"  {q!r}")
        print("   plain vector:"); show(store.search(q, 3), 3)
        print("   HyDE        :"); show(hyde(q, 3), 3)
        print()

    print("=" * 80)
    print("PART 5: LLM reranking (hybrid top-8 -> LLM scores each -> best 4)")
    print("=" * 80)
    reranked = make_reranked(hybrid, candidates=8)
    for q in ["What should I clean before running a calibration?", q2]:
        print(f"  {q!r}")
        print("   hybrid only:"); show(hybrid(q, 4))
        print("   reranked   :"); show(reranked(q, 4))
        print()

    print("=" * 80)
    print("PART 6: the E-417 question end to end, before vs after")
    print("=" * 80)
    for name, fn in [("vector", store.search), ("hybrid", hybrid)]:
        r = answer(E417, fn, k=4)
        print(f"  [{name}] {r['answer']}")
    print()


if __name__ == "__main__":
    main()
