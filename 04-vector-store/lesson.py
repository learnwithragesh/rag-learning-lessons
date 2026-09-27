"""
Lesson 4: Vector store and retrieval.

Part 1: INDEXING: chunk -> embed -> store -> save to disk (data/index/)
Part 2: RETRIEVAL: top-k semantic search for several questions
Part 3: METADATA FILTERING: restrict search to one document
Part 4: SCALE: how brute-force search time grows with collection size

Run from the course root:   python3 04-vector-store/lesson.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.chunking import chunk_all
from common.vector_store import DEFAULT_INDEX_DIR, VectorStore


def part1_index() -> VectorStore:
    print("=" * 80)
    print("PART 1: indexing (runs once, offline)")
    print("=" * 80)
    chunks = chunk_all()
    t = time.perf_counter()
    store = VectorStore()
    store.add(chunks)
    print(f"  embedded {len(store)} chunks in {time.perf_counter() - t:.2f}s")
    print(f"  vector matrix shape: {store.vectors.shape}  (chunks x dimensions)")
    store.save()
    print(f"  saved to {DEFAULT_INDEX_DIR}/")
    print("  (vectors.npy = the numbers, chunks.json = text + metadata)\n")
    return store


def print_results(results):
    for rank, r in enumerate(results, 1):
        first_line, _, body = r["text"].partition("\n")
        print(f"    #{rank}  score={r['score']:.3f}  [{r['source']} > {r['section']}]")
        print(f"         {body[:95]}...")


def part2_retrieve(store: VectorStore):
    print("=" * 80)
    print("PART 2: retrieval (runs for every question)")
    print("=" * 80)
    questions = [
        "How many vacation days do I get?",
        "The robot says E-417, what should I do?",
        "How long does it take to charge the Zephyr-3?",
        "Who founded the company?",
        "What happens if my Wi-Fi is 2.4 GHz?",
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        print_results(store.search(q, k=3))
    print()


def part2b_failure(store: VectorStore):
    print("=" * 80)
    print("PART 2b: a real retrieval FAILURE (exact codes)")
    print("=" * 80)
    print("  Where does the correct 'E-417' chunk rank among all 24 chunks?\n")
    for q in ["The robot says E-417, what should I do?", "E-417", "lidar calibration error"]:
        ranked = store.search(q, k=len(store))
        rank = next(i for i, r in enumerate(ranked, 1) if r["section"].startswith("E-417"))
        mark = "OK " if rank <= 3 else "MISS"
        print(f"  [{mark}] rank {rank:>2}  for query {q!r}")
    print("\n  -> Embeddings capture MEANING ('lidar calibration') but are weak at")
    print("     EXACT tokens like error codes, product IDs, names, numbers.")
    print("     With k=3 the LLM would never see the right chunk.")
    print("     Fix: hybrid search (keyword + vector), Lesson 7.\n")


def part3_filter(store: VectorStore):
    print("=" * 80)
    print("PART 3: metadata filtering")
    print("=" * 80)
    q = "How long does something take?"
    print(f"  Q: {q!r}   (vague on purpose)\n")
    print("  No filter:")
    print_results(store.search(q, k=2))
    print("\n  where={'source': 'it_faq.md'}:")
    print_results(store.search(q, k=2, where={"source": "it_faq.md"}))
    print("\n  -> filters narrow the search space BEFORE ranking. Used for")
    print("     permissions (only docs this user may see), product lines, dates...\n")


def part4_scale():
    print("=" * 80)
    print("PART 4: brute-force search cost vs collection size (random 768-d vectors)")
    print("=" * 80)
    rng = np.random.default_rng(0)
    q = rng.standard_normal(768).astype(np.float32)
    for n in [1_000, 10_000, 100_000, 500_000]:
        m = rng.standard_normal((n, 768)).astype(np.float32)
        t = time.perf_counter()
        for _ in range(10):
            scores = m @ q
            np.argpartition(-scores, 5)[:5]  # top-5 without a full sort
        ms = (time.perf_counter() - t) / 10 * 1000
        print(f"  {n:>8,} vectors  ->  {ms:8.2f} ms per query   ({m.nbytes / 1e6:,.0f} MB RAM)")
    print("  -> time grows linearly with N. Fine up to ~1M on one machine;")
    print("     beyond that, vector DBs use ANN indexes (HNSW, IVF) for ~log(N) search.\n")


if __name__ == "__main__":
    store = part1_index()
    part2_retrieve(store)
    part2b_failure(store)
    part3_filter(store)
    part4_scale()
