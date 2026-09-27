"""
Lesson 2: Embeddings and similarity.

Part 1: cosine similarity by hand on tiny 2-D "toy" vectors
Part 2: real embeddings from nomic-embed-text (768 numbers per text)
Part 3: similarity matrix, where similar meanings get high scores
Part 4: semantic search vs keyword search (the reason RAG uses embeddings)

Run from the course root:   python3 02-embeddings/lesson.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import embed


def cosine_similarity(a, b) -> float:
    """cos(theta) = (a . b) / (|a| * |b|)   ranges from -1 (opposite) to 1 (same direction)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def part1_toy_vectors():
    print("=" * 80)
    print("PART 1: cosine similarity by hand (toy 2-D vectors)")
    print("=" * 80)
    # Imagine a 2-D meaning space: x = "about animals", y = "about vehicles"
    toy = {
        "cat": [0.9, 0.1],
        "dog": [0.8, 0.2],
        "car": [0.1, 0.9],
        "truck": [0.2, 0.95],
    }
    for w1, w2 in [("cat", "dog"), ("car", "truck"), ("cat", "car")]:
        print(f"  cos({w1:5}, {w2:5}) = {cosine_similarity(toy[w1], toy[w2]):.3f}")
    print("  -> Close directions score near 1. Unrelated directions score lower.\n")


def part2_real_embedding():
    print("=" * 80)
    print("PART 2: a real embedding")
    print("=" * 80)
    [vec] = embed(["Employees get 23 days of paid annual leave."])
    v = np.array(vec)
    print(f"  Dimensions : {len(v)}")
    print(f"  First 8    : {np.round(v[:8], 4).tolist()}")
    print(f"  Length |v| : {np.linalg.norm(v):.4f}")
    print(f"  -> A sentence became a list of {len(v)} numbers. Each number is not")
    print("     human-readable on its own; the MEANING lives in the direction.\n")


def part3_similarity_matrix():
    print("=" * 80)
    print("PART 3: similarity matrix")
    print("=" * 80)
    sentences = [
        "How many vacation days do I get?",              # 0  leave
        "Employees receive 23 days of paid annual leave.",  # 1  leave
        "What's my time-off allowance this year?",       # 2  leave
        "The robot battery lasts about 9 hours.",        # 3  battery
        "How long does the Zephyr-3 run on one charge?", # 4  battery
        "Rotterdam is a port city in the Netherlands.",  # 5  unrelated
    ]
    vecs = np.array(embed(sentences))
    # Normalise every vector to length 1. Then dot product == cosine similarity.
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T

    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")
    print("\n       " + "".join(f"  [{j}] " for j in range(len(sentences))))
    for i in range(len(sentences)):
        print(f"  [{i}] " + "".join(f" {sim[i, j]:.2f} " for j in range(len(sentences))))
    print("\n  Look at [0] vs [2]: 'vacation days' and 'time-off allowance' share")
    print("  almost NO words, yet score high. [3] vs [4] likewise (battery/charge).")
    print("  Unrelated pairs score lower. The absolute numbers depend on the model;")
    print("  what matters is the RANKING.\n")


def keyword_score(query: str, doc: str) -> int:
    """Naive keyword search: count how many query words appear in the doc."""
    q = {w.strip("?.,!").lower() for w in query.split()}
    d = {w.strip("?.,!").lower() for w in doc.split()}
    return len(q & d)


def part4_semantic_vs_keyword():
    print("=" * 80)
    print("PART 4: semantic search vs keyword search")
    print("=" * 80)
    docs = [
        "Every full-time employee receives 23 days of paid annual leave per calendar year.",
        "The Zephyr-3 docks itself at a charging station when the battery falls below 18 percent.",
        "Passwords must be at least 14 characters long and are rotated every 180 days.",
        "All employees must be in the office on Tuesdays and Thursdays.",
    ]

    # nomic-embed-text is trained with task prefixes: one for the stored
    # documents and one for the questions. Other models may not need them.
    doc_vecs = np.array(embed(["search_document: " + d for d in docs]))

    for query in ["How much holiday can I take?", "When should I recharge the robot?"]:
        [q_vec] = embed(["search_query: " + query])
        rows = [(keyword_score(query, d), cosine_similarity(q_vec, dv), d)
                for d, dv in zip(docs, doc_vecs)]
        rows.sort(key=lambda r: -r[1])
        print(f"  Query: {query!r}")
        print(f"  {'keyword':>7}  {'cosine':>6}  document")
        for kw, cs, d in rows:
            print(f"  {kw:>7}  {cs:>6.3f}  {d[:70]}")
        print(f"  margin between #1 and #2: {rows[0][1] - rows[1][1]:.3f}\n")

    print("  Keyword search finds 0 useful overlapping words ('holiday' is not")
    print("  'annual leave', 'recharge' is not 'charging'). Embeddings still rank")
    print("  the right document #1. That's why RAG retrieves with embeddings.")
    print()
    print("  BUT notice the margins are small. Small embedding models squeeze all")
    print("  scores into a narrow band, so #2 can be almost as 'similar' as #1.")
    print("  That's why real systems retrieve the top-k (not just 1) and add")
    print("  hybrid search / reranking (Lesson 7).\n")


if __name__ == "__main__":
    part1_toy_vectors()
    part2_real_embedding()
    part3_similarity_matrix()
    part4_semantic_vs_keyword()
