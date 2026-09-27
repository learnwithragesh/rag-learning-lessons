"""
A minimal vector store in ~70 lines of numpy (Lesson 4).

It does exactly what Chroma / FAISS / pgvector do at their core:
  * keep a matrix of normalised embeddings  (N chunks x D dimensions)
  * keep the chunk text + metadata next to it
  * search = one matrix-vector product, then take the top-k scores

Search here is brute force (exact). Real vector DBs add ANN indexes
(e.g. HNSW) to make this fast for millions of vectors.
"""

import json
from pathlib import Path

import numpy as np

from common.llm import embed

# nomic-embed-text task prefixes (see Lesson 2.5). Use "" for models without them.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


class VectorStore:
    def __init__(self):
        self.vectors = np.zeros((0, 0), dtype=np.float32)  # (N, D)
        self.chunks: list[dict] = []                        # N dicts

    # ---------------- indexing ----------------
    def add(self, chunks: list[dict]) -> None:
        """Embed chunk texts and append them to the store."""
        vecs = np.array(embed([DOC_PREFIX + c["text"] for c in chunks]), dtype=np.float32)
        vecs = _normalise(vecs)
        self.vectors = vecs if len(self.chunks) == 0 else np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)

    # ---------------- searching ----------------
    def search(self, query: str, k: int = 3, where: dict | None = None) -> list[dict]:
        """Return the k most similar chunks: [{**chunk, "score": float}, ...].
        `where` is an optional metadata filter, e.g. {"source": "it_faq.md"}."""
        [q] = embed([QUERY_PREFIX + query])
        q = _normalise(np.array(q, dtype=np.float32))
        scores = self.vectors @ q  # cosine similarity with EVERY chunk at once

        candidates = range(len(self.chunks))
        if where:
            candidates = [i for i in candidates
                          if all(self.chunks[i].get(key) == val for key, val in where.items())]
        best = sorted(candidates, key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]

    # ---------------- persistence ----------------
    def save(self, folder: Path = DEFAULT_INDEX_DIR) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(json.dumps(self.chunks, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, folder: Path = DEFAULT_INDEX_DIR) -> "VectorStore":
        store = cls()
        store.vectors = np.load(folder / "vectors.npy")
        store.chunks = json.loads((folder / "chunks.json").read_text())
        return store

    def __len__(self) -> int:
        return len(self.chunks)
