"""
Advanced retrieval (Lesson 7). Every retriever has the same signature as
VectorStore.search:  retriever(question, k) -> list of chunk dicts with "score".

  BM25Index          keyword search, implemented from scratch
  hybrid_search      vector + BM25 merged with Reciprocal Rank Fusion (RRF)
  hyde_search        HyDE: embed a hypothetical answer instead of the question
  llm_rerank         re-order candidates with the LLM as a relevance judge
"""

import math
import re
from collections import Counter

from common.llm import chat

# ============================================================== BM25 (keywords)
STOPWORDS = set("""a an and are as at be by can do does for from how i if in is it
its my of on or should the this to what when where which who why with you your""".split())


def tokenize(text: str) -> list[str]:
    # keeps codes like "e-417", "zephyr-3", "wi-fi" as single tokens
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25, the classic search-engine ranking function.

    score(q, d) = sum over query terms t of
                  IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

      tf    how often t occurs in d        (more = better, but saturates via k1)
      IDF   how RARE t is across all docs  (rare terms like 'e-417' weigh a lot)
      |d|   document length                (b penalises long documents)
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c["text"]) for c in chunks]
        self.tfs = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        n = len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, dl, s = self.tfs[i], len(self.docs[i]), 0.0
        for t in query_tokens:
            if t in tf:
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[t] * num / den
        return s

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = tokenize(query)
        scores = [(self.score(q, i), i) for i in range(len(self.chunks))]
        # Drop chunks with score 0: they share no word with the query. Returning
        # them anyway would hand random chunks "votes" in hybrid fusion (Lesson 7).
        scores = sorted((s for s in scores if s[0] > 0), reverse=True)
        return [{**self.chunks[i], "score": s} for s, i in scores[:k]]


# ============================================================== Hybrid (RRF)
def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """RRF: each list votes 1/(c + rank) for each chunk. Uses only RANKS, so we
    never have to compare a cosine score (0..1) with a BM25 score (0..20+)."""
    fused, by_id = {}, {}
    for results in result_lists:
        for rank, r in enumerate(results, 1):
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (c + rank)
            by_id[r["id"]] = r
    best = sorted(fused, key=lambda i: -fused[i])[:k]
    return [{**by_id[i], "score": fused[i]} for i in best]


def make_hybrid(vector_store, bm25: BM25Index, candidates: int = 20):
    """Return a retriever that fuses vector and BM25 results."""
    def hybrid_search(question: str, k: int = 4) -> list[dict]:
        return reciprocal_rank_fusion(
            [vector_store.search(question, candidates), bm25.search(question, candidates)], k
        )
    return hybrid_search


# ============================================================== HyDE
HYDE_PROMPT = """Write a short passage (2-3 sentences) from a company's internal
documentation that would answer this question. Invent plausible details.

Question: {question}
Passage:"""


def make_hyde(vector_store):
    """HyDE (Hypothetical Document Embeddings): questions and answers look
    different, so first let the LLM write a FAKE answer, then search with it.
    The fake answer's facts may be wrong, but its wording resembles real docs."""
    def hyde_search(question: str, k: int = 4) -> list[dict]:
        fake = chat(HYDE_PROMPT.format(question=question), temperature=0.0)
        return vector_store.search(fake, k)
    return hyde_search


# ============================================================== LLM reranker
RERANK_PROMPT = """Rate how useful the PASSAGE is for answering the QUESTION.
Reply with only a number from 0 (irrelevant) to 10 (directly answers it).

QUESTION: {question}

PASSAGE:
{passage}

Score:"""


def llm_rerank(question: str, candidates: list[dict], k: int) -> list[dict]:
    """Cross-encoder style reranking: the model reads question AND passage
    together (unlike embeddings, which encode them separately). Much more
    accurate, but one LLM call per candidate, so only rerank a short list."""
    scored = []
    for c in candidates:
        out = chat(RERANK_PROMPT.format(question=question, passage=c["text"]))
        m = re.search(r"\d+(\.\d+)?", out)
        scored.append({**c, "rerank_score": float(m.group()) if m else 0.0})
    scored.sort(key=lambda c: -c["rerank_score"])  # stable: ties keep hybrid order
    return scored[:k]


def make_reranked(first_stage, candidates: int = 8):
    """Two-stage retrieval: cheap retriever finds `candidates`, LLM picks best k."""
    def reranked_search(question: str, k: int = 4) -> list[dict]:
        return llm_rerank(question, first_stage(question, candidates), k)
    return reranked_search
