# Lesson 7: Advanced Retrieval

> **Goal:** fix the retrieval weaknesses we measured. The main tools are **hybrid search**, **reranking** and **query transformation**. All are implemented from scratch in [`common/retrievers.py`](../common/retrievers.py).

---

## 7.1 Where we are

From Lessons 4 to 6: pure vector search misses **exact identifiers**. "The robot says E-417" ranked the right chunk **9th of 24**, so RAG answered "I don't know". Generation was fine; **retrieval** was the bottleneck.

## 7.2 Keyword search: BM25

**BM25** is the classic search-engine ranking formula (used in Elasticsearch, Lucene, OpenSearch). It scores a chunk by the query words it contains:

```
score(q, d) = Σ over query terms t:   IDF(t) × tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
```

| Part | Meaning | Effect |
|---|---|---|
| **tf(t,d)** | how often term t appears in chunk d | more matches → higher, but **saturates** (k1 ≈ 1.2 to 2): the 10th "robot" adds little |
| **IDF(t)** | inverse document frequency: log of (how many chunks) / (how many contain t) | **rare words weigh a lot**. Measured: IDF("e-417") = **2.81**, IDF("robot") = **0.78** |
| **\|d\| / avgdl** | chunk length vs average | long chunks are penalised a bit (b ≈ 0.75) so they don't win just by containing more words |

Tokenisation matters: our tokenizer keeps `e-417`, `zephyr-3`, `wi-fi` as **single tokens** and drops stopwords ("the", "what", …).

| | Vector search | BM25 |
|---|---|---|
| Strength | meaning, paraphrases, synonyms | exact terms: codes, names, IDs, rare jargon |
| Weakness | exact identifiers, numbers | different wording ("holiday" ≠ "annual leave") |

## 7.3 Hybrid search with Reciprocal Rank Fusion (RRF)

Run **both** searches and merge the rankings. The scores can't be added directly: cosine is 0 to 1, BM25 is 0 to 20+. **RRF uses only ranks**:

```
RRF(chunk) = Σ over each result list:  1 / (c + rank_in_that_list)        (c = 60, a standard constant)
```

Example: E-417 is rank 9 in vector (1/69 = 0.0145) and rank 1 in BM25 (1/61 = 0.0164), so its total is **0.0309**, enough for the top 4.

A chunk ranked well by **both** lists wins. A chunk found by only one list can still get in.

> **Bug we hit while building this lesson:** BM25 first returned chunks with a score of **0** (no shared words) in random order. RRF gave those random chunks rank votes, and "Annual Leave" fell out of the hybrid top 4. **Fix:** BM25 returns only chunks with score > 0. *Only put real matches into the fusion.*

Weighted variants exist (e.g. 0.7 × vector + 0.3 × keyword on normalised scores). Most vector DBs (Qdrant, Weaviate, Elasticsearch, pgvector + tsvector) offer hybrid search built in.

## 7.4 Results: which retriever is best? (Part 3)

**Main eval set** (24 questions, mostly using the documents' own words):

| Retriever | hit@1 | hit@3 | hit@4 | MRR@4 |
|---|---|---|---|---|
| vector | 0.88 | 0.92 | 0.96 | 0.91 |
| BM25 | **0.96** | **1.00** | **1.00** | **0.98** |
| hybrid (RRF) | 0.88 | **1.00** | **1.00** | 0.93 |

**Paraphrase eval set** ([`data/eval_paraphrase.json`](../data/eval_paraphrase.json), 10 questions phrased *differently* from the docs, e.g. "Who started the business?", "How heavy a load can the machine carry?"):

| Retriever | hit@1 | hit@3 | hit@4 | MRR@4 |
|---|---|---|---|---|
| vector | **0.70** | **0.90** | **0.90** | **0.80** |
| BM25 | 0.20 | 0.40 | 0.40 | 0.30 |
| hybrid (RRF) | 0.40 | **0.90** | **0.90** | 0.62 |

**Interpretation:**
- On the first set alone, you'd conclude "BM25 is best, drop vectors". **That would be a mistake caused by a biased eval set** (we wrote questions using document words).
- On paraphrases BM25 collapses (0.40).
- **Hybrid is never the worst** and gets hit@4 = 1.00 / 0.90 on both. That robustness is why hybrid is the production default.
- Hybrid's hit@1 is lower, because fusion mixes rankings. That's what **reranking** addresses.

> **Lesson on evals:** make sure your eval set reflects how *real users* ask. Users rarely know your documents' exact wording.

## 7.5 Reranking (two-stage retrieval)

```
 Stage 1 (fast, recall):     hybrid search → top 20–50 candidates
 Stage 2 (slow, precision):  reranker scores each (question, passage) pair → keep best 3–5
```

**Why is a reranker more accurate than embeddings?**
- **Bi-encoder** (embeddings): question and passage are encoded **separately**; similarity is a dot product. Fast, since passages are pre-computed, but coarse.
- **Cross-encoder** (reranker): reads question **and** passage **together**, so it can reason about whether *this passage answers this question*. Accurate, but it must run for every pair at query time. That's why it only sees a short list.

We used llama3.1 as a reranker ("rate 0 to 10"). **Observed:** it helped on "What should I clean before a calibration?" (moved *Maintenance Schedule*, which says "clean the lidar window weekly", from #4 to #2). But it was **coarse**: most scores were 8, and it gave *Parental Leave* a 9 for "How much holiday can I take?".

> For real systems use a **dedicated reranker**: `bge-reranker-v2`, `mxbai-rerank`, Cohere Rerank, Jina Reranker, Voyage rerank. They're trained for exactly this and are fast. A general chat model is a poor substitute, which we saw here.

## 7.6 Query transformation

Sometimes the problem is the **question**, not the index.

| Technique | Idea | When |
|---|---|---|
| **HyDE** (Hypothetical Document Embeddings) | LLM writes a *fake answer*; you embed and search with that. It *looks like* a document, so it matches documents better. | Short or vague questions |
| **Query rewriting** | LLM rewrites a messy or conversational query into a clean search query | Chatbots ("and what about the other one?" → standalone question) |
| **Multi-query** | LLM generates 3 to 5 phrasings; search each; fuse with RRF | Ambiguous questions |
| **Decomposition** | Split "Compare X and Y" into two sub-questions, retrieve for each | Multi-part / multi-hop questions |
| **Step-back** | Ask a more general question first ("what are the leave policies?") | Questions needing background |

**HyDE results (Part 4):** for "How much holiday can I take?", the top score rose from **0.630 to 0.846** and the margin over #2 grew from 0.006 to 0.085. The fake answer's wording ("employees are entitled to X days of annual leave…") looks like the handbook. For "my robot keeps stopping because the motor gets hot" it changed little, because plain search already worked.
**Cost:** one extra LLM call per question, which adds latency.

## 7.7 Other advanced techniques (awareness)

| Technique | What it does |
|---|---|
| **Parent-document / small-to-big** | Search small chunks (precise), but send the larger parent section to the LLM (context) |
| **Sentence-window retrieval** | Retrieve a sentence, then expand to its ±N neighbouring sentences |
| **Contextual retrieval** | LLM-written context prepended to each chunk before embedding *and* BM25 (Lesson 3.4) |
| **MMR** (Maximal Marginal Relevance) | Pick results that are relevant *and* different from each other, to avoid 4 near-duplicate chunks |
| **Metadata self-query** | LLM extracts filters from the question ("policies from 2024" → `where year=2024`) |
| **Fine-tuned embeddings** | Train the embedding model on your domain's (question, chunk) pairs |

## 7.8 Hands-on

```bash
python3 07-advanced-retrieval/lesson.py      # ~30–60 s
```

**Part 6, the payoff:**
```
[vector] I don't know based on the available documents.
[hybrid] First clean the lidar window. If the error persists, run the "Auto-Calibrate"
         routine from the Nimbus Fleet Console [3].
```
Same LLM, same prompt, better retrieval, correct answer.

## 7.9 Key takeaways
- Vectors catch **meaning**; BM25 catches **exact terms**. Use **hybrid (RRF)** to get both.
- Only fuse real matches (filter zero-score keyword results).
- **Rerank** a short candidate list with a cross-encoder for precision at the top.
- **Transform queries** (HyDE, rewriting, multi-query) when questions are short, vague or conversational.
- Check improvements on an eval set that includes **paraphrases**, or you'll draw the wrong conclusions.

## 7.10 Try it yourself
1. In `make_hybrid`, change `candidates=20` to `5`. What happens to the E-417 query?
2. Change RRF's `c` from 60 to 1. Rankings get dominated by #1 positions. Watch Part 3 change.
3. Chat with hybrid retrieval: `python3 05-full-pipeline/ask.py --hybrid "The robot says E-417, what should I do?"`. Compare with the same question without `--hybrid`. (See how `ask.py` plugs in `make_hybrid`.)
4. Run the full generation eval (Lesson 6) with the hybrid retriever and compare correctness.

➡️ Next: [Lesson 8: Production RAG and Beyond](../08-production/README.md)
