# Lesson 4: Vector Store and Retrieval

> **Goal:** store all chunk embeddings and, for any question, quickly find the top-k most similar chunks. This is the **R** in RAG.

---

## 4.1 What a vector store is

A **vector store** (or vector database) holds three things for each chunk:

```
 id                       vector (768 floats)            text + metadata
 ─────────────────────    ──────────────────────────     ─────────────────────────────
 employee_handbook.md#0   [0.012, -0.173, 0.044, …]      "…23 days of paid annual leave…"  source=…, section=…
 employee_handbook.md#1   [0.031, -0.090, 0.012, …]      "…12 days of paid sick leave…"    source=…, section=…
 …                        …                              …
```

and supports one main operation:

> **search(query, k)** → the k chunks whose vectors are most similar to the query's vector.

## 4.2 Retrieval step by step

```
"How many vacation days do I get?"
        │ 1. embed (with "search_query: " prefix)
        ▼
   q = [0.02, -0.11, …]                         (768 numbers, normalised)
        │ 2. score = V · q   (dot product with every stored vector)
        ▼
   scores = [0.656, 0.630, 0.619, 0.41, …]      (one per chunk)
        │ 3. take top-k (k=3)
        ▼
   #1 Annual Leave (0.656)   #2 Sick Leave (0.630)   #3 Remote Work (0.619)
```

Because all vectors are normalised, step 2 is **one matrix multiplication**, `V @ q`. That's the whole trick. Our [`common/vector_store.py`](../common/vector_store.py) is ~70 lines.

## 4.3 Choosing k

- **k too small** (1): if the best chunk is ranked #2 (common, see Lesson 2's tiny margins), you miss it.
- **k too large** (20): the prompt fills with noise, cost goes up, and the LLM can get distracted.
- **Typical:** k = 3 to 10. Or retrieve 20 to 50 and then **rerank** down to 3 to 5 (Lesson 7).

## 4.4 Exact vs approximate search (ANN)

Our store compares the query with **every** vector: exact **k-nearest-neighbour (kNN)** search, cost O(N × D).

*Measured in Part 4:* 1k vectors ≈ 0.04 ms, 100k ≈ 3 ms, 500k ≈ 21 ms per query (and 1.5 GB RAM). Linear growth. Brute force is fine up to around 1M vectors; beyond that you need **Approximate Nearest Neighbour (ANN)** indexes, which give up a little accuracy for large speed-ups:

| Index | Idea (intuition) |
|---|---|
| **HNSW** (Hierarchical Navigable Small World) | A multi-layer graph of "neighbour" links. Start at the top (few long-range links), greedily move closer to the query, drop to denser layers. Like using motorways, then main roads, then streets. **The most common default.** |
| **IVF** (Inverted File) | Cluster vectors into, say, 1000 groups (k-means). At query time, search only the nearest few clusters. |
| **PQ** (Product Quantisation) | Compress each vector into a few bytes, so much less memory is needed and distances are approximated. Often combined: IVF-PQ. |

The measure of ANN quality is **recall@k**: what fraction of the true top-k the approximate search returns (e.g. 0.98).

## 4.5 Vector database landscape

| Tool | Type | Good for |
|---|---|---|
| **numpy (this course)** | library | learning, < 100k vectors |
| **FAISS** (Meta) | library | fast in-process ANN, no server |
| **Chroma** | embedded DB | quick prototypes, simple Python API |
| **LanceDB** | embedded DB | on-disk, larger-than-memory |
| **pgvector** | Postgres extension | you already use Postgres; SQL + vectors + joins |
| **Qdrant, Weaviate, Milvus** | servers | production, filtering, hybrid search, scale |
| **Pinecone** | managed cloud | zero ops |
| **Elasticsearch / OpenSearch** | search engines | already used for keyword search, now with vectors |

They all do the same core job as our 70 lines, plus ANN indexes, persistence, filtering, concurrency, replication and backups.

## 4.6 Metadata filtering

`store.search(q, k, where={"source": "it_faq.md"})` ranks **only** the chunks that match the filter. Uses:
- **Security / multi-tenancy**: only documents this user is allowed to see. *Always filter in retrieval, never trust the LLM to "not mention" secret data.*
- Scope: "only product manuals", "only 2025 policies", "only language = de".

## 4.7 Hands-on

```bash
python3 04-vector-store/lesson.py
```

This builds the index at `data/index/` (**later lessons load it from there**) and runs retrieval.

### What we observed (see `sample_output.txt`)

| Query | Top result | Verdict |
|---|---|---|
| How many vacation days do I get? | Annual Leave (0.656) | ✅ |
| How long does it take to charge the Zephyr-3? | Battery and Charging (0.767) | ✅ |
| Who founded the company? | History (0.719) | ✅ |
| What happens if my Wi-Fi is 2.4 GHz? | E-777 Fleet Connection Lost (0.628) | ✅ (even across different wording) |
| **The robot says E-417, what should I do?** | E-101 (wrong!). **E-417 ranked 9th of 24** | ❌ |

**Part 2b, an important failure:** embeddings capture *meaning* but are weak on **exact identifiers** (error codes, part numbers, names, SKUs). The query "E-417" alone ranked the right chunk only 5th; "lidar calibration error" ranked it 1st. All error-code chunks *look alike* to the embedding model ("robot + error + fix"). With k=3 the LLM would never see the answer. This is why production systems use **hybrid search**, which we build in Lesson 7.

## 4.8 Key takeaways
- A vector store = matrix of normalised vectors + text/metadata + a top-k dot-product search.
- Retrieval quality **is** RAG quality: if the right chunk isn't retrieved, the LLM can't answer.
- Pure vector search misses exact-match queries; plan for hybrid search.
- Use metadata filters for permissions and scope.
- Brute force is fine for small data; ANN (HNSW) for millions.

## 4.9 Try it yourself
1. Add your own questions to `part2_retrieve`. Can you find more failures?
2. Try `k=5`: does E-417 get in? (No, it's rank 9.) What k would you need?
3. Put a new `.md` file in `data/knowledge_base/`, rerun, and query it. That's how RAG "learns" new facts with no training.

➡️ Next: [Lesson 5: The Full RAG Pipeline](../05-full-pipeline/README.md), where we connect retrieval to the LLM.
