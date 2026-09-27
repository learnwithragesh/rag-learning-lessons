# Lesson 8: Production RAG and Beyond

> **Goal:** know what changes when a RAG prototype becomes a real system, and where RAG is heading. This lesson is theory and checklists only (no code). It's a map for your next projects.

---

## 8.1 From our course code to production

| Concern | Our course | Production |
|---|---|---|
| Documents | 5 clean Markdown files | 10k to 10M files: PDFs, Word, HTML, Confluence, Slack, tickets, databases |
| Parsing | `read_text()` | OCR, table extraction, layout-aware parsers (docling, unstructured, LlamaParse) |
| Vector store | numpy matrix in RAM | Qdrant / pgvector / Weaviate / Elasticsearch with HNSW |
| Retrieval | vector / BM25 / hybrid in Python | built-in hybrid search + dedicated reranker model |
| LLM | local llama3.1 8B | stronger hosted or self-hosted model, with fallbacks |
| Evaluation | 27 + 10 hand-written questions | hundreds of real-user questions, automated in CI, monitored live |
| Updates | rebuild the whole index | incremental: detect changed files and re-embed only those chunks |

## 8.2 The ingestion pipeline (keeping the index fresh)

```
 sources ─► connectors ─► parse ─► clean ─► chunk ─► enrich ─► embed ─► upsert into vector DB
 (S3, SharePoint,          (PDF→text,  (dedupe,  (L3)   (metadata,   (L2)    (with id, hash,
  Confluence, DB)           tables)     boilerplate)      summaries)           permissions, date)
```

- Give every chunk a **stable ID** and a **content hash**. When a file changes, delete its old chunks and insert new ones. Skip unchanged chunks.
- Handle **deletions** (a removed policy must stop being retrieved).
- **Version your embedding model**: changing models means re-embedding *everything* (Lesson 2.7).
- Schedule it (cron, queue, webhooks) and **monitor** it (failed parses, empty chunks).

## 8.3 Security and access control ⚠️

| Risk | Mitigation |
|---|---|
| User sees documents they shouldn't | Store permissions (ACLs) as chunk metadata and **filter at retrieval time** (Lesson 4.6). Never rely on the LLM to hide content. |
| **Prompt injection** via documents ("ignore previous instructions…") | Treat retrieved text as data; delimit it clearly; limit the tools and actions the LLM can take; review sources you ingest (Lesson 5.7, exercise 2) |
| Sensitive data (PII) in chunks or logs | Redact at ingestion; control log retention |
| Data leaving your environment | Local models (like this course) or providers with suitable data agreements |

## 8.4 Latency and cost

A typical query budget:

```
 embed query      ~10–50 ms
 vector search    ~5–50 ms
 BM25 + fusion    ~5–20 ms
 rerank 20 docs   ~50–300 ms   (dedicated reranker)
 LLM generation   ~0.5–5 s     ← dominant
```

Levers:
- **Stream** the LLM answer token by token, so perceived latency drops a lot.
- **Cache**: embeddings of repeated queries; answers to frequent questions (semantic cache); provider **prompt caching** for long fixed system prompts.
- Keep k small **after** reranking; fewer tokens means cheaper and faster.
- Smaller or faster model for easy questions, larger for hard ones (**routing**).

## 8.5 Observability and continuous evaluation

Log **every** request: question, retrieved chunk IDs and scores, the prompt, the answer, latency, tokens, user feedback (👍/👎).
- Turn 👎 answers into new **eval set** entries.
- Run the eval suite (Lesson 6) on every change to chunking, models or prompts, as a regression test in CI.
- Track retrieval metrics and answer quality over time. Watch for drift when documents change.
- Tools: Langfuse, Arize Phoenix, LangSmith, TruLens, RAGAS, DeepEval.

## 8.6 Frameworks: should you use one?

| Framework | Notes |
|---|---|
| **LangChain / LangGraph** | Huge ecosystem of loaders, splitters and integrations; LangGraph for agent workflows |
| **LlamaIndex** | Focused on RAG: indexing, retrievers, query engines |
| **Haystack** | Pipeline-oriented, production-minded |
| **DSPy** | Optimises prompts and pipelines programmatically |

**Advice:** because you built everything from scratch, you now know what these frameworks do inside. Use them for **connectors and integrations**. Keep the core logic (chunking, retrieval, prompt) **visible and testable**. Every framework call maps to something in `common/`:

| Framework concept | Your code |
|---|---|
| Document loader | `load_documents()` |
| Text splitter | `chunk_markdown()` |
| Embeddings | `embed()` |
| VectorStore / Retriever | `VectorStore.search()`, `make_hybrid()` |
| Prompt template | `SYSTEM_PROMPT`, `PROMPT_TEMPLATE` |
| Chain / Query engine | `answer()` |
| Evaluator | `evaluate_retrieval()`, `judge_faithfulness()` |

## 8.7 Beyond basic RAG

| Pattern | Idea | Use when |
|---|---|---|
| **Conversational RAG** | Rewrite each follow-up ("what about part-timers?") into a standalone query using chat history, then retrieve | Chatbots |
| **Agentic RAG** | The LLM decides *whether*, *what* and *how many times* to search, as tool calls in a loop: search → read → search again → answer | Multi-step questions, several sources |
| **Corrective RAG (CRAG) / Self-RAG** | The model grades retrieved chunks; if they're poor, it rewrites the query, searches again, or falls back to web search | High-reliability needs |
| **GraphRAG** | Extract entities and relationships into a knowledge graph; retrieve via graph traversal and community summaries | "Global" questions ("main themes across all reports?"), multi-hop relations |
| **Multimodal RAG** | Embed images, tables, diagrams and slides (e.g. with vision-language models) | Manuals with diagrams, scanned docs |
| **Structured RAG / Text-to-SQL** | The LLM writes a query against a database instead of searching text | Numbers, aggregations ("total sales in Q3") |
| **Long-context models vs RAG** | Models with 200k to 1M+ token windows can read whole document sets | Small corpora. RAG is still cheaper, faster, citable and permission-aware at scale; the two are often combined |

## 8.8 Production checklist

- [ ] Parsing verified on real documents (tables, PDFs) and not only clean text
- [ ] Chunking tuned with measurements; contextual headers or contextual retrieval
- [ ] Hybrid retrieval + reranker
- [ ] Permission filtering at retrieval time
- [ ] Prompt with grounding, refusal and citations; injection-aware
- [ ] Eval set from real questions (with paraphrases and unanswerable ones); judge calibrated
- [ ] Evals run automatically on every change
- [ ] Incremental ingestion with deletes; embedding model versioned
- [ ] Logging, tracing, user feedback loop
- [ ] Streaming, caching, latency budget, cost monitoring

## 8.9 Where to go next

1. **Swap in your own documents.** Put them in `data/knowledge_base/` (as .md or .txt), rebuild (Lesson 4), write 20 eval questions, and measure.
2. **Try a real vector DB:** replace `VectorStore` with Chroma or Qdrant, keeping the same `search(question, k)` signature.
3. **Add a real reranker**, e.g. `sentence-transformers` CrossEncoder `BAAI/bge-reranker-base`, and compare it with the llama3.1 reranker using Lesson 6's metrics.
4. **Build conversational RAG:** add chat history and query rewriting to `ask.py`.
5. **Build agentic RAG:** give an LLM a `search(query)` tool and let it decide when to call it.

## 8.10 Recommended reading
- Lewis et al., 2020: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (the original RAG paper)
- Robertson and Zaragoza, 2009: *The Probabilistic Relevance Framework: BM25 and Beyond*
- Cormack et al., 2009: *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*
- Gao et al., 2022: *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Liu et al., 2023: *Lost in the Middle: How Language Models Use Long Contexts*
- Anthropic, 2024: *Introducing Contextual Retrieval* (blog post)
- Edge et al., 2024: *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (GraphRAG)
- MTEB leaderboard (Hugging Face), for choosing embedding models

🎓 **You've completed the course.** Go back to [the main README](../README.md) for the one-page summary.
