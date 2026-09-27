# Lesson 2: Embeddings and Similarity

> **Goal:** understand how a computer can measure *"how similar in meaning"* two texts are. This is the engine of retrieval.

---

## 2.1 The problem

In Lesson 1 we hand-picked the right paragraph. To automate that, the computer must answer:

> *"Which of my 10,000 paragraphs is most related to this question?"*

**Keyword matching** (does the paragraph contain the question's words?) fails when people use different words for the same idea:

| Question says | Document says |
|---|---|
| "holiday", "vacation", "time off" | "annual leave" |
| "recharge the robot" | "docks at a charging station" |
| "my laptop is slow" | "hardware refresh policy" |

We need to compare **meaning**, not spelling.

## 2.2 What an embedding is

An **embedding** is a list of numbers (a **vector**) that represents the meaning of a piece of text.

```
"Employees get 23 days of paid annual leave."
          │
          ▼  embedding model (nomic-embed-text)
[0.0005, 0.0154, -0.1758, -0.0239, ... 768 numbers in total]
```

- An **embedding model** is a neural network trained so that **texts with similar meaning end up with vectors pointing in similar directions**.
- It's a *different kind of model* from a chat LLM: it outputs numbers, not text.
- The number of values is the **dimension** (nomic-embed-text: 768; OpenAI text-embedding-3-small: 1536; many others use 384 to 4096).
- No single number means anything readable. Meaning is spread across all of them.

### How are they trained? (intuition)
The model sees millions of pairs that *should* be close (a question and its answer, a title and its article, paraphrases) and pairs that *should* be far apart. Training pulls the good pairs together and pushes the bad pairs apart in vector space. This is called **contrastive learning**.

## 2.3 Meaning as geometry

Picture a 2-D map (real embeddings have hundreds of dimensions, but the idea is the same):

```
  vehicles ▲
           │    • truck
           │  • car
           │
           │                 • dog
           │               • cat
           └──────────────────────► animals
```

"cat" and "dog" point in nearly the same direction. "cat" and "car" don't. **Similar meaning means similar direction.**

## 2.4 Measuring similarity: cosine similarity

The standard measure is **cosine similarity**: the cosine of the angle θ between two vectors.

```
                 a · b            Σ aᵢ·bᵢ
cos(θ) =  ─────────────── = ──────────────────
             |a| × |b|        √Σaᵢ²  ×  √Σbᵢ²
```

| Value | Meaning |
|---|---|
| **1.0** | same direction (same meaning) |
| **0** | unrelated (perpendicular) |
| **-1** | opposite |

**Worked example** (toy vectors from the lesson): cat = [0.9, 0.1], car = [0.1, 0.9]
- a · b = 0.9×0.1 + 0.1×0.9 = **0.18**
- |cat| = √(0.81+0.01) = 0.906, and |car| is the same
- cos = 0.18 / (0.906 × 0.906) = **0.22** (not similar)

**Tip: normalisation.** If you scale every vector to length 1 up front ("normalise"), then cosine similarity equals the plain **dot product**, which is very fast. Vector databases do this constantly. `nomic-embed-text` already returns length-1 vectors (the lesson prints `|v| = 1.0000`).

Other distance measures you'll see: **dot product** (equal to cosine for normalised vectors) and **Euclidean / L2 distance** (smaller means more similar; gives the same ranking as cosine for normalised vectors).

## 2.5 Asymmetric search and task prefixes

A question ("How much holiday?") and its answer ("Employees receive 23 days…") look different: one is short and asks, the other is long and tells. Some embedding models are trained with **prefixes** so they know which side is which:

```
stored chunks:  "search_document: Employees receive 23 days…"
user question:  "search_query: How much holiday can I take?"
```

`nomic-embed-text` uses these prefixes. Other models (e.g. BGE, E5) use different ones, and some need none. **Always read your embedding model's card.**

## 2.6 Hands-on

```bash
python3 02-embeddings/lesson.py
```

| Part | What you see |
|---|---|
| 1 | Cosine similarity on toy 2-D vectors, computed by hand |
| 2 | A real embedding: 768 numbers, length 1.0 |
| 3 | A similarity matrix: paraphrases score higher than unrelated sentences |
| 4 | Keyword search vs embedding search on the same query |

### What we observed (see `sample_output.txt`)

- "How many vacation days do I get?" vs "What's my time-off allowance?" = **0.62**, even with almost no shared words.
- "battery lasts 9 hours" vs "run on one charge" = **0.61**, compared with 0.32 for battery vs Rotterdam.
- Query "How much holiday can I take?" has **0 useful keyword matches**, yet the embedding ranked the leave policy **#1**.
- ⚠️ **But the margin was tiny: 0.596 vs 0.593** (the password policy came second!). Scores from small models sit in a narrow band (roughly 0.3 to 0.7 here). This is normal, and it's why:
  1. You never trust a single top result; you take the **top-k** (e.g. 3 to 5).
  2. You don't hard-code a similarity cut-off without testing on your own data.
  3. Advanced systems add **hybrid search and reranking** (Lesson 7).

## 2.7 Choosing an embedding model (practical notes)

| Consideration | Why it matters |
|---|---|
| **Quality on retrieval benchmarks** | See the MTEB leaderboard (Hugging Face) → "Retrieval" tab |
| **Language** | Multilingual data needs a multilingual model |
| **Max input length** | Text beyond the limit is cut off silently |
| **Dimension** | More dimensions: often better quality, but more storage and slower search |
| **Local vs API** | Local (Ollama, sentence-transformers) is private and free; APIs (OpenAI, Voyage, Cohere) are often stronger |
| **⚠️ Consistency** | **Documents and queries MUST be embedded with the same model.** If you change models, you must re-embed everything. |

## 2.8 Key takeaways

- An embedding turns text into a vector; **similar meaning means similar direction**.
- **Cosine similarity** measures that direction match; on normalised vectors it's just a dot product.
- Embeddings beat keywords on paraphrases, but scores are *relative*: use rankings and top-k, not absolute thresholds.
- Same model for documents and queries, always.

## 2.9 Try it yourself
1. Add your own sentences to `part3_similarity_matrix()` and predict the scores before running.
2. In Part 4, remove the `search_query:` / `search_document:` prefixes. Do the rankings change?
3. Change the embedding model: `EMBED_MODEL=all-minilm python3 02-embeddings/lesson.py` (384 dimensions). Compare quality.

➡️ Next: [Lesson 3: Chunking](../03-chunking/README.md). Documents are too long to embed whole, so how should we cut them?
