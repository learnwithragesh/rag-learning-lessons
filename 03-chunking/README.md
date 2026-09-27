# Lesson 3: Chunking

> **Goal:** learn how to cut documents into pieces ("chunks") that are good units for retrieval. Chunking is one of the settings with the **biggest effect on RAG quality**, and beginners often overlook it.

---

## 3.1 Why split documents at all?

We embed text and retrieve by similarity. Why not embed each whole document?

1. **Blurry vectors.** One vector for a document about five topics is an *average* of all five, so it matches none of them strongly.
   *Measured in the lesson: query "home office stipend" vs whole handbook = **0.68**; vs the focused chunk = **0.84**.*
2. **Model input limits.** Embedding models have a maximum input length (often 512 to 8192 tokens). Longer text gets **silently cut off**.
3. **Prompt budget.** We paste retrieved text into the LLM prompt. Five 50-word chunks are cheaper and more focused than five 20-page documents.
4. **Precise citations.** "Handbook › Sick Leave" is more useful than "Handbook".

## 3.2 The chunk-size trade-off

```
  too SMALL                              too LARGE
  ◄──────────────────────────────────────────────────►
  • precise matches                      • full context
  • but loses context                    • but blurry embeddings
    ("it expires March 31": what does?)  • wastes prompt tokens
  • answer may span several chunks       • irrelevant text distracts LLM
```

**Typical starting point:** 200 to 500 tokens (roughly 150 to 400 words) per chunk, with 10 to 20% overlap. Then **measure** (Lesson 6) and tune for your data. There's no universal best size.

> **Token vs word:** LLMs count *tokens* (word pieces). Rule of thumb for English: 1 token ≈ ¾ word, so 100 tokens ≈ 75 words.

## 3.3 Chunking strategies

| # | Strategy | How | Pros | Cons |
|---|---|---|---|---|
| 1 | **Fixed-size characters** | cut every N chars | trivial | cuts words and sentences in half |
| 2 | **Fixed-size tokens/words + overlap** | sliding window of N words, repeating the last M | simple and predictable; overlap protects boundary facts | still ignores meaning and structure |
| 3 | **Sentence-aware** | pack whole sentences up to a size limit | no broken sentences | may mix two topics |
| 4 | **Structure-aware** | split on headings, paragraphs, HTML tags, code functions | one topic per chunk | needs structured input |
| 5 | **Recursive** (LangChain's `RecursiveCharacterTextSplitter`) | try to split on `\n\n`, then `\n`, then `. `, then ` `, until chunks fit | good general default | not structure-aware |
| 6 | **Semantic** | embed sentences; start a new chunk where similarity between neighbours drops | follows topic shifts | slower, needs tuning |
| 7 | **Agentic / LLM-based** | ask an LLM to split into self-contained propositions | high quality | expensive |

This course implements **1 to 4** from scratch in [`common/chunking.py`](../common/chunking.py).

### What overlap looks like

```
chunk 0: [w1 ........................... w60]
chunk 1:                      [w46 ................................ w105]
                               └ overlap (15 words) ┘
```
If a key fact sits on a boundary, at least one chunk contains it whole.

## 3.4 Contextual chunk headers (important!)

When you cut a document, each chunk **loses its surroundings**. Remember Lesson 1, Q2: the E-417 paragraph never said "Zephyr-3", so the model refused to answer.

**Fix:** put the document title and section heading at the start of every chunk:

```
Zephyr-3 Error Codes and Troubleshooting > E-417: Lidar Calibration Drift
The lidar's measured distances disagree with known fiducial marker positions...
```

*Measured in the lesson: similarity for "What does error E-417 mean on the Zephyr-3?" went from **0.518 → 0.584** just by adding the header.* Without the header, the words "E-417" and "Zephyr-3" didn't appear in the chunk at all.

> **Advanced version:** *Contextual Retrieval* (published by Anthropic, 2024) uses an LLM to write a 1 to 2 sentence summary of where each chunk fits in its document, and prepends that. It costs more but reduces retrieval failures a lot.

## 3.5 Metadata: keep it with every chunk

Every chunk in our code is a dict:

```python
{"id": "employee_handbook.md#1",
 "text": "Nimbus Robotics — Employee Handbook > Sick Leave\nEmployees may take...",
 "source": "employee_handbook.md",
 "section": "Sick Leave"}
```

Metadata makes the following possible:
- **Citations**: "Source: employee_handbook.md › Sick Leave"
- **Filtering**: "only search product manuals", "only documents this user may see"
- **Updates**: delete and re-index all chunks from a changed file

Real systems also store page number, URL, last-modified date, author, access level, and so on.

## 3.6 Real-world document parsing (a note)

Our data is clean Markdown. In real projects, **step 0 is parsing**: PDFs (tables, columns, headers/footers), Word, HTML, scanned images (OCR). Tools include `pypdf`, `pymupdf`, `unstructured`, `docling`, `LlamaParse`. Garbage in means garbage out: bad parsing ruins everything downstream.

## 3.7 Hands-on

```bash
python3 03-chunking/lesson.py
```

| Part | What you see |
|---|---|
| 1 | The same handbook chunked 4 ways. Spot the broken words in strategy 1 and the mixed topics in 3. |
| 2 | Whole document vs focused chunk similarity (0.68 vs 0.84) |
| 3 | Contextual header improves E-417 retrieval (0.518 vs 0.584) |
| 4 | Whole knowledge base: 5 docs → **24 chunks**, ~51 words each, with metadata |

## 3.8 Key takeaways
- Chunk so that **each chunk is about one thing and makes sense on its own**.
- Respect document structure (headings, paragraphs) when you can; use overlap when you can't.
- **Prepend context** (title > section) to each chunk.
- Keep **metadata** with every chunk.
- Chunk size is a setting you tune by measuring, not by guessing.

## 3.9 Try it yourself
1. In `part1_strategies`, change the fixed size to 100 and to 1000. Which is more readable?
2. Set `overlap=0` in strategy 2 and look for a fact that gets split.
3. Write a **recursive splitter**: split on `"\n\n"`; if a piece is still too long, split it on `". "`; and so on.

➡️ Next: [Lesson 4: Vector Store and Retrieval](../04-vector-store/README.md): store all 24 chunk embeddings and search them.
