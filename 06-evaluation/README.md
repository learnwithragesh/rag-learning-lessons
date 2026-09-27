# Lesson 6: Evaluating a RAG System

> **Goal:** stop eyeballing and start **measuring**. Every change you make later (chunk size, k, hybrid search, prompt, model) should be judged by numbers on a fixed test set.

---

## 6.1 Why evaluation matters

Without evaluation, RAG tuning is guesswork. "It looks better on the 3 questions I tried" often hides a regression on 20 other questions. The loop is:

```
  build eval set ─► measure baseline ─► change ONE thing ─► measure again ─► keep or revert
         ▲                                                                        │
         └──────────────────────────── repeat ────────────────────────────────────┘
```

## 6.2 The evaluation set

[`data/eval_set.json`](../data/eval_set.json): **27 questions**, each labelled with:

```json
{"question": "The robot says E-417, what should I do?",
 "expected_section": "E-417: Lidar Calibration Drift",   ← which chunk SHOULD be retrieved
 "must_contain": ["Auto-Calibrate"]}                      ← fact(s) the answer MUST include
```

It deliberately mixes:
- **Paraphrases** ("time off for vacations" vs "annual leave")
- **Exact identifiers** (E-101, E-204, E-417, E-520, NimbusConnect)
- **Unanswerable** questions (`expected_section: null`), where the correct behaviour is to refuse

**Building a good eval set** (real projects):
- Collect **real user questions** (logs, support tickets), not only ones you invent.
- Cover every document type and every question type (lookup, comparison, multi-hop, unanswerable).
- 50 to 200 questions is a solid start. LLMs can *draft* questions from your chunks, but a human should review them.
- **Never tune on the test set only**. Keep a held-out portion.

## 6.3 Evaluate the two halves separately

A RAG answer can fail in **retrieval** or in **generation**. Measure them separately, or you won't know what to fix.

### A) Retrieval metrics (fast, no LLM needed)

| Metric | Definition | Intuition |
|---|---|---|
| **Hit@k** (a.k.a. Recall@k) | fraction of questions whose correct chunk appears in the top-k | "Did the right evidence make it into the prompt?" |
| **MRR** (Mean Reciprocal Rank) | average of 1/rank of the correct chunk (0 if missing) | rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25. Rewards ranking it *high*. |
| Precision@k | fraction of the top-k that are relevant | Useful when many chunks can be relevant |
| nDCG | ranking quality with graded relevance | Standard in search research |

### B) Generation metrics (slow, needs an LLM)

| Metric | Question it answers | How we compute it |
|---|---|---|
| **Correctness** | Is the answer right? | Answer contains all `must_contain` facts (string check) |
| **Refusal accuracy** | Does it say "I don't know" when the docs lack the answer? | Checks for the refusal phrase on unanswerable questions |
| **Faithfulness / groundedness** | Is every claim supported by the retrieved passages (no hallucination)? | **LLM-as-a-judge** |
| Answer relevance | Does it actually address the question? | LLM judge (not implemented here) |
| Context relevance | Were the retrieved chunks useful? | LLM judge (not implemented here) |

> **Correct ≠ faithful.** An answer can be correct by luck (from the model's memory) but unfaithful to the passages, or faithful to a *wrong* passage. You want both.

## 6.4 LLM-as-a-judge

For fuzzy criteria like "is this supported?", we ask an LLM to grade. It scales, but **the judge is itself a model that makes mistakes.**

## 6.5 Case study: evaluate your evaluator ⚠️

This happened while building this lesson, and it's the most important part of it.

**Judge v1** asked llama3.1 for a single word: `SUPPORTED` or `UNSUPPORTED`.
→ Faithfulness came out at **0.56**. But reading the rows, it had marked obviously correct answers like *"Ines Varga and Tomasz Brenner [1]"* as UNSUPPORTED, and even flagged "I don't know" refusals, which the prompt explicitly said count as supported.

We fixed the **judge**, not the RAG system:

| Fix | Why it helps |
|---|---|
| Ask the judge to **list claims and quote the supporting sentence** before deciding | Reasoning before the verdict ("chain of thought") makes small models far more accurate |
| Give the judge the **question** too | "Within 1 hour" can't be checked without knowing what was asked |
| **Parse defensively** | The judge wrote `**VERDICT:** SUPPORTED` in markdown bold, which broke our exact-string check. Now: regex, ignore `*`, take the *last* verdict |
| Handle **refusals in code** | A refusal makes no claims, so there's no need to ask an LLM |

Then we **calibrated** it on 12 answers where we *know* the truth ([`data/judge_calibration.json`](../data/judge_calibration.json): 7 true answers, 5 with planted errors such as "12 characters" instead of 14, or "climbs stairs"):
→ **Judge v2 agrees with human labels on 11/12**, catches all 5 planted errors, and is slightly *over-strict* (it rejected "Within 1 hour").

**Result:** faithfulness went from **0.56 to 0.89**. *Same RAG system, better ruler.*

> **Rule:** before trusting an LLM-judge metric, test the judge on labelled examples. Tools like RAGAS, TruLens, DeepEval and Phoenix package these judges, and the same rule applies to them.

## 6.6 Hands-on

```bash
python3 06-evaluation/lesson.py                   # everything (~5 min)
python3 06-evaluation/lesson.py --retrieval-only  # just retrieval metrics (seconds)
```

### Baseline results: vector search (see `sample_output.txt`, `results_vector.json`)

**Retrieval:**

| k | Hit@k | MRR |
|---|---|---|
| 1 | 0.88 | 0.88 |
| 3 | 0.92 | 0.90 |
| **4** | **0.96** | **0.91** |
| 5 | 0.96 | 0.91 |
| 10 | 1.00 | 0.91 |

Only miss at k=4: **"The robot says E-417, what should I do?"** (our known exact-code problem).

**Generation (k=4):**

| Metric | Score |
|---|---|
| Correctness (24 answerable) | **0.96** (23/24; the one failure is E-417, caused by retrieval) |
| Refusal accuracy (3 unanswerable) | **1.00** |
| Faithfulness (judge v2) | **0.89** (some flags are judge over-strictness, e.g. "Within 1 hour") |

**Reading the numbers:** the single correctness failure lines up exactly with the single retrieval miss. **Generation is fine; retrieval is the bottleneck.** So Lesson 7 improves retrieval.

> ⚠️ Our knowledge base is tiny (24 chunks), so scores are high. On real data with thousands of similar chunks, baseline hit@k is often 0.6 to 0.8, and these techniques matter much more.

## 6.7 Key takeaways
- Build a **labelled eval set** early, including unanswerable and exact-match questions.
- Measure **retrieval** (hit@k, MRR) and **generation** (correctness, faithfulness, refusal) **separately**.
- LLM judges are useful but fallible: **calibrate them** on known cases, make them reason before deciding, parse defensively.
- Change one thing at a time and re-measure.

## 6.8 Try it yourself
1. Add 5 questions of your own to `eval_set.json` (include one multi-part question) and rerun.
2. Run `--retrieval-only` after changing `max_words` in `chunk_all()` (rebuild the index with Lesson 4 first). Does hit@k change?
3. Add a deliberately wrong answer to `judge_calibration.json`, one that's subtly wrong, like "2,500 hours" instead of "3,000 hours". Does the judge catch it?

➡️ Next: [Lesson 7: Advanced Retrieval](../07-advanced-retrieval/README.md). We fix the E-417 problem with hybrid search, reranking and query rewriting.
