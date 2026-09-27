# Lesson 1: Why RAG Exists

> **Goal:** understand the problem RAG solves and the core idea behind it, before touching any math.

---

## 1.1 What an LLM actually knows

A Large Language Model (LLM) such as Llama, GPT or Claude is trained on a huge snapshot of text. After training, its knowledge is **frozen inside its weights** (billions of numbers). That has four consequences:

| Limitation | What it means | Example |
|---|---|---|
| **Knowledge cutoff** | It knows nothing after its training date. | "Who won yesterday's match?" |
| **No private data** | It has never seen *your* documents. | "What is our company's leave policy?" |
| **Hallucination** | When it doesn't know, it often invents a fluent, confident, wrong answer. | Makes up "20 days of leave". |
| **No sources** | It can't tell you *where* a fact came from. | You can't verify it. |

## 1.2 Three ways to give an LLM new knowledge

1. **Fine-tuning**: retrain the model on your data.
   - Expensive, slow, has to be redone whenever data changes, and it *still* hallucinates. It's good for teaching **style/behaviour** and poor for teaching **facts**.
2. **Put everything in the prompt** ("context stuffing").
   - Works for small data. Breaks when you have thousands of documents: context windows have limits, cost grows with every token, and models get worse at finding facts buried in very long prompts ("lost in the middle").
3. **RAG (Retrieval-Augmented Generation)**:
   - Keep your documents **outside** the model. For each question, **retrieve** only the few relevant pieces and put *those* in the prompt.

## 1.3 The core idea, as an analogy

> An LLM without RAG is a smart student taking a **closed-book exam**.
> RAG turns it into an **open-book exam**: before answering, the student looks up the right pages in the textbook.

The student (LLM) still does the reasoning and writing. The library (your documents) supplies the facts.

## 1.4 The name, word by word

- **Retrieval**: find the pieces of text relevant to the question.
- **Augmented**: add (augment) those pieces to the prompt.
- **Generation**: the LLM generates an answer using them.

## 1.5 The RAG pipeline at a glance

RAG has **two phases**:

```
 ┌──────────────────── PHASE A: INDEXING (done once, offline) ───────────────────┐
 │                                                                               │
 │  Documents ──► Split into chunks ──► Embed each chunk ──► Store in vector DB  │
 │  (PDF, md…)     (Lesson 3)            (Lesson 2)           (Lesson 4)          │
 └───────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────── PHASE B: QUERYING (every question) ───────────────────────┐
 │                                                                               │
 │  Question ──► Embed question ──► Find most similar chunks ──► Build prompt    │
 │                                                                  │            │
 │                                   Answer ◄── LLM generates ◄─────┘            │
 │                                   (Lesson 5)                                  │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Every later lesson zooms into one box of this diagram.

## 1.6 Benefits of RAG

- ✅ **Up to date**: update a document and answers change right away. No retraining.
- ✅ **Private data**: works on your internal documents.
- ✅ **Fewer hallucinations**: the model answers from the evidence you give it.
- ✅ **Citations**: you know which chunk each answer came from.
- ✅ **Cheap**: no training. Just a database and some prompts.
- ✅ **Access control**: you can filter which documents a user is allowed to retrieve.

## 1.7 Hands-on

Run:

```bash
python3 01-why-rag/lesson.py
```

The script asks `llama3.1` three questions about the fictional company **Nimbus Robotics** in two ways, then runs one more experiment:

1. **Without context**: the model has never heard of Nimbus. Watch it guess or hallucinate.
2. **With the right paragraph pasted into the prompt**: now it answers correctly.
3. **Forced answer (experiment C)**: when pushed to "reply with only a number", the model invents one.

Step 2 is RAG done **by hand**. We picked the right paragraph ourselves. The rest of the course teaches the computer to find that paragraph automatically, among thousands.

### What we observed when running it (see `sample_output.txt`)

| Experiment | Result | Lesson |
|---|---|---|
| A: no context | Model said "I couldn't find any information…" | Newer models sometimes *refuse* rather than invent. Either way, no answer. |
| B: with context, Q1 & Q3 | "23 days", "Tuesdays and Thursdays" ✅ | Evidence in the prompt means a correct, grounded answer. |
| B: with context, Q2 | **"I don't know… it doesn't mention the Zephyr-3"** ❌ | The pasted snippet talked about E-417 but never said *which product*. Our strict prompt made the model refuse. **A chunk that loses its surrounding context loses meaning.** We fix this in Lesson 3 (contextual chunk headers). |
| C: forced number | "20", "20", once a refusal. Never 23 | **Hallucination**: confident and wrong. |

## 1.8 Key takeaways

- LLMs have frozen, public, un-cited knowledge.
- RAG = **look it up first, then answer**.
- The hard part of RAG is **retrieval**: finding the *right* text. That's why most of this course is about retrieval.

➡️ Next: [Lesson 2: Embeddings](../02-embeddings/README.md), which covers how a computer can tell that two texts mean similar things.
