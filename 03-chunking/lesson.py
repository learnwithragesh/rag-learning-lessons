"""
Lesson 3: Chunking.

Part 1: the four chunking strategies applied to the same document
Part 2: why chunk size matters (a whole document vs a focused chunk)
Part 3: contextual headers fix the "E-417 but which product?" problem from Lesson 1
Part 4: chunk the whole knowledge base (what the next lessons will use)

Run from the course root:   python3 03-chunking/lesson.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.chunking import (
    chunk_all,
    chunk_fixed_chars,
    chunk_markdown,
    chunk_sentences,
    chunk_words,
    load_documents,
)
from common.llm import embed


def show(title, chunks, limit=3):
    print(f"\n--- {title}: {len(chunks)} chunks (showing {min(limit, len(chunks))}) ---")
    for i, c in enumerate(chunks[:limit]):
        text = c["text"] if isinstance(c, dict) else c
        print(f"  [{i}] {text!r}")


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def part1_strategies(handbook):
    print("=" * 80)
    print("PART 1: four chunking strategies on employee_handbook.md")
    print("=" * 80)
    text = handbook["text"]
    show("1) Fixed 300 characters", chunk_fixed_chars(text, 300))
    print("  ^ cuts words and sentences in half, e.g. mid-word at the edges.")
    show("2) 60-word windows, 15-word overlap", chunk_words(text, 60, 15), limit=2)
    print("  ^ the end of chunk [0] is repeated at the start of chunk [1] (overlap).")
    show("3) Sentence packing (<=60 words)", chunk_sentences(text, 60))
    print("  ^ never cuts a sentence, but can still mix two topics in one chunk.")
    show("4) Markdown sections + contextual header", chunk_markdown(handbook))
    print("  ^ one topic per chunk, and each chunk says where it came from.")


def part2_size_matters(handbook):
    print("\n" + "=" * 80)
    print("PART 2: why not just embed the whole document?")
    print("=" * 80)
    query = "search_query: Is there a stipend for my home office desk?"
    whole = "search_document: " + handbook["text"]
    focused = [c for c in chunk_markdown(handbook) if c["section"] == "Home Office Stipend"][0]
    q, w, f = embed([query, whole, "search_document: " + focused["text"]])
    print(f"  similarity(query, WHOLE handbook)        = {cos(q, w):.3f}")
    print(f"  similarity(query, 'Home Office' chunk)   = {cos(q, f):.3f}")
    print("  -> one vector for 5 topics is a blurry average of all of them.")
    print("     A focused chunk matches more strongly, and it's cheaper to put")
    print("     in the prompt.")


def part3_contextual_header():
    print("\n" + "=" * 80)
    print("PART 3: contextual headers (the fix for Lesson 1, Q2)")
    print("=" * 80)
    [errors_doc] = [d for d in load_documents() if d["source"] == "zephyr3_error_codes.md"]
    with_h = [c for c in chunk_markdown(errors_doc, add_header=True) if "E-417" in c["text"]][0]
    without = [c for c in chunk_markdown(errors_doc, add_header=False) if "E-417" not in c["text"] and "fiducial" in c["text"]][0]
    print(f"  WITHOUT header: {without['text'][:90]!r}...")
    print(f"  WITH header   : {with_h['text'][:90]!r}...")
    print("  Without the header, the chunk text never says 'E-417' or 'Zephyr-3'")
    print("  (that was in the heading line, which chunking removed)!\n")

    query = "search_query: What does error E-417 mean on the Zephyr-3?"
    q, a, b = embed([query, "search_document: " + without["text"], "search_document: " + with_h["text"]])
    print(f"  similarity(query, chunk WITHOUT header) = {cos(q, a):.3f}")
    print(f"  similarity(query, chunk WITH header)    = {cos(q, b):.3f}")
    print("  -> the header gives both better retrieval AND gives the LLM the")
    print("     context it needs to answer. (In Lesson 1 it refused for lack of it.)")


def part4_whole_kb():
    print("\n" + "=" * 80)
    print("PART 4: chunk the whole knowledge base")
    print("=" * 80)
    chunks = chunk_all()
    lengths = [len(c["text"].split()) for c in chunks]
    print(f"  {len(load_documents())} documents -> {len(chunks)} chunks")
    print(f"  words per chunk: min={min(lengths)}, avg={sum(lengths)/len(lengths):.0f}, max={max(lengths)}")
    print("  Example chunk with metadata:")
    for k, v in chunks[5].items():
        print(f"    {k:8}: {v!r}")


if __name__ == "__main__":
    [handbook] = [d for d in load_documents() if d["source"] == "employee_handbook.md"]
    part1_strategies(handbook)
    part2_size_matters(handbook)
    part3_contextual_header()
    part4_whole_kb()
