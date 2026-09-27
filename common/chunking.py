"""
Loading documents and splitting them into chunks (Lesson 3).

Every chunk is a dict:
    {"id": "employee_handbook.md#3", "text": "...", "source": "employee_handbook.md",
     "section": "Remote Work Policy"}

Keeping metadata (source, section) next to the text is what later lets us
cite sources and filter results.
"""

import re
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"


def load_documents(folder: Path = KB_DIR) -> list[dict]:
    """Read every .md / .txt file in the folder -> [{"source", "text"}]."""
    docs = []
    for path in sorted(folder.glob("*")):
        if path.suffix in {".md", ".txt"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


# --------------------------------------------------------------------------
# Strategy 1: fixed-size character chunks (the naive baseline)
# --------------------------------------------------------------------------
def chunk_fixed_chars(text: str, size: int = 300) -> list[str]:
    """Cut every `size` characters, ignoring words and sentences."""
    return [text[i : i + size] for i in range(0, len(text), size)]


# --------------------------------------------------------------------------
# Strategy 2: fixed-size word windows WITH overlap
# --------------------------------------------------------------------------
def chunk_words(text: str, size: int = 60, overlap: int = 15) -> list[str]:
    """Windows of `size` words; each window repeats the last `overlap` words
    of the previous one so a fact sitting on a boundary isn't split in half."""
    assert 0 <= overlap < size
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------
# Strategy 3: sentence-aware packing
# --------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    # Simple rule: split after . ! ? followed by whitespace. Real projects may
    # use nltk / spaCy, which handle abbreviations like "e.g." better.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_sentences(text: str, max_words: int = 60) -> list[str]:
    """Pack whole sentences into a chunk until adding one more would exceed max_words."""
    chunks, current = [], []
    for sent in split_sentences(text):
        if current and len(" ".join(current + [sent]).split()) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------
# Strategy 4: structure-aware (markdown sections) + contextual header
# --------------------------------------------------------------------------
def split_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (document_title, [(section_heading, section_body), ...])."""
    title = ""
    sections = []
    heading, body = None, []
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(body).strip()))
    return title, sections


def chunk_markdown(doc: dict, max_words: int = 120, add_header: bool = True) -> list[dict]:
    """One chunk per markdown section (long sections are split further by
    sentences). If add_header, each chunk starts with
    'Document title > Section heading' so it still makes sense alone."""
    title, sections = split_markdown_sections(doc["text"])
    chunks = []
    for heading, body in sections:
        for piece in chunk_sentences(body, max_words=max_words):
            text = f"{title} > {heading}\n{piece}" if add_header else piece
            chunks.append(
                {
                    "id": f"{doc['source']}#{len(chunks)}",
                    "text": text,
                    "source": doc["source"],
                    "section": heading,
                }
            )
    return chunks


def chunk_all(docs: list[dict] | None = None, **kwargs) -> list[dict]:
    """Chunk the whole knowledge base with the recommended strategy (4)."""
    docs = docs if docs is not None else load_documents()
    return [c for d in docs for c in chunk_markdown(d, **kwargs)]
