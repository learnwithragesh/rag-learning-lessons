"""
Tiny helper for talking to the local Ollama server.

We deliberately use ONLY the Python standard library (urllib + json) so there
is no magic: every "AI call" in this course is just an HTTP POST to
http://localhost:11434.

  embed(texts)  -> list of vectors (one list[float] per text)
  chat(prompt)  -> the model's text answer
"""

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1")


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL} ({e}).\n"
            "Start it with:  ollama serve"
        )


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Turn a list of strings into a list of embedding vectors."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(_post("/api/embed", {"model": EMBED_MODEL, "input": batch})["embeddings"])
    return vectors


def chat(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    """Send one prompt to the chat model and return its answer as text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    out = _post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    return out["message"]["content"].strip()
