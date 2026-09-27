"""
Evaluation helpers (Lesson 6). Reused by Lesson 7 to compare retrievers.

Retrieval metrics (need only the retriever, fast):
    hit@k : fraction of questions whose correct chunk is in the top-k
    MRR   : mean of 1/rank of the correct chunk (0 if not found in top-k)

Generation metrics (need the LLM, slower):
    correctness  : answer contains all the required facts (string check)
    refusal      : unanswerable questions correctly get "I don't know"
    faithfulness : an LLM judge checks every claim is supported by the passages
"""

import json
import re
from pathlib import Path
from typing import Callable

from common.llm import chat
from common.rag import answer, format_context

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
REFUSAL = "i don't know"


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    return json.loads(path.read_text())


# ------------------------------------------------------------------ retrieval
def evaluate_retrieval(retriever: Callable, eval_set: list[dict], k: int = 4) -> dict:
    answerable = [e for e in eval_set if e["expected_section"]]
    hits, rr, misses = 0, 0.0, []
    for e in answerable:
        results = retriever(e["question"], k)
        sections = [r["section"] for r in results]
        if e["expected_section"] in sections:
            rank = sections.index(e["expected_section"]) + 1
            hits += 1
            rr += 1 / rank
        else:
            misses.append(e["question"])
    n = len(answerable)
    return {"hit@k": hits / n, "mrr": rr / n, "k": k, "n": n, "misses": misses}


# ------------------------------------------------------------------ generation
# Version 1 of this judge asked for a one-word verdict and was wrong ~half the
# time with llama3.1 (see Lesson 6 README, section 6.5). Making the judge quote
# evidence first, giving it the question, and parsing defensively fixed that.
JUDGE_PROMPT = """You are a fact-checker. Decide whether the ANSWER to the QUESTION is supported by the PASSAGES.

PASSAGES:
{context}

QUESTION: {question}
ANSWER: {answer}

Think step by step:
1. List each factual claim the ANSWER makes (read it together with the QUESTION).
2. For each claim, quote the sentence from the PASSAGES that states it, or write "NOT FOUND".
   Paraphrases count as supported. Citation markers like [1] are not claims.
3. On the final line write exactly "VERDICT: SUPPORTED" if every claim was found, otherwise "VERDICT: UNSUPPORTED"."""


def parse_verdict(judge_output: str) -> bool:
    """Take the LAST verdict, tolerating markdown like '**VERDICT:** SUPPORTED'."""
    found = re.findall(r"VERDICT\W*(UNSUPPORTED|SUPPORTED)", judge_output.upper().replace("*", ""))
    return bool(found) and found[-1] == "SUPPORTED"


def judge_faithfulness(question: str, answer_text: str, chunks: list[dict]) -> bool:
    """LLM-as-a-judge: does the answer stick to the retrieved evidence?"""
    if REFUSAL in _normalise(answer_text):
        return True  # a refusal makes no claims, so there is nothing to fact-check
    out = chat(JUDGE_PROMPT.format(context=format_context(chunks), question=question,
                                   answer=answer_text))
    return parse_verdict(out)


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).lower()


def evaluate_generation(retriever: Callable, eval_set: list[dict], k: int = 4,
                        verbose: bool = True) -> dict:
    rows = []
    for e in eval_set:
        r = answer(e["question"], retriever, k=k)
        text = _normalise(r["answer"])
        refused = REFUSAL in text
        if e["expected_section"]:
            correct = all(_normalise(f) in text for f in e["must_contain"])
        else:
            correct = refused  # unanswerable: the right behaviour is to refuse
        faithful = judge_faithfulness(e["question"], r["answer"], r["chunks"])
        rows.append({"question": e["question"], "answer": r["answer"],
                     "correct": correct, "faithful": faithful,
                     "answerable": bool(e["expected_section"])})
        if verbose:
            flag = ("OK  " if correct else "FAIL") + (" " if faithful else " UNFAITHFUL")
            print(f"  [{flag}] {e['question']}\n           -> {r['answer'][:110]}")
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "correctness": sum(r["correct"] for r in ans) / len(ans),
        "refusal_accuracy": sum(r["correct"] for r in unans) / len(unans) if unans else None,
        "faithfulness": sum(r["faithful"] for r in rows) / len(rows),
        "rows": rows,
    }
