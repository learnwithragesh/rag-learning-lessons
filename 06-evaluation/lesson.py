"""
Lesson 6: Evaluating a RAG system.

Part 1: retrieval metrics (hit@k, MRR) for several k, which is fast
Part 2: evaluate the EVALUATOR: is the LLM judge itself trustworthy? (12 labelled cases)
Part 3: generation metrics (correctness, refusal, faithfulness), which is slower (~5 min on a laptop)

Run from the course root:   python3 06-evaluation/lesson.py
Skip the slow part:         python3 06-evaluation/lesson.py --retrieval-only
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.evaluate import (
    evaluate_generation,
    evaluate_retrieval,
    judge_faithfulness,
    load_eval_set,
)
from common.vector_store import VectorStore


def main():
    store = VectorStore.load()
    eval_set = load_eval_set()
    n_ans = sum(1 for e in eval_set if e["expected_section"])
    print(f"Eval set: {len(eval_set)} questions ({n_ans} answerable, {len(eval_set) - n_ans} unanswerable)\n")

    print("=" * 80)
    print("PART 1: retrieval metrics (vector search)")
    print("=" * 80)
    print(f"  {'k':>3}  {'hit@k':>6}  {'MRR':>6}")
    for k in [1, 3, 4, 5, 10]:
        m = evaluate_retrieval(store.search, eval_set, k=k)
        print(f"  {k:>3}  {m['hit@k']:>6.2f}  {m['mrr']:>6.2f}")
    m = evaluate_retrieval(store.search, eval_set, k=4)
    print(f"\n  Misses at k=4 ({len(m['misses'])}):")
    for q in m["misses"]:
        print(f"    - {q}")

    if "--retrieval-only" in sys.argv:
        return

    print("\n" + "=" * 80)
    print("PART 2: calibrate the LLM judge on answers whose truth we KNOW")
    print("=" * 80)
    cases = json.loads((Path(__file__).resolve().parent.parent / "data" / "judge_calibration.json").read_text())
    agree = 0
    for c in cases:
        verdict = judge_faithfulness(c["question"], c["answer"], store.search(c["question"], 4))
        agree += verdict == c["supported"]
        mark = "agree   " if verdict == c["supported"] else "DISAGREE"
        print(f"  [{mark}] truth={'supported  ' if c['supported'] else 'UNSUPPORTED'} {c['answer'][:60]}")
    print(f"\n  judge agrees with human labels on {agree}/{len(cases)} cases")

    print("\n" + "=" * 80)
    print("PART 3: generation metrics (vector search, k=4), with the LLM as judge")
    print("=" * 80)
    g = evaluate_generation(store.search, eval_set, k=4)
    print(f"\n  correctness (answerable)   : {g['correctness']:.2f}")
    print(f"  refusal accuracy (unanswer.): {g['refusal_accuracy']:.2f}")
    print(f"  faithfulness (LLM judge)   : {g['faithfulness']:.2f}")

    out = Path(__file__).parent / "results_vector.json"
    out.write_text(json.dumps({"retrieval_k4": m, **g}, indent=1))
    print(f"\n  full per-question results saved to {out.name}")


if __name__ == "__main__":
    main()
