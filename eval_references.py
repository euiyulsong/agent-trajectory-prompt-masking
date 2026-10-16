#!/usr/bin/env python3
"""Reference-based lexical, semantic, and optional LLM-judge evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def ngram_f1(prediction: str, reference: str, n: int) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    pgrams = Counter(tuple(p[i : i + n]) for i in range(max(0, len(p) - n + 1)))
    rgrams = Counter(tuple(r[i : i + n]) for i in range(max(0, len(r) - n + 1)))
    overlap = sum((pgrams & rgrams).values())
    pcount, rcount = sum(pgrams.values()), sum(rgrams.values())
    if pcount == 0 and rcount == 0:
        return 1.0
    if overlap == 0:
        return 0.0
    precision, recall = overlap / pcount, overlap / rcount
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_pairs(prediction_path: Path, eval_path: Path) -> list[dict[str, str]]:
    predictions = {str(row["task_id"]): row for row in load_jsonl(prediction_path)}
    refs = load_jsonl(eval_path)
    pairs = []
    for row in refs:
        task_id = str(row["task_id"])
        if task_id not in predictions:
            continue
        pred = predictions[task_id].get("prediction", "")
        ref = row["messages"][-1].get("content", "")
        pairs.append({"task_id": task_id, "prediction": pred, "reference": ref})
    return pairs


def rouge_scores(pairs: list[dict[str, str]]) -> dict[str, Any]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for pair in pairs:
        scores = scorer.score(pair["reference"], pair["prediction"])
        rows.append({
            "task_id": pair["task_id"],
            **{name: {"precision": score.precision, "recall": score.recall, "f1": score.fmeasure}
               for name, score in scores.items()},
        })
    result = {name: {key: sum(row[name][key] for row in rows) / len(rows) for key in ("precision", "recall", "f1")}
              for name in ("rouge1", "rouge2", "rougeL")}
    return {"n": len(rows), "aggregation": "mean per-example precision/recall/F1", "scores": result, "per_task": rows}


def bertscore(pairs: list[dict[str, str]], model_type: str, batch_size: int) -> dict[str, Any]:
    from bert_score import score

    predictions = [pair["prediction"] for pair in pairs]
    references = [pair["reference"] for pair in pairs]
    precision, recall, f1 = score(
        predictions, references, model_type=model_type, batch_size=batch_size,
        lang="en", verbose=True, rescale_with_baseline=False,
    )
    rows = [
        {"task_id": pair["task_id"], "precision": float(p), "recall": float(r), "f1": float(s)}
        for pair, p, r, s in zip(pairs, precision, recall, f1)
    ]
    return {
        "n": len(rows), "model_type": model_type, "baseline_rescaling": False,
        "scores": {key: sum(row[key] for row in rows) / len(rows) for key in ("precision", "recall", "f1")},
        "per_task": rows,
    }


def llm_judge(pairs: list[dict[str, str]], model: str, base_url: str | None,
              api_key: str | None, limit: int | None) -> dict[str, Any]:
    """Pairwise reference-grounded judge. Requires explicit OpenAI-compatible endpoint."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
    selected = pairs[:limit] if limit else pairs
    results = []
    system = (
        "You are a strict evaluator. Compare a model response with a reference answer for the user's task. "
        "Judge only task-relevant correctness, completeness, and unsupported claims. Ignore wording differences. "
        "Do not reward copied reasoning if the actual requested result is wrong. Return JSON only with keys: "
        "score (integer 0-4), rationale (one concise sentence). 0=unrelated/incorrect, 1=mostly incorrect, "
        "2=partially correct, 3=mostly correct with minor omissions, 4=correct and complete."
    )
    for index, pair in enumerate(selected, 1):
        prompt = (
            f"Task:\n{pair['task_id']}\n\nReference trajectory/final response:\n{pair['reference']}\n\n"
            f"Model response:\n{pair['prediction']}\n\nScore this response."
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            score_value = max(0, min(4, int(parsed["score"])))
            rationale = str(parsed.get("rationale", ""))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            score_value, rationale = None, "Judge returned invalid JSON"
        results.append({"task_id": pair["task_id"], "score": score_value, "rationale": rationale})
        if index % 10 == 0:
            print(f"LLM judge {index}/{len(selected)}", flush=True)
    valid = [row["score"] for row in results if row["score"] is not None]
    return {
        "n": len(results), "judge_model": model, "score_scale": "0-4, higher is better",
        "mean_score": sum(valid) / len(valid) if valid else None,
        "score_counts": dict(Counter(str(value) for value in valid)),
        "per_task": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-limit", type=int)
    args = parser.parse_args()
    pairs = prepare_pairs(args.predictions, args.eval)
    report: dict[str, Any] = {
        "prediction_file": str(args.predictions), "reference_file": str(args.eval),
        "rouge": rouge_scores(pairs),
        "bertscore": bertscore(pairs, args.bertscore_model, args.bertscore_batch_size),
    }
    if args.judge:
        if not args.judge_model:
            parser.error("--judge requires --judge-model")
        report["llm_judge"] = llm_judge(
            pairs, args.judge_model, args.judge_base_url, args.judge_api_key, args.judge_limit
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(pairs), "rouge": report["rouge"]["scores"], "bertscore": report["bertscore"]["scores"]}, indent=2))
    if "llm_judge" in report:
        print(json.dumps({k: v for k, v in report["llm_judge"].items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
