#!/usr/bin/env python3
"""Extract final-answer spans from ToolBench Finish calls and score normalized EM."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


def extract_final_answer(text: str) -> str | None:
    """Extract a ToolBench Finish Action Input final_answer field if present."""
    marker = re.search(r"Action\s*:\s*Finish\b", text, flags=re.IGNORECASE)
    if not marker:
        return None
    tail = text[marker.end() :]
    # Find the JSON payload, allowing harmless whitespace after Action Input.
    action_input = re.search(r"Action\s*Input\s*:\s*", tail, flags=re.IGNORECASE)
    if action_input:
        tail = tail[action_input.end() :]
    start = tail.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(tail[start:])
    except json.JSONDecodeError:
        return None
    value = obj.get("final_answer") if isinstance(obj, dict) else None
    return value if isinstance(value, str) else None


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_files(prediction_path: Path, eval_path: Path) -> dict[str, Any]:
    predictions = {
        str(row["task_id"]): row
        for row in (json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines())
        if row.get("task_id")
    }
    refs = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for row in refs:
        task_id = str(row["task_id"])
        candidate = predictions.get(task_id)
        if candidate is None:
            continue
        pred_final = extract_final_answer(candidate.get("prediction", ""))
        ref_final = extract_final_answer(row["messages"][-1].get("content", ""))
        # Subspan EM is defined only where the reference has a Finish/final_answer.
        # A generated non-Finish action is counted as not correct (not dropped).
        exact = bool(pred_final is not None and ref_final is not None and normalize(pred_final) == normalize(ref_final))
        rows.append({
            "task_id": task_id,
            "prediction_finish": pred_final is not None,
            "reference_finish": ref_final is not None,
            "subspan_em": float(exact),
            "prediction_final_answer": pred_final,
            "reference_final_answer": ref_final,
        })
    applicable = [r for r in rows if r["reference_finish"]]
    hits = sum(r["subspan_em"] for r in applicable)
    return {
        "metric": "ToolBench final_answer normalized exact match",
        "normalization": "lowercase, remove ASCII punctuation, collapse whitespace",
        "n_predictions_matched": len(rows),
        "n_reference_finish": len(applicable),
        "n_prediction_finish": sum(r["prediction_finish"] for r in applicable),
        "subspan_em_correct": int(hits),
        "subspan_em": hits / len(applicable) if applicable else None,
        "per_task": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_files(args.predictions, args.eval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
