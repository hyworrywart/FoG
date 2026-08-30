"""Evaluate structured predictions using set Hit and set F1."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from .datasets import load_dataset


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def score_sets(prediction, gold):
    predicted = {normalize_answer(str(value)) for value in prediction if normalize_answer(str(value))}
    expected = {normalize_answer(answer.get("name") or answer.get("id", "")) for answer in gold}
    expected.discard("")
    overlap = predicted & expected
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"hit": bool(overlap), "precision": precision, "recall": recall, "f1": f1}


def evaluate(results, dataset):
    by_id = {str(item["id"]): item for item in dataset}
    scores = [score_sets(item.get("prediction", []), by_id[str(item["question_id"])]["answers"]) for item in results if str(item.get("question_id")) in by_id]
    count = len(scores)
    return {"count": count, "hit": sum(score["hit"] for score in scores) / count if count else 0.0, "f1": sum(score["f1"] for score in scores) / count if count else 0.0}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--results-file", required=True)
    args = parser.parse_args(argv)
    metrics = evaluate(json.loads(Path(args.results_file).read_text()), load_dataset(args.data_file))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
