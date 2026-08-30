#!/usr/bin/env python3
"""Build the mandatory deterministic two-hop relation cache."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FoG.cache import relation_cache_path
from FoG.datasets import load_dataset
from FoG.knowledge_graph.freebase import FreebaseStore


def collect_two_hop(store, topics):
    frontier, seen, relations = sorted(topics), set(topics), set()
    for _ in range(2):
        following = set()
        for entity in frontier:
            for row in store.unfiltered_neighbors(entity):
                relations.add(row["relation"])
                if row.get("expandable", True) and row["entity"] not in seen:
                    following.add(row["entity"])
        seen.update(following)
        frontier = sorted(following)
    return sorted(relations)


def build_relation_cache(dataset, questions, output_dir, store=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "dataset": dataset,
        "hop": 2,
        "question_count": len(questions),
        "status": "in_progress",
        "completed": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    store = store or FreebaseStore()
    for item in questions:
        relations = collect_two_hop(store, item["topic_entities"])
        relation_cache_path(output, item["id"]).write_text(
            "".join(f"{relation}\n" for relation in relations),
            encoding="utf-8",
        )
        manifest["completed"].append(str(item["id"]))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("webqsp", "cwq", "grailqa"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    build_relation_cache(args.dataset, load_dataset(args.data_file), args.output_dir)


if __name__ == "__main__":
    main()
