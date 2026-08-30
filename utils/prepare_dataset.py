#!/usr/bin/env python3
"""Normalize a benchmark dataset and build its two-hop relation cache."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FoG.datasets import CONVERTERS
from utils.build_relation_cache import build_relation_cache


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONVERTERS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relations-dir", required=True)
    args = parser.parse_args(argv)
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    normalized = CONVERTERS[args.dataset](raw)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_relation_cache(args.dataset, normalized, args.relations_dir)


if __name__ == "__main__":
    main()
