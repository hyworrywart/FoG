"""Question-level relation-cache naming and completeness checks."""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_question_id(question_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(question_id)).strip("._")
    if not value:
        raise ValueError("question id cannot be sanitized to an empty filename")
    return value


def relation_cache_path(directory, question_id):
    return Path(directory) / f"{sanitize_question_id(question_id)}.txt"


def require_relation_caches(directory, questions):
    directory = Path(directory)
    missing = [str(item["id"]) for item in questions if not relation_cache_path(directory, item["id"]).is_file()]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing 2-hop relation cache for {len(missing)} selected question(s): {preview}")


def load_relations(directory, question_id):
    path = relation_cache_path(directory, question_id)
    return sorted(set(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))

