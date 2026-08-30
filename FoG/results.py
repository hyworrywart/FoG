"""Atomic JSON-array result storage with question-ID resume."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("results file must contain a JSON array")
        return data

    def completed_ids(self):
        return {str(item["question_id"]) for item in self.load() if "question_id" in item}

    def append(self, result):
        values = self.load()
        values = [item for item in values if str(item.get("question_id")) != str(result["question_id"])]
        values.append(result)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(values, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
