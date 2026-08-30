"""Minimal OpenAI-compatible chat client with per-kind accounting."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterable


class OpenAIClient:
    def __init__(self, text_model: str, vision_model: str | None = None):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = OpenAI(
            api_key=key,
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        self.text_model = text_model
        self.vision_model = vision_model
        self.calls: list[dict] = []
        self.token_stats = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete_text(self, prompt: str, *, kind: str, temperature: float = 0.3) -> str:
        return self._complete(self.text_model, prompt, [], kind, temperature)

    def complete_vision(
        self, prompt: str, images: Iterable[Path], *, kind: str, temperature: float = 0.3
    ) -> str:
        if not self.vision_model:
            raise RuntimeError("A vision model is required for the vision pipeline")
        return self._complete(self.vision_model, prompt, images, kind, temperature)

    def _complete(self, model, prompt, images, kind, temperature):
        content = []
        image_count = 0
        for path in images:
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": "high"}})
            image_count += 1
        content.append({"type": "text", "text": prompt})
        self.calls.append({"kind": kind, "model": model, "image_count": image_count})
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=1024,
        )
        usage = getattr(response, "usage", None)
        if usage:
            self.token_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self.token_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0)
        return response.choices[0].message.content or ""
