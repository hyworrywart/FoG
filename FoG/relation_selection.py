"""Sentence-transformer relation fuzzy matching with fixed release limits."""

from __future__ import annotations

import re


def relation_phrases(cot: str) -> list[str]:
    parts = re.split(r"\s*(?:->|<-)\s*", cot)
    return [parts[index].strip(' "') for index in range(1, len(parts), 2) if parts[index].strip()]


class RelationSelector:
    def __init__(self, model_id: str, encoder=None):
        self.model_id = model_id
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_id)
        return self._encoder

    def select(self, cot: str, relations: list[str], top_k_cot: int = 20, top_k_per_relation: int = 10) -> list[str]:
        import numpy as np

        candidates = sorted(set(relations))
        if not candidates:
            return []
        queries = [cot, *relation_phrases(cot)]
        embeddings = self.encoder.encode([*queries, *candidates], normalize_embeddings=True)
        query_vectors = embeddings[:len(queries)]
        relation_vectors = embeddings[len(queries):]
        selected = []
        limits = [top_k_cot, *([top_k_per_relation] * (len(queries) - 1))]
        for vector, limit in zip(query_vectors, limits):
            scores = np.asarray(relation_vectors) @ np.asarray(vector)
            order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), candidates[index]))[:limit]
            for index in order:
                if candidates[index] not in selected:
                    selected.append(candidates[index])
        return selected

