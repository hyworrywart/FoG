"""Shared graph-reasoning algorithm for FoG-V and FoG-T."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..cache import sanitize_question_id
from ..knowledge_graph.graph import build_contracted_graph, explore_filtered_graph
from ..knowledge_graph.paths import PathVerifier, extract_paths, first_thinking_cot
from ..prompts import (
    answer_with_vision_paths_prompt,
    text_path_select_retry_prompt,
    vision_cot_prompt,
    vision_explore_prompt,
    vision_path_select_complementary_set_prompt,
    vision_path_select_retry_prompt,
    vision_path_select_text_only_prompt,
    vision_single_path_correction_prompt,
    vision_summary_cot_prompt,
)


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: str
    dataset: str | None = None
    text_model: str = "gpt-3.5-turbo-0125"
    vision_model: str | None = "gpt-4o-2024-11-20"
    sentence_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b"
    max_hops: int = 2
    top_k_cot: int = 20
    top_k_per_relation: int = 10
    max_iterations: int = 3
    corrections_per_path: int = 2

    def __post_init__(self):
        if self.pipeline not in {"vision", "text"}:
            raise ValueError("pipeline must be vision or text")
        if self.dataset is not None and self.dataset not in {"webqsp", "cwq", "grailqa"}:
            raise ValueError("dataset must be webqsp, cwq, or grailqa")


def format_topics(topics: dict[str, str]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in topics.items()) + "}"


def stable_unique(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def parse_sufficiency(response: str) -> bool | None:
    match = re.search(r"(?im)^\s*response\s*:\s*\{?\s*(yes|no)\s*\}?", response or "")
    return match and match.group(1).casefold() == "yes"


def parse_prediction(response: str) -> list[str]:
    match = re.search(r"(?im)^\s*answer\s*:\s*\{([^}]*)\}", response or "")
    if not match:
        return []
    return stable_unique(part.strip() for part in match.group(1).split("|") if part.strip())


class FoGPipeline:
    def __init__(self, config: PipelineConfig, llm, store, relation_selector, output_dir: str | Path):
        self.config = config
        self.llm = llm
        self.store = store
        self.relation_selector = relation_selector
        self.output_dir = Path(output_dir)
        self.verifier = PathVerifier(store)

    def _text(self, prompt, kind):
        return self.llm.complete_text(prompt, kind=kind, temperature=0.3)

    def _select(self, prompt, kind, image_views):
        if self.config.pipeline == "vision":
            return self.llm.complete_vision(prompt, image_views, kind=kind, temperature=0.3)
        return self._text(prompt, kind)

    def _verify_set(self, candidates, topics, names, text_description):
        valid, failed = [], []
        for candidate in candidates:
            current = candidate
            retrieved = None
            for attempt in range(self.config.corrections_per_path + 1):
                try:
                    retrieved = self.verifier.verify(current, topics, names)
                except (ValueError, KeyError):
                    retrieved = None
                if retrieved:
                    break
                if attempt < self.config.corrections_per_path:
                    response = self._text(
                        vision_single_path_correction_prompt.format(
                            failed_path=current, text_description=text_description
                        ),
                        "path_correction",
                    )
                    corrected = extract_paths("Path 1: " + response, 1)
                    if not corrected:
                        break
                    current = corrected[0]
            if retrieved:
                valid.append(retrieved)
            else:
                failed.append(candidate)
        return valid, failed

    def run(self, item: dict, candidate_relations: list[str]) -> dict:
        started = time.monotonic()
        question_id = str(item["id"])
        question = " ".join(str(item["question"]).split())
        topics = dict(item["topic_entities"])
        cot_response = self._text(
            vision_cot_prompt.format(question=question, topic_entities=format_topics(topics)),
            "rpp",
        )
        thinking_cot = first_thinking_cot(cot_response)
        selected_relations = self.relation_selector.select(
            thinking_cot,
            candidate_relations,
            top_k_cot=self.config.top_k_cot,
            top_k_per_relation=self.config.top_k_per_relation,
        )
        # The relation set is frozen here and never augmented by exploration.
        selected_relations = tuple(selected_relations)
        centers = topics.copy()
        all_paths: list[str] = []
        final_response = ""
        records = []
        question_dir = self.output_dir / sanitize_question_id(question_id)

        for iteration in range(1, self.config.max_iterations + 1):
            graph, names = explore_filtered_graph(
                self.store, list(centers), list(selected_relations), self.config.max_hops
            )
            names.update(topics)
            names.update(centers)
            contracted = build_contracted_graph(
                graph,
                selected_relations,
                topics,
                names,
                centers,
                self.config.max_hops,
            )
            image_views = []
            image_path = None
            if self.config.pipeline == "vision":
                image_path, image_views = contracted.render(question_dir / f"iteration_{iteration}")
                selector_template = vision_path_select_complementary_set_prompt
            else:
                selector_template = vision_path_select_text_only_prompt
            prompt = selector_template.format(
                question=question,
                topic_entities=format_topics(topics),
                text_description=contracted.text_description,
                thinking_cot=thinking_cot,
            )
            selection = self._select(prompt, "path_selection", image_views)
            candidates = extract_paths(selection)
            anchors = topics | centers
            verified, failed = self._verify_set(candidates, anchors, names, contracted.text_description)
            retry_candidates = []
            if failed:
                retry_template = vision_path_select_retry_prompt if self.config.pipeline == "vision" else text_path_select_retry_prompt
                retry_prompt = retry_template.format(
                    failed_paths="\n".join(failed),
                    question=question,
                    topic_entities=format_topics(topics),
                    text_description=contracted.text_description,
                    thinking_cot=thinking_cot,
                )
                retry_response = self._select(retry_prompt, "path_selection_retry", image_views)
                retry_candidates = [path for path in extract_paths(retry_response) if path not in failed]
                retry_verified, _ = self._verify_set(retry_candidates, anchors, names, contracted.text_description)
                verified.extend(retry_verified)
            all_paths = stable_unique([*all_paths, *verified])
            paths_text = "\n".join(f"Path {index}: {path}" for index, path in enumerate(all_paths, 1))
            summary = self._text(
                vision_summary_cot_prompt.format(
                    question=question,
                    topic_entities=format_topics(topics),
                    paths=paths_text,
                ),
                "summary",
            )
            final_response = self._text(
                answer_with_vision_paths_prompt.format(
                    question=question,
                    thinking_cot=thinking_cot,
                    paths=paths_text,
                    summarized_cot=summary,
                ),
                "sufficiency_answer",
            )
            sufficient = parse_sufficiency(final_response)
            records.append({
                "iteration": iteration,
                "centers": centers,
                "selected_paths": candidates,
                "retry_paths": retry_candidates,
                "verified_paths": list(verified),
                "graph_image": str(image_path) if image_path else None,
                "text_description": contracted.text_description,
                "sufficient": sufficient,
            })
            if sufficient is True or iteration == self.config.max_iterations:
                break
            candidate_ids = stable_unique(re.findall(r"\b(?:m|g)\.[A-Za-z0-9_]+", "\n".join(all_paths)))
            candidate_ids = [entity for entity in candidate_ids if entity not in topics]
            if not candidate_ids:
                centers = topics.copy()
                continue
            explore_response = self._text(
                vision_explore_prompt.format(
                    question=question,
                    current_paths=paths_text,
                    summarized_paths=summary,
                    summary=final_response,
                ),
                "exploration",
            )
            requested = stable_unique(re.findall(r"\b(?:m|g)\.[A-Za-z0-9_]+", explore_response))
            chosen = [entity for entity in requested if entity in candidate_ids][:3]
            if not chosen:
                centers = topics.copy()
            else:
                centers = {entity: names.get(entity, entity) for entity in chosen}

        calls = list(getattr(self.llm, "calls", []))
        final_sufficient = parse_sufficiency(final_response)
        return {
            "schema_version": "1.0",
            "question_id": question_id,
            "question": question,
            "topic_entities": topics,
            "thinking_cot": thinking_cot,
            "retrieved_paths": all_paths,
            "LLM_answer": final_response,
            "prediction": parse_prediction(final_response) if final_sufficient is True else [],
            "iterations": len(records),
            "LLM_calls": len(calls),
            "token_stats": dict(getattr(self.llm, "token_stats", {})),
            "elapsed_time": time.monotonic() - started,
            "run_config": asdict(self.config),
            "iteration_records": records,
        }
