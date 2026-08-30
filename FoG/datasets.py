"""Normalized release dataset loading and raw dataset conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED = ("id", "question", "topic_entities", "answers")


def validate_question(item: dict) -> dict:
    missing = [key for key in REQUIRED if key not in item]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    if not str(item["id"]).strip() or not str(item["question"]).strip():
        raise ValueError("id and question must be non-empty")
    if not isinstance(item["topic_entities"], dict) or not item["topic_entities"]:
        raise ValueError("topic_entities must be a non-empty object")
    if not isinstance(item["answers"], list) or not item["answers"]:
        raise ValueError("answers must be a non-empty array")
    for answer in item["answers"]:
        if not isinstance(answer, dict) or not (answer.get("id") or answer.get("name")):
            raise ValueError("each answer requires an id or name")
    return item


def load_dataset(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("normalized data must be a JSON array")
    return [validate_question(item) for item in data]


def convert_webqsp(raw) -> list[dict]:
    questions = raw.get("Questions", raw) if isinstance(raw, dict) else raw
    result = []
    for item in questions:
        parses = item.get("Parses", [])
        topics, answers = {}, []
        for parse in parses:
            mid = parse.get("TopicEntityMid")
            if mid:
                topics[mid] = parse.get("TopicEntityName") or mid
            for answer in parse.get("Answers", []):
                value = {"id": answer.get("AnswerArgument", ""), "name": answer.get("EntityName") or answer.get("AnswerArgument", "")}
                if value not in answers:
                    answers.append(value)
        result.append(validate_question({"id": str(item.get("QuestionId", "")), "question": item.get("ProcessedQuestion") or item.get("RawQuestion", ""), "topic_entities": topics, "answers": answers}))
    return result


def convert_cwq(raw) -> list[dict]:
    result = []
    for item in raw:
        topics = item.get("topic_entities") or item.get("topic_entity") or item.get("entities") or {}
        if isinstance(topics, list):
            topics = {entry.get("id", entry.get("mid")): entry.get("name", entry.get("id", entry.get("mid"))) for entry in topics}
        if not topics:
            mids = sorted(set(re.findall(r"\bns:((?:m|g)\.[A-Za-z0-9_]+)", item.get("sparql", ""))))
            topics = {mid: mid for mid in mids}
        raw_answers = item.get("answers", item.get("answer", []))
        if not isinstance(raw_answers, list):
            raw_answers = [raw_answers]
        answers = []
        for value in raw_answers:
            if isinstance(value, dict):
                answers.append({
                    "id": value.get("id") or value.get("answer_id") or value.get("answer_argument", ""),
                    "name": value.get("name") or value.get("answer") or value.get("entity_name", ""),
                })
            else:
                answers.append({"id": "", "name": str(value)})
        result.append(validate_question({"id": str(item.get("id", item.get("ID", ""))), "question": item.get("question", ""), "topic_entities": topics, "answers": answers}))
    return result


def convert_grailqa(raw) -> list[dict]:
    result = []
    for item in raw:
        topics = item.get("topic_entities") or item.get("topic_entity") or {}
        if not topics:
            graph_query = item.get("graph_query", {})
            topics = {node["id"]: node.get("friendly_name") or node["id"] for node in graph_query.get("nodes", []) if node.get("node_type") == "entity"}
        answers = [{"id": answer.get("answer_argument", ""), "name": answer.get("entity_name") or answer.get("answer_argument", "")} for answer in item.get("answer", [])]
        result.append(validate_question({"id": str(item.get("qid", item.get("id", ""))), "question": item.get("question", ""), "topic_entities": topics, "answers": answers}))
    return result


CONVERTERS = {"webqsp": convert_webqsp, "cwq": convert_cwq, "grailqa": convert_grailqa}
