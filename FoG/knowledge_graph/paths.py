"""Path parsing and original Freebase SPARQL verification."""

from __future__ import annotations

import re

NS = "http://rdf.freebase.com/ns/"


def parse_relation_path(path: str) -> tuple[list[str], list[str]]:
    parts = [part.strip() for part in re.split(r"\s*(?:->|<-)\s*", path) if part.strip()]
    arrows = re.findall(r"(->|<-)", path)
    if len(parts) < 3 or len(parts) % 2 == 0 or len(arrows) != len(parts) - 1:
        raise ValueError(f"Invalid relation path: {path}")
    directions = []
    for index in range(len(parts) // 2):
        pair = arrows[index * 2:index * 2 + 2]
        if pair[0] != pair[1]:
            raise ValueError(f"Inconsistent arrows in relation path: {path}")
        directions.append(pair[0])
    return parts, directions


def extract_paths(response: str, limit: int = 3) -> list[str]:
    result = []
    for line in (response or "").splitlines():
        match = re.match(r"\s*(?:[-*]\s*)?Path\s*\d+\s*:\s*(.+)", line, re.I)
        if not match:
            continue
        candidate = match.group(1).replace("**", "").replace("`", "").strip()
        try:
            parse_relation_path(candidate)
        except ValueError:
            continue
        if candidate not in result:
            result.append(candidate)
        if len(result) == limit:
            break
    return result


def first_thinking_cot(response: str) -> str:
    """Keep only the first arrow chain; malformed output is never regenerated."""
    for line in (response or "").splitlines():
        if "->" in line or "<-" in line:
            line = re.sub(r"^\s*(?:Thinking\s+CoT\s*:)?\s*", "", line, flags=re.I)
            return line.strip()
    return ""


class PathVerifier:
    def __init__(self, store, query_limit: int = 100):
        self.store = store
        self.query_limit = query_limit

    def verify(self, path: str, topics: dict[str, str], names: dict[str, str] | None = None) -> str | None:
        parts, directions = parse_relation_path(path)
        names = names or {}
        topic_positions: dict[int, str] = {}
        variable_positions = []
        for position in range(0, len(parts), 2):
            token = parts[position].strip().strip('"').casefold()
            found = next((eid for eid, name in topics.items() if token in {eid.casefold(), name.casefold()} or token in name.casefold() or name.casefold() in token), None)
            if found:
                topic_positions[position] = found
            else:
                variable_positions.append(position)
        if not topic_positions:
            return None
        variables = {position: f"?n{index + 1}" for index, position in enumerate(variable_positions)}

        def term(position):
            return f"ns:{topic_positions[position]}" if position in topic_positions else variables[position]

        triples = []
        for index, arrow in enumerate(directions):
            relation = parts[index * 2 + 1].strip("`*_")
            left, right = term(index * 2), term(index * 2 + 2)
            triples.append(f"{left} ns:{relation} {right}" if arrow == "->" else f"{right} ns:{relation} {left}")
        select = " ".join(variables[position] for position in sorted(variable_positions)) or "*"
        triple_pattern = " .\n  ".join(triples)
        query = f'''PREFIX ns: <{NS}>
SELECT DISTINCT {select} WHERE {{
  {triple_pattern} .
}} LIMIT {self.query_limit}'''
        rows = self.store.query(query)
        if not rows:
            return None
        values: dict[int, list[str]] = {position: [] for position in variable_positions}
        for row in rows:
            for position in variable_positions:
                raw = row.get(variables[position][1:], {}).get("value", "")
                value = raw.removeprefix(NS)
                if value and value not in values[position]:
                    values[position].append(value)
        segments = []
        for index in range(len(directions) + 1):
            position = index * 2
            if position in topic_positions:
                eid = topic_positions[position]
                segments.append(f"{{{eid}: {topics[eid]}}}")
            else:
                formatted = [f"{value}: {names.get(value, value)}" for value in values[position]]
                segments.append("{" + ", ".join(formatted) + "}")
            if index < len(directions):
                relation = parts[index * 2 + 1].strip("`*_")
                arrow = directions[index]
                segments.append("{" + f" {arrow} {relation} {arrow} " + "}")
        return " - ".join(segments)
