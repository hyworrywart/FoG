"""Freebase access through a local SPARQL endpoint."""

from __future__ import annotations

import os
from typing import Sequence

import requests

NS = "http://rdf.freebase.com/ns/"


class FreebaseStore:
    def __init__(self, endpoint: str | None = None, timeout: int = 300):
        self.endpoint = (endpoint or os.environ.get("FREEBASE_ENDPOINT", "http://localhost:8890/sparql")).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def query(self, sparql: str) -> list[dict]:
        response = self.session.post(
            self.endpoint,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["results"]["bindings"]

    def neighbors(self, entity: str, relations: Sequence[str]) -> list[dict]:
        if not relations:
            return []
        values = " ".join(f"ns:{relation}" for relation in sorted(set(relations)))
        rows = self.query(f"""PREFIX ns: <{NS}>
SELECT DISTINCT ?relation ?connected ?name ?direction WHERE {{
 VALUES ?relation {{ {values} }}
 {{ ns:{entity} ?relation ?connected . BIND(\"forward\" AS ?direction) }}
 UNION
 {{ ?connected ?relation ns:{entity} . BIND(\"backward\" AS ?direction) }}
 OPTIONAL {{ ?connected ns:type.object.name ?name . FILTER(lang(?name) = \"\" || langMatches(lang(?name), \"en\")) }}
}}""")
        result = []
        for row in rows:
            value = row.get("connected", {}).get("value", "")
            entity_id = value.removeprefix(NS)
            if not (entity_id.startswith("m.") or entity_id.startswith("g.")):
                continue
            result.append({
                "relation": row["relation"]["value"].removeprefix(NS),
                "entity": entity_id,
                "name": row.get("name", {}).get("value", entity_id),
                "direction": row["direction"]["value"],
            })
        return result

    def unfiltered_neighbors(self, entity: str) -> list[dict]:
        rows = self.query(f"""PREFIX ns: <{NS}>
SELECT DISTINCT ?relation ?connected ?name ?direction WHERE {{
 {{ ns:{entity} ?relation ?connected . BIND(\"forward\" AS ?direction) }}
 UNION
 {{ ?connected ?relation ns:{entity} . BIND(\"backward\" AS ?direction) }}
 FILTER(STRSTARTS(STR(?relation), \"{NS}\"))
 OPTIONAL {{ ?connected ns:type.object.name ?name . FILTER(lang(?name) = \"\" || langMatches(lang(?name), \"en\")) }}
}}""")
        result = []
        for row in rows:
            entity_id = row.get("connected", {}).get("value", "").removeprefix(NS)
            relation = row.get("relation", {}).get("value", "").removeprefix(NS)
            if relation.startswith(("common.", "freebase.")) or relation in {"type.object.name", "type.object.type"}:
                continue
            result.append({
                "relation": relation,
                "entity": entity_id,
                "name": row.get("name", {}).get("value", entity_id),
                "direction": row["direction"]["value"],
                "expandable": entity_id.startswith(("m.", "g.")),
            })
        return result

    def entity_name(self, entity: str) -> str:
        rows = self.query(f'''PREFIX ns: <{NS}>
SELECT ?name WHERE {{ ns:{entity} ns:type.object.name ?name . FILTER(lang(?name) = "" || langMatches(lang(?name), "en")) }} LIMIT 1''')
        return rows[0]["name"]["value"] if rows else entity
