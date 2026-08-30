"""Filtered graph construction and deterministic union-find contraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

Edge = tuple[str, str, str]


def explore_filtered_graph(store, centers: list[str], relations: list[str], max_hops: int = 2):
    """Build a directed graph using only the relation set fixed before iteration."""
    graph: dict[str, dict[str, dict[str, set[str]]]] = {entity: {} for entity in centers}
    names: dict[str, str] = {}
    seen = set(centers)
    frontier = sorted(set(centers))
    for _ in range(max_hops):
        following = set()
        for entity in frontier:
            for row in store.neighbors(entity, relations):
                neighbor, relation, direction = row["entity"], row["relation"], row["direction"]
                names[neighbor] = row.get("name", neighbor)
                graph.setdefault(entity, {}).setdefault(neighbor, {"forward": set(), "backward": set()})[direction].add(relation)
                reverse = "backward" if direction == "forward" else "forward"
                graph.setdefault(neighbor, {}).setdefault(entity, {"forward": set(), "backward": set()})[reverse].add(relation)
                if neighbor not in seen:
                    following.add(neighbor)
        seen.update(following)
        frontier = sorted(following)
        if not frontier:
            break
    return graph, names


@dataclass
class ContractedGraph:
    graph: nx.MultiDiGraph
    entity_to_node: dict[str, str]
    number_to_entities: dict[int, set[str]]
    text_description: str

    def render(self, output_stem: Path) -> tuple[Path, list[Path]]:
        from graphviz import Digraph

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        dot = Digraph("knowledge_graph", engine="dot")
        dot.attr(rankdir="TB", dpi="144", splines="spline", ranksep="0.8", nodesep="0.6")
        dot.attr("node", fontname="Arial", fontsize="12")
        dot.attr("edge", fontname="Arial", fontsize="10")
        for node, data in sorted(self.graph.nodes(data=True)):
            if data["type"] == "topic_entity":
                dot.node(node, data["name"], shape="box", style="filled", fillcolor="#FFD700", penwidth="2")
            elif data["type"] == "center_entity":
                dot.node(node, data["name"], shape="box", style="filled", fillcolor="#90EE90", penwidth="2")
            else:
                dot.node(node, str(data["number"]), shape="circle", style="filled", fillcolor="#87CEEB", fixedsize="true", width="0.35")
        for src, dst, data in sorted(self.graph.edges(data=True), key=lambda item: (item[0], item[1], item[2]["relation"], item[2]["direction"])):
            relation = data["relation"]
            dot.edge(src, dst, label=relation.rsplit(".", 1)[-1], dir="forward" if data["direction"] == "forward" else "back")
        rendered = Path(dot.render(str(output_stem), format="png", cleanup=True))
        text_path = output_stem.with_suffix(".txt")
        text_path.write_text(self.text_description, encoding="utf-8")
        return rendered, make_image_views(rendered)


def _collect_edges(graph, centers, selected, max_hops):
    queue = deque((entity, 0) for entity in sorted(centers))
    visited = {entity: 0 for entity in centers}
    edges = set()
    while queue:
        entity, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in sorted(graph.get(entity, {})):
            data = graph[entity][neighbor]
            valid = False
            for direction in ("forward", "backward"):
                for relation in sorted(data.get(direction, set()) & selected):
                    edges.add((entity, neighbor, relation, direction))
                    valid = True
            if valid and neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    return sorted(edges)


def build_contracted_graph(graph, relations, topics, names, centers=None, max_hops=2):
    """Contract entities sharing identical incident relation patterns."""
    centers = centers or topics
    special = {entity: "topic_entity" for entity in topics}
    special.update({entity: "center_entity" for entity in centers if entity not in special})
    edges = _collect_edges(graph, centers, set(relations), max_hops)
    parent: dict[str, str] = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
            return True
        return False

    changed = True
    while changed:
        changed = False
        by_target, by_source = {}, {}
        for src, dst, relation, direction in edges:
            src_root = src if src in special else find(src)
            dst_root = dst if dst in special else find(dst)
            if src not in special:
                key = (dst_root, relation, direction)
                changed |= union(src, by_target.setdefault(key, src))
            if dst not in special:
                key = (src_root, relation, direction)
                changed |= union(dst, by_source.setdefault(key, dst))

    groups = sorted({find(entity) for edge in edges for entity in edge[:2] if entity not in special})
    group_nodes = {group: f"n_{index}" for index, group in enumerate(groups, 1)}
    entity_to_node = {}
    nx_graph = nx.MultiDiGraph()
    for entity, kind in sorted(special.items()):
        node = ("te_" if kind == "topic_entity" else "ce_") + entity
        entity_to_node[entity] = node
        nx_graph.add_node(node, type=kind, name=(topics | centers).get(entity, names.get(entity, entity)), entities={entity})
    for edge in edges:
        for entity in edge[:2]:
            if entity not in entity_to_node:
                node = group_nodes[find(entity)]
                entity_to_node[entity] = node
                if node not in nx_graph:
                    nx_graph.add_node(node, type="intermediate", entities=set())
                nx_graph.nodes[node]["entities"].add(entity)
    added = set()
    for src, dst, relation, direction in edges:
        left, right = entity_to_node[src], entity_to_node[dst]
        if left == right:
            continue
        canonical = (left, right, relation, direction)
        reverse = (right, left, relation, "backward" if direction == "forward" else "forward")
        if canonical not in added and reverse not in added:
            nx_graph.add_edge(left, right, relation=relation, direction=direction)
            added.add(canonical)
    intermediate = sorted(node for node, data in nx_graph.nodes(data=True) if data["type"] == "intermediate")
    number_to_entities = {}
    for number, node in enumerate(intermediate, 1):
        nx_graph.nodes[node]["number"] = number
        number_to_entities[number] = set(nx_graph.nodes[node]["entities"])

    def label(node):
        data = nx_graph.nodes[node]
        return data["name"] if data["type"] != "intermediate" else f"Node {data['number']}"

    descriptions = []
    for src, dst, data in sorted(nx_graph.edges(data=True), key=lambda item: (label(item[0]), label(item[1]), item[2]["relation"], item[2]["direction"])):
        if data["direction"] == "forward":
            descriptions.append(f"{label(src)}  ->  {data['relation']}  ->  {label(dst)}")
        else:
            descriptions.append(f"{label(dst)}  ->  {data['relation']}  ->  {label(src)}")
    return ContractedGraph(nx_graph, entity_to_node, number_to_entities, "\n".join(descriptions))


def make_image_views(path: Path, wide_ratio: float = 1.8, max_crops: int = 4) -> list[Path]:
    """Return one overview plus overlapping horizontal crops for a wide graph."""
    from PIL import Image

    image = Image.open(path)
    if image.width <= image.height * wide_ratio:
        return [path]
    overview = path.with_name(path.stem + "_overview.png")
    copy = image.copy()
    copy.thumbnail((1600, 1000))
    copy.save(overview)
    crop_width = min(image.width, max(image.height * 2, 1200))
    count = min(max_crops, max(2, (image.width + crop_width - 1) // crop_width))
    starts = [round(index * (image.width - crop_width) / (count - 1)) for index in range(count)]
    crops = []
    for index, start in enumerate(starts, 1):
        crop_path = path.with_name(f"{path.stem}_crop{index}.png")
        image.crop((start, 0, start + crop_width, image.height)).save(crop_path)
        crops.append(crop_path)
    return [overview, *crops]

