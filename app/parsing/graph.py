from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import Diagram, Label, Node

Point = Tuple[float, float]


@dataclass
class Edge:
    source: str
    target: str
    condition: Optional[str] = None


@dataclass
class Step:
    id: str
    type: str
    text: str
    actor: Optional[str]
    next: List[Tuple[str, Optional[str]]]


@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]
    outgoing: Dict[str, List[Edge]]
    indegree: Dict[str, int]
    node_order: List[str]
    centers: Dict[str, Point]
    radius: Dict[str, float]

MISSING_CONDITION_TEXT = "\u0443\u0441\u043b\u043e\u0432\u0438\u0435 \u043d\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e"

def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _distance_to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return _distance(p, a)
    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + abx * t, ay + aby * t)
    return _distance(p, closest)


def _nearest_node(point: Point, centers: Dict[str, Point], radius: Dict[str, float]) -> Optional[str]:
    nearest_id: Optional[str] = None
    nearest_dist = float("inf")
    for node_id, center in centers.items():
        dist = _distance(point, center)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id = node_id
    if nearest_id is None:
        return None
    if nearest_dist <= radius.get(nearest_id, 0.0):
        return nearest_id
    return None


def _assign_gateway_conditions(
    outgoing_edges: List[Edge],
    labels: Iterable[Label],
    source_center: Point,
    centers: Dict[str, Point],
    max_distance: float,
) -> None:
    cleaned_labels: List[Tuple[int, str, Point]] = []
    for idx, label in enumerate(labels):
        text = _clean_text(label.txt)
        if not text:
            continue
        cleaned_labels.append((idx, text, label.cnt))

    edge_candidates: List[Tuple[Edge, List[Tuple[float, int, str]]]] = []
    for edge in outgoing_edges:
        target_center = centers[edge.target]
        candidates: List[Tuple[float, int, str]] = []
        for idx, text, cnt in cleaned_labels:
            dist = _distance_to_segment(cnt, source_center, target_center)
            if dist <= max_distance:
                candidates.append((dist, idx, text))
        candidates.sort(key=lambda item: item[0])
        edge_candidates.append((edge, candidates))

    edge_candidates.sort(
        key=lambda item: item[1][0][0] if item[1] else float("inf")
    )

    used_labels = set()
    for edge, candidates in edge_candidates:
        for _, idx, text in candidates:
            if idx in used_labels:
                continue
            edge.condition = text
            used_labels.add(idx)
            break


def build_graph(diagram: Diagram, label_distance: float = 40.0, radius_pad: float = 20.0) -> Graph:
    nodes = {node.id: node for node in diagram.nodes}
    node_order = [node.id for node in diagram.nodes]
    centers: Dict[str, Point] = {}
    radius: Dict[str, float] = {}
    for node in diagram.nodes:
        centers[node.id] = (float(node.cnt[0]), float(node.cnt[1]))
        radius[node.id] = max(float(node.wh[0]), float(node.wh[1])) / 2.0 + radius_pad

    edges: List[Edge] = []
    seen_edges = set()
    for arrow in diagram.arrows:
        target = _nearest_node(arrow.tip, centers, radius)
        if target is None:
            continue
        for start in arrow.starts:
            if _distance(start, arrow.tip) < 2.0:
                continue
            source = _nearest_node(start, centers, radius)
            if source is None or source == target:
                continue
            key = (source, target)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(Edge(source=source, target=target))
    outgoing: Dict[str, List[Edge]] = {node_id: [] for node_id in nodes}
    indegree: Dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in edges:
        outgoing[edge.source].append(edge)
        indegree[edge.target] += 1

    for node_id, node in nodes.items():
        if node.type.lower() != "gateway":
            continue
        gateway_edges = outgoing.get(node_id, [])
        if len(gateway_edges) <= 1:
            continue
        _assign_gateway_conditions(
            gateway_edges,
            diagram.labels,
            centers[node_id],
            centers,
            label_distance,
        )

    return Graph(
        nodes=nodes,
        edges=edges,
        outgoing=outgoing,
        indegree=indegree,
        node_order=node_order,
        centers=centers,
        radius=radius,
    )


def _flow_axis(graph: Graph) -> str:
    if graph.edges:
        score = 0
        for edge in graph.edges:
            sx, sy = graph.centers[edge.source]
            tx, ty = graph.centers[edge.target]
            if abs(tx - sx) >= abs(ty - sy):
                score += 1
            else:
                score -= 1
        return "x" if score >= 0 else "y"

    xs = [graph.centers[n][0] for n in graph.nodes]
    ys = [graph.centers[n][1] for n in graph.nodes]
    if not xs or not ys:
        return "x"
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return "x" if width >= height else "y"


def _node_sort_key(graph: Graph, node_id: str, axis: str) -> Tuple[float, float]:
    x, y = graph.centers[node_id]
    return (x, y) if axis == "x" else (y, x)


def build_steps(graph: Graph) -> List[Step]:
    axis = _flow_axis(graph)
    start_nodes = [
        node_id
        for node_id in graph.node_order
        if graph.indegree.get(node_id, 0) == 0
        or graph.nodes[node_id].type.lower() == "startevent"
    ]
    start_nodes.sort(
        key=lambda node_id: (
            0 if graph.nodes[node_id].type.lower() == "startevent" else 1,
            *_node_sort_key(graph, node_id, axis),
        )
    )

    indegree = dict(graph.indegree)
    order: List[str] = []
    ready = list(start_nodes)
    seen_ready = set(ready)

    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        outgoing = sorted(
            graph.outgoing.get(node_id, []),
            key=lambda edge: _node_sort_key(graph, edge.target, axis),
        )
        for edge in outgoing:
            indegree[edge.target] = indegree.get(edge.target, 0) - 1
            if indegree[edge.target] == 0 and edge.target not in seen_ready:
                ready.append(edge.target)
                seen_ready.add(edge.target)
                ready.sort(key=lambda nid: _node_sort_key(graph, nid, axis))

    for node_id in graph.node_order:
        if node_id not in order:
            order.append(node_id)

    steps: List[Step] = []
    for node_id in order:
        node = graph.nodes[node_id]
        text = _clean_text(node.txt)
        outgoing_sorted = sorted(
            graph.outgoing.get(node_id, []),
            key=lambda edge: _node_sort_key(graph, edge.target, axis),
        )
        next_list = [(edge.target, edge.condition) for edge in outgoing_sorted]
        steps.append(
            Step(
                id=node_id,
                type=node.type,
                text=text,
                actor=_clean_text(node.actor) if node.actor else None,
                next=next_list,
            )
        )

    return steps


def _action_text(step: Optional[Step], start_emitted: bool) -> Tuple[str, bool]:
    actor = None
    if step and step.actor:
        actor = step.actor
    actor_part = f" (\u0438\u0441\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c: {actor})" if actor else ""

    if step is None:
        return "\u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435", start_emitted

    step_type = step.type.lower()
    if step_type == "startevent":
        if start_emitted:
            return f"\u0421\u043e\u0431\u044b\u0442\u0438\u0435 \u0441\u0442\u0430\u0440\u0442\u0430{actor_part}", start_emitted
        return f"\u041d\u0430\u0447\u0430\u043b\u043e \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430{actor_part}", True
    if step_type == "endevent":
        return f"\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0438\u0435 \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430{actor_part}", start_emitted
    if step_type == "gateway":
        if step.text:
            return f"\u0423\u0441\u043b\u043e\u0432\u0438\u0435: {step.text}{actor_part}", start_emitted
        return f"\u0423\u0441\u043b\u043e\u0432\u0438\u0435{actor_part}", start_emitted

    base = step.text or "\u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435 \u0431\u0435\u0437 \u0442\u0435\u043a\u0441\u0442\u0430"
    return f"{base}{actor_part}", start_emitted


def format_description(graph: Graph, steps: List[Step]) -> str:
    step_map = {step.id: step for step in steps}
    emitted = set()
    visited_gateway = set()
    lines: List[str] = []
    start_emitted = False

    def emit_linear(start_id: str, prefix: str) -> None:
        current = start_id
        idx = 1
        while current:
            if current in emitted:
                break
            step = step_map.get(current)
            if step is None or step.type.lower() == "gateway":
                break
            line_prefix = f"{prefix}.{idx}"
            text, _ = _action_text(step, True)
            lines.append(f"{line_prefix} {text}.")
            emitted.add(current)
            if len(step.next) != 1:
                break
            next_id = step.next[0][0]
            if next_id in emitted:
                break
            current = next_id
            idx += 1

    def emit_gateway(node_id: str, prefix: str) -> None:
        nonlocal start_emitted
        if node_id in visited_gateway:
            return
        visited_gateway.add(node_id)
        step = step_map.get(node_id)
        if step is None:
            return
        for idx, (target, condition) in enumerate(step.next, start=1):
            target_step = step_map.get(target)
            target_text, start_emitted = _action_text(target_step, start_emitted)
            branch_prefix = f"{prefix}.{idx}"
            if condition:
                lines.append(f"{branch_prefix} \u0415\u0441\u043b\u0438: {condition} \u2192 {target_text}.")
            else:
                lines.append(f"{branch_prefix} {target_text}.")
            emitted.add(target)
            if target_step and target_step.type.lower() == "gateway" and target_step.next:
                emit_gateway(target, branch_prefix)
            elif target_step and len(target_step.next) == 1:
                emit_linear(target_step.next[0][0], branch_prefix)

    counter = 1
    for step in steps:
        if step.id in emitted:
            continue
        if step.type.lower() == "gateway" and step.next:
            text, start_emitted = _action_text(step, start_emitted)
            lines.append(f"{counter}. {text}.")
            emitted.add(step.id)
            emit_gateway(step.id, str(counter))
        else:
            text, start_emitted = _action_text(step, start_emitted)
            lines.append(f"{counter}. {text}.")
            emitted.add(step.id)
        counter += 1

    return "\n".join(lines)
