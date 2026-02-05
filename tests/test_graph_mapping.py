from app.parsing.graph import build_graph, build_steps
from app.parsing.schema import Diagram


def _sample_diagram():
    return Diagram(
        source_file="test.png",
        nodes=[
            {"id": "n0", "type": "Task", "txt": "Start", "cnt": [0, 0], "wh": [10, 10]},
            {"id": "n1", "type": "Gateway", "txt": "", "cnt": [50, 0], "wh": [10, 10]},
            {"id": "n2", "type": "Task", "txt": "Path A", "cnt": [100, 0], "wh": [10, 10]},
            {"id": "n3", "type": "Task", "txt": "Path B", "cnt": [100, 50], "wh": [10, 10]},
        ],
        labels=[
            {"txt": "yes", "cnt": [75, 0], "wh": [10, 5]},
            {"txt": "no", "cnt": [75, 50], "wh": [10, 5]},
        ],
        arrows=[
            {"id": "a0", "tip": [50, 0], "starts": [[0, 0]]},
            {"id": "a1", "tip": [100, 0], "starts": [[50, 0]]},
            {"id": "a2", "tip": [100, 50], "starts": [[50, 0]]},
        ],
    )


def test_graph_edges_and_conditions():
    diagram = _sample_diagram()
    graph = build_graph(diagram)

    edges = {(e.source, e.target): e.condition for e in graph.edges}
    assert ("n0", "n1") in edges
    assert ("n1", "n2") in edges
    assert ("n1", "n3") in edges

    assert edges[("n1", "n2")] == "yes"
    assert edges[("n1", "n3")] == "no"


def test_build_steps_order():
    diagram = _sample_diagram()
    graph = build_graph(diagram)
    steps = build_steps(graph)

    assert steps[0].id == "n0"
    assert any(step.id == "n1" for step in steps)
