import os

os.environ["GPU_ONLY"] = "0"
os.environ["LLAMA_DISABLE"] = "1"

from fastapi.testclient import TestClient

from app.main import app


def _sample_diagram():
    return {
        "source_file": "test.png",
        "nodes": [
            {"id": "n0", "type": "Task", "txt": "Start", "cnt": [0, 0], "wh": [10, 10]},
            {"id": "n1", "type": "Gateway", "txt": "", "cnt": [50, 0], "wh": [10, 10]},
            {"id": "n2", "type": "Task", "txt": "Path A", "cnt": [100, 0], "wh": [10, 10]},
            {"id": "n3", "type": "Task", "txt": "Path B", "cnt": [100, 50], "wh": [10, 10]},
        ],
        "labels": [
            {"txt": "yes", "cnt": [75, 0], "wh": [10, 5]},
            {"txt": "no", "cnt": [75, 50], "wh": [10, 5]},
        ],
        "arrows": [
            {"id": "a0", "tip": [50, 0], "starts": [[0, 0]]},
            {"id": "a1", "tip": [100, 0], "starts": [[50, 0]]},
            {"id": "a2", "tip": [100, 50], "starts": [[50, 0]]},
        ],
    }


def test_describe_endpoint():
    client = TestClient(app)
    response = client.post("/api/describe", json=_sample_diagram())
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload.get("rows")
