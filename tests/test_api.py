from fastapi.testclient import TestClient

from research_lab.api import app


def test_registry_and_explanation_endpoints():
    client = TestClient(app)
    response = client.get("/api/papers")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "attention"
    response = client.get("/api/papers/attention/explanation?level=2")
    assert response.status_code == 200
    assert len(response.json()) == 7


def test_invalid_config_is_rejected():
    client = TestClient(app)
    response = client.post("/api/papers/attention/experiment", json={"model": {"d_model": 7, "heads": 2}})
    assert response.status_code == 422


def test_content_and_equation_endpoints_are_traceable():
    client = TestClient(app)
    content = client.get("/api/papers/attention/content")
    equations = client.get("/api/papers/attention/equations")
    assert content.status_code == 200
    assert equations.status_code == 200
    assert any(item["id"] == "attention" for item in equations.json())
    assert content.json()["concepts"][0]["source"].endswith("attention.py")


def test_health_and_process_metrics_are_available():
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["requests"] >= 1
    assert "X-Request-Duration-Ms" in metrics.headers
