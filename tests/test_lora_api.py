import time

from fastapi.testclient import TestClient

from research_lab.api import app


def test_lora_registry_content_and_equations():
    client = TestClient(app)
    papers = client.get("/api/papers").json()
    assert {paper["id"] for paper in papers} == {"attention", "lora"}
    content = client.get("/api/papers/lora/content")
    equations = client.get("/api/papers/lora/equations")
    assert content.status_code == 200
    assert equations.status_code == 200
    assert any(item["id"] == "lora-update" for item in equations.json())


def test_lora_benchmark_endpoint():
    client = TestClient(app)
    response = client.post("/api/papers/lora/benchmark", json={"input_dim": 8, "output_dim": 8, "rank": 2, "repeats": 5, "warmup": 2})
    assert response.status_code == 200
    assert response.json()["scientific_status"] == "engineering-measurement"


def test_lora_experiment_round_trip():
    client = TestClient(app)
    response = client.post("/api/papers/lora/experiment", json={"seed": 7, "input_dim": 6, "output_dim": 4, "rank": 2, "target_rank": 2, "steps": 5, "train_size": 32, "eval_size": 8, "batch_size": 8, "learning_rate": 0.08})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    result = None
    for _ in range(100):
        result = client.get(f"/api/papers/lora/results/{run_id}").json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["artifacts"]["adapter"] == "adapter.safetensors"
