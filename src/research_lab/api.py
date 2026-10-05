"""Registry-driven FastAPI transport; request validation stays in Pydantic configs."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from research_lab.artifacts import ArtifactStore
from research_lab.config import BenchmarkConfig, ExperimentConfig
from research_lab.registry import entries, module

app = FastAPI(title="AI Research Implementation Lab", version="0.1.0")
logger = logging.getLogger("research_lab.api")
metrics = {"requests": 0, "errors": 0, "total_latency_ms": 0.0}
api_token = os.getenv("LAB_API_TOKEN", "")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
store = ArtifactStore()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-run")


@app.middleware("http")
async def observe_and_authorize(request: Request, call_next):
    started = time.perf_counter()
    if api_token and request.method == "POST" and request.headers.get("X-Lab-Token") != api_token:
        metrics["errors"] += 1
        return JSONResponse(status_code=401, content={"detail": "authentication required"})
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics["requests"] += 1
    metrics["total_latency_ms"] += elapsed_ms
    if response.status_code >= 500:
        metrics["errors"] += 1
    response.headers["X-Request-Duration-Ms"] = f"{elapsed_ms:.3f}"
    logger.info(json.dumps({"method": request.method, "path": request.url.path, "status": response.status_code, "latency_ms": round(elapsed_ms, 3)}))
    return response


def get_module(paper_id: str):
    try:
        return module(paper_id, store=store)
    except KeyError:
        raise HTTPException(404, "paper not found") from None
    except ValueError:
        raise HTTPException(500, "paper module is not available") from None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics_endpoint() -> dict[str, float]:
    requests = metrics["requests"]
    return {**metrics, "average_latency_ms": metrics["total_latency_ms"] / max(requests, 1)}


@app.get("/api/papers")
def papers() -> list[dict[str, Any]]:
    return [{**entry, **get_module(entry["id"]).metadata()} for entry in entries()]


@app.get("/api/papers/{paper_id}")
def paper(paper_id: str) -> dict[str, Any]:
    return get_module(paper_id).metadata()


@app.get("/api/papers/{paper_id}/explanation")
def explanation(paper_id: str, level: int = 5) -> list[dict[str, Any]]:
    if level < 0 or level > 5:
        raise HTTPException(422, "level must be between 0 and 5")
    return get_module(paper_id).explain(level)


@app.get("/api/papers/{paper_id}/validation")
def validation(paper_id: str) -> dict[str, Any]:
    return get_module(paper_id).validate()


@app.get("/api/papers/{paper_id}/content")
def content(paper_id: str) -> dict[str, Any]:
    research_module = get_module(paper_id)
    if not hasattr(research_module, "content"):
        raise HTTPException(404, "paper content is not available")
    return research_module.content()


@app.get("/api/papers/{paper_id}/equations")
def equations(paper_id: str) -> list[dict[str, Any]]:
    value = content(paper_id)
    return value.get("equations", [])


@app.get("/api/papers/{paper_id}/references")
def references(paper_id: str) -> list[dict[str, Any]]:
    return get_module(paper_id).get_references()


@app.get("/api/papers/{paper_id}/limitations")
def limitations(paper_id: str) -> list[str]:
    return get_module(paper_id).get_limitations()


def execute(run_id: str, config: ExperimentConfig) -> None:
    try:
        store.update(run_id, status="running")
        get_module("attention").run_experiment(config, run_id)
    except Exception as exc:  # persisted status is safe; detail is server-side only
        store.update(run_id, status="failed", error_type=type(exc).__name__)


@app.post("/api/papers/{paper_id}/experiment", status_code=202)
def experiment(paper_id: str, config: ExperimentConfig) -> dict[str, Any]:
    if paper_id != "attention":
        raise HTTPException(404, "paper does not support experiments")
    run_id = store.create(paper_id, "experiment", config.model_dump())
    executor.submit(execute, run_id, config)
    return {"run_id": run_id, "status": "queued"}


@app.post("/api/papers/{paper_id}/benchmark")
def run_benchmark(paper_id: str, config: BenchmarkConfig) -> dict[str, Any]:
    return get_module(paper_id).benchmark(config)


@app.get("/api/papers/{paper_id}/results")
def results(paper_id: str) -> list[dict[str, Any]]:
    get_module(paper_id)
    return store.list(paper_id)


@app.get("/api/papers/{paper_id}/results/{run_id}")
def result(paper_id: str, run_id: str) -> dict[str, Any]:
    get_module(paper_id)
    try:
        value = store.read(run_id)
    except (ValueError, FileNotFoundError, OSError):
        raise HTTPException(404, "run not found") from None
    if value.get("paper_id") != paper_id:
        raise HTTPException(404, "run not found")
    return value


@app.post("/api/papers/{paper_id}/predict")
def predict(paper_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if paper_id != "attention":
        raise HTTPException(404, "paper does not support prediction")
    run_id, text = payload.get("run_id"), payload.get("text")
    if not isinstance(run_id, str) or not isinstance(text, str):
        raise HTTPException(422, "run_id and text are required")
    try:
        run = store.read(run_id)
        if run.get("status") != "completed":
            raise ValueError("run is not completed")
        return get_module(paper_id).predict(run_id, text, int(payload.get("beam_size", 1)))
    except (ValueError, FileNotFoundError, OSError, KeyError):
        raise HTTPException(400, "prediction could not be executed") from None


FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
