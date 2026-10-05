"""Safe JSON and safetensors artifacts with atomic metadata and constrained run IDs."""

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("LAB_RUNS_DIR") or "runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid run ID")
        return self.root / run_id

    def create(self, paper_id: str, kind: str, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        self.path(run_id).mkdir()
        self.write(run_id, "result.json", {
            "run_id": run_id, "paper_id": paper_id, "kind": kind, "status": "queued",
            "created_at": datetime.now(UTC).isoformat(), "config": config,
        })
        return run_id

    def write(self, run_id: str, name: str, value: Any) -> None:
        if name not in {"result.json", "dataset.json", "vocab.json"}:
            raise ValueError("unsupported artifact name")
        path = self.path(run_id) / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def read(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.path(run_id) / "result.json").read_text())

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        value = self.read(run_id)
        value.update(changes)
        self.write(run_id, "result.json", value)
        return value

    def list(self, paper_id: str) -> list[dict[str, Any]]:
        results = []
        for path in self.root.glob("*/result.json"):
            try:
                value = json.loads(path.read_text())
                if value["paper_id"] == paper_id:
                    results.append(value)
            except (ValueError, KeyError, OSError):
                continue
        return sorted(results, key=lambda item: item["created_at"], reverse=True)
