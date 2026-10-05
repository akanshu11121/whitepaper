"""Reproducibility metadata and serialized RNG-dependent execution."""

import hashlib
import os
import platform
import random
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

ROOT = Path(os.getenv("LAB_PROJECT_ROOT", str(Path.cwd() if (Path.cwd() / "papers/registry.yaml").exists() else Path(__file__).resolve().parents[2]))).resolve()
COMPUTE_LOCK = threading.RLock()
torch.set_num_threads(int(os.getenv("LAB_TORCH_THREADS", "2")))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "os": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_logical_count": psutil.cpu_count(), "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_revision": revision, "source_sha256": source_hash(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "bitwise_reproducibility_claimed": False,
    }
