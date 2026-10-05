"""YAML registry loader with a fixed allowlist of import targets."""

from importlib import import_module
from typing import Any

import yaml

from research_lab.runtime import ROOT

REGISTRY = ROOT / "papers" / "registry.yaml"
ALLOWLIST = {"research_lab.papers.attention.module:AttentionModule"}


def entries() -> list[dict[str, Any]]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["papers"]


def get_entry(paper_id: str) -> dict[str, Any]:
    for entry in entries():
        if entry["id"] == paper_id:
            return entry
    raise KeyError(paper_id)


def module(paper_id: str, store=None):
    target = get_entry(paper_id)["module"]
    if target not in ALLOWLIST:
        raise ValueError("module target is not allowlisted")
    path, name = target.split(":", 1)
    return getattr(import_module(path), name)(store=store)
