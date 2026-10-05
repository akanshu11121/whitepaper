# Contributing a paper module

1. Add the source material and verify title, authors, venue, identifier, license,
   official links and available implementation.
2. Write `VALIDATION.md` before coding. Classify each component as reproducible,
   partially reproducible, or not reproducible; record every assumption.
3. Create `papers/<id>/README.md`, `TRACEABILITY.md`, metadata/content, and a module
   implementing the typed contract in `research_lab/core.py`.
4. Keep equations and paper-faithful paths explicit. Optimizations must be named and
   measured separately. Never insert author results as local results.
5. Add meaningful numerical, scientific, integration and edge-case tests, then wire
   the module through `papers/registry.yaml` and an allowlist entry.
6. Persist config, seed, environment, data fingerprint, artifacts and status for all
   experiments. Validate uploaded data, reject path traversal and never deserialize
   untrusted pickle files.
7. Update the UI only through API contracts and add traceability rows for substantial
   paper claims.

Before submitting:

```bash
uv run pytest
uv run ruff check src tests
uv run mypy
cd frontend && npm run build
```

Keep changes focused and explain any missing evidence in the paper module report.
