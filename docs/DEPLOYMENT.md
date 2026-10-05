# Deployment and operations

For Netlify-only hosting, follow [`NETLIFY.md`](../NETLIFY.md). Netlify serves the
frontend; it does not execute the Python/PyTorch backend. The default site is a
standalone browser lab, while the full API is an optional separately hosted service.

## Local production-shaped run

```bash
docker compose up --build
curl http://localhost:8000/api/health
```

The container serves the built React application and FastAPI from port 8000. The
`research_runs` volume persists JSON provenance and safetensors checkpoints. Configure
`LAB_RUNS_DIR` and `LAB_TORCH_THREADS` through the environment.

## Remote deployment checklist

1. Supply `LAB_API_TOKEN` from a secret manager and send it as `X-Lab-Token` on POSTs.
2. Put the container behind TLS, an ingress rate limit and identity-aware access control.
3. Replace local CORS origins with the deployed UI origin.
4. Mount encrypted persistent artifact storage and apply dataset retention rules.
5. Monitor JSON logs, `/api/health`, `/api/metrics`, failed run statuses and latency.
6. Run CI lint/type/unit/frontend/build/audit jobs before publishing an image.
7. Keep deployment promotion manual; the supplied workflow only builds images on
   `workflow_dispatch` and does not deploy to production.

GPU/VRAM monitoring and cloud cost tracking are not enabled by default because this
repository is CPU-first. Add them at the deployment layer when GPU training is required.
