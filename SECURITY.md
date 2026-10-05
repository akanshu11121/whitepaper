# Security review

## Implemented controls

- Experiment configuration is parsed by strict Pydantic models with bounded widths,
  sequence lengths, pair counts and update budgets.
- Uploaded parallel text is represented as JSON pairs; reserved tokens, empty values,
  duplicate source sequences, oversized sequences and suspicious train/test overlap are
  rejected. Arbitrary file uploads are not accepted.
- Registry module imports use a fixed allowlist; user input cannot select an arbitrary
  Python import or shell command.
- Artifact IDs are strict 32-character hex strings and supported artifact names are
  allowlisted. Model weights use safetensors; untrusted pickle deserialization is not used.
- YAML is parsed with `yaml.safe_load`. User text is handled as tokens, not executed or
  interpreted as a prompt.
- CORS is limited to local development origins. Mutating API calls support optional
  `LAB_API_TOKEN` / `X-Lab-Token` protection for deployments behind a shared network.
- The container uses a non-root user, a health check, a persistent data volume, and no
  embedded secrets. `.env` files and runtime artifacts are ignored by Git.
- Request method/path/status/latency are emitted as JSON logs and `/api/metrics` exposes
  process-local counts. No user text is included in logs.

## Deployment boundary

The default service is a local research application, not a public multi-tenant service.
Set an API token, put TLS/authentication at an ingress, restrict CORS to the actual UI
origin, isolate the runs volume, and add a rate limit before exposing it remotely.
Do not put credentials, private datasets, or unreviewed model files in the repository.

## Review status

No security scan can certify arbitrary user-supplied datasets or future paper modules.
New modules must preserve the registry allowlist, safe artifact handling, bounded input
validation and the no-pickle policy. Dependency checks are run in CI with `pip-audit`
and `npm audit --audit-level=high`.
