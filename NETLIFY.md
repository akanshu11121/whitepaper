# Netlify deployment

The site is now compatible with a **frontend-only Netlify deployment**. Netlify
does not run the repository's Python/PyTorch service; the default deployment uses
the standalone browser lab and bundles the paper catalog, educational content,
equations, traceability documents, and source excerpts during the build.

## Deploy from GitHub

1. Open Netlify and choose **Add new project → Import an existing project**.
2. Select `akanshu11121/whitepaper`.
3. Set the production branch to `main`.
4. Netlify will read `netlify.toml`:
   - Base directory: `frontend`
   - Build command: `npm ci --include=dev && npm run build`
   - Publish directory: `dist`
   - Node: 22
5. Deploy.

The build script reads the root `papers/` directory before Vite runs, so the base
directory setting is important. Do not set the publish directory to the repository
root or to `frontend/src`.

## Modes

### Browser mode (default)

`netlify.toml` sets:

```text
VITE_LAB_MODE=browser
```

This requires no backend, Python runtime, database, or secret. It provides a
bounded, seeded, browser-native attention-routing study. Its UI labels the study
as a reduced experiment rather than presenting it as full Transformer training.

### Full Python API mode

To use the full PyTorch implementation, first deploy `research_lab.api:app` to a
separate HTTPS-capable service. Then add Netlify build environment variables:

```text
VITE_LAB_MODE=api
VITE_API_URL=https://your-api.example/api
```

The API must allow the Netlify origin in CORS. If the API is unavailable, API mode
shows a visible error; it does not silently fabricate server results. `VITE_*`
values are public build-time values, so never place secrets in them.

## Netlify CLI

```bash
npm install --global netlify-cli
netlify login
netlify link
netlify deploy --build
netlify deploy --prod --build
```

Use `netlify deploy --build` for a draft URL first. The production command should
only be run after checking the draft deployment.

## Common 404 fix

The SPA redirect is already in `netlify.toml`:

```toml
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

This is required for refreshing deep links. The Netlify configuration was merged
from the existing Netlify troubleshooting branch and is now part of the feature
branch history.
