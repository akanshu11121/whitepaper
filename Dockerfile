FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LAB_RUNS_DIR=/data/runs LAB_TORCH_THREADS=2
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY papers ./papers
RUN pip install --no-cache-dir .
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN useradd --create-home --uid 10001 lab && mkdir -p /data/runs && chown -R lab:lab /app /data
USER lab
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "research_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
