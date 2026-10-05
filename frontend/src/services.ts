import catalog from "./generated/catalog.json";
import { browserBenchmark, browserExperiment, browserPredict, browserRun } from "./lab/browser";
import type { Benchmark, Equation, ExperimentConfig, Explanation, Inspection, Paper, Run } from "./types";

const configuredApi = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, "");
const mode = import.meta.env.VITE_LAB_MODE?.trim();
if (mode && !["api", "browser", "auto"].includes(mode)) throw new Error("VITE_LAB_MODE must be browser, api, or auto.");
export const API_MODE = mode === "api" || (mode !== "browser" && Boolean(configuredApi));
const API = configuredApi || "/api";
const papers = catalog as unknown as (Paper & {
  content: { concepts: { title: string; explanations: string[]; section: string; source: string }[]; equations: Equation[] };
  docs: Record<string, string>; source: Record<string, string>;
})[];

export function staticPaper(id = "attention") {
  const paper = papers.find(item => item.id === id);
  if (!paper) throw new Error("Paper not found in static catalog.");
  return paper;
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(30000) });
  } catch { throw new Error("Research API is unreachable. Check the HTTPS API URL/CORS settings, or redeploy in browser mode."); }
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Research API returned ${response.status}: ${message.slice(0, 300)}`);
  }
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("API returned HTML instead of JSON. Set VITE_API_URL to the FastAPI service, including /api.");
  return response.json();
}

// Education loads from the bundle even when an optional API is unavailable.
export async function get<T>(path: string): Promise<T> {
  if (path === "/papers") return papers as unknown as T;
  const explanation = path.match(/^\/papers\/([^/]+)\/explanation\?level=(\d)$/);
  if (explanation) {
    const level = Number(explanation[2]);
    if (level > 5) throw new Error("Explanation level must be between 0 and 5.");
    return staticPaper(explanation[1]).content.concepts.map(concept => ({ title: concept.title, body: concept.explanations[level], source: concept.source, section: concept.section } satisfies Explanation)) as T;
  }
  const run = path.match(/^\/papers\/attention\/results\/([^/]+)$/);
  if (run) return (API_MODE ? await request<Run>(path) : browserRun(run[1])) as T;
  throw new Error(`Unsupported route: ${path}`);
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  if (API_MODE) return request<T>(path, body);
  if (path === "/papers/attention/experiment") return browserExperiment(body as ExperimentConfig) as T;
  if (path === "/papers/attention/predict") {
    const payload = body as { run_id: string; text: string };
    return browserPredict(payload.run_id, payload.text) as T;
  }
  if (path === "/papers/attention/benchmark") return await browserBenchmark() as T;
  throw new Error(`Unsupported browser operation: ${path}`);
}

export function startExperiment(config: ExperimentConfig) { return post<Run>("/papers/attention/experiment", config); }
export function predict(run_id: string, text: string) { return post<Inspection>("/papers/attention/predict", { run_id, text, beam_size: 1 }); }
export function benchmark(config: unknown) { return post<Benchmark>("/papers/attention/benchmark", config); }
