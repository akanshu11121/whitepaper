/** A bounded attention-routing experiment, not the full encoder–decoder Transformer.
 * Learn Q[length][target_position] and shared K[source_position] from known alignments.
 * V is one-hot token identity. Optimize alignment cross entropy with explicit SGD.
 */
import { attention, percentile, seededRandom } from "./attention";
import type { Benchmark, ExperimentConfig, HistoryPoint, Inspection, Run } from "../types";

const WIDTH = 16;
const MAX_LENGTH = 8;
const LEARNING_RATE = 0.8;
export const BROWSER_NOTE = "Browser-native attention-routing study: learned position queries/keys, one-hot token values, known copy/reversal alignment supervision, and SGD. This reduced experiment is not the paper's full Transformer or PyTorch training.";
type Router = { queries: number[][][]; keys: number[][]; config: ExperimentConfig };
const models = new Map<string, Router>();
const runs = new Map<string, Run>();

export function browserRun(id: string) {
  const run = runs.get(id);
  if (!run) throw new Error("Browser run not found. Start a new experiment after reloading the page.");
  return run;
}

function initialize(config: ExperimentConfig): Router {
  const random = seededRandom(config.seed);
  const vector = () => Array.from({ length: WIDTH }, () => (random() - .5) * 1.6);
  return {
    queries: Array.from({ length: MAX_LENGTH + 1 }, (_, length) => Array.from({ length }, vector)),
    keys: Array.from({ length: MAX_LENGTH }, vector), config,
  };
}

function probabilities(model: Router, length: number) {
  return attention(model.queries[length], model.keys.slice(0, length),
    Array.from({ length }, (_, i) => Array.from({ length }, (_, j) => Number(i === j)))).weights;
}

function update(model: Router, length: number) {
  const weights = probabilities(model, length);
  const q = model.queries[length], k = model.keys;
  const dq = q.map(() => Array(WIDTH).fill(0) as number[]);
  const dk = k.map(() => Array(WIDTH).fill(0) as number[]);
  let loss = 0;
  for (let i = 0; i < length; i++) {
    const target = model.config.task === "copy" ? i : length - 1 - i;
    loss -= Math.log(Math.max(weights[i][target], 1e-300)) / length;
    for (let j = 0; j < length; j++) {
      const gradient = (weights[i][j] - Number(j === target)) / (length * Math.sqrt(WIDTH));
      for (let d = 0; d < WIDTH; d++) {
        dq[i][d] += gradient * k[j][d]; dk[j][d] += gradient * q[i][d];
      }
    }
  }
  q.forEach((row, i) => row.forEach((_, d) => { row[d] -= LEARNING_RATE * dq[i][d]; }));
  k.forEach((row, j) => row.forEach((_, d) => { row[d] -= LEARNING_RATE * dk[j][d]; }));
  return loss;
}

function generate(model: Router, sequence: number[]) {
  const values = sequence.map(token => Array.from({ length: model.config.symbols }, (_, index) => Number(token === index)));
  const result = attention(model.queries[sequence.length], model.keys.slice(0, sequence.length), values);
  const tokens = result.output.map(row => row.indexOf(Math.max(...row)));
  return { tokens, confidence: result.output.map((row, i) => row[tokens[i]]), weights: result.weights };
}

export function browserExperiment(config: ExperimentConfig): Run {
  if (!Number.isInteger(config.steps) || config.steps < 1 || config.steps > 5000 ||
      !Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295 ||
      config.min_length !== 3 || config.max_length !== MAX_LENGTH || config.symbols !== 12 ||
      !["copy", "reverse"].includes(config.task)) throw new Error("Browser study supports seed≥0, 1–5000 updates, lengths 3–8 and 12 symbols.");
  const id = crypto.randomUUID();
  const run: Run = { run_id: id, status: "running", config, history: [], note: BROWSER_NOTE, scientific_status: "browser-derived-alignment-study" };
  runs.set(id, run);
  void trainRouter(id, config);
  return run;
}

async function trainRouter(id: string, config: ExperimentConfig) {
  const run = browserRun(id);
  try {
    const model = initialize(config);
    const start = performance.now();
    const history: HistoryPoint[] = [];
    for (let step = 1; step <= config.steps; step++) {
      // Every update covers all supported lengths; no fabricated loss curve.
      const loss = Array.from({ length: 6 }, (_, i) => update(model, i + 3)).reduce((a, b) => a + b, 0) / 6;
      if (step === 1 || step % Math.max(1, Math.floor(config.steps / 20)) === 0 || step === config.steps) {
        history.push({ step, loss, learning_rate: LEARNING_RATE });
        run.history = [...history];
      }
      if (step % 20 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }
    const random = seededRandom(config.seed ^ 0x12345678);
    const examples = Array.from({ length: config.eval_size }, () => {
      const length = 3 + Math.floor(random() * 6);
      const source = Array.from({ length }, () => Math.floor(random() * 12));
      const target = config.task === "copy" ? source : [...source].reverse();
      const prediction = generate(model, source).tokens;
      return { source: source.join(" "), target: target.join(" "), prediction: prediction.map(String),
        correct: target.every((token, i) => prediction[i] === token) };
    });
    const total = examples.reduce((sum, example) => sum + example.prediction.length, 0);
    const correct = examples.reduce((sum, example) => sum + example.prediction.filter((token, i) => token === example.target.split(" ")[i]).length, 0);
    let alignmentNll = 0;
    for (let length = 3; length <= MAX_LENGTH; length++) {
      const weights = probabilities(model, length);
      alignmentNll += weights.reduce((sum, row, i) => sum - Math.log(Math.max(row[config.task === "copy" ? i : length - 1 - i], 1e-300)), 0) / length / 6;
    }
    models.set(id, model);
    Object.assign(run, {
      status: "completed", duration_seconds: (performance.now() - start) / 1000,
      parameter_count: model.queries.reduce((sum, rows) => sum + rows.length * WIDTH, 0) + MAX_LENGTH * WIDTH,
      test: { exact_match: examples.filter(x => x.correct).length / examples.length, token_accuracy: correct / total, alignment_nll: alignmentNll },
      examples, environment: { runtime: "JavaScript Number / float64", hardware: "visitor browser CPU", user_agent: navigator.userAgent,
        seed: config.seed, training_data: "known positional alignments, lengths 3–8", evaluation_data: "seeded numeric sequences; same lengths; new token sequences; not a translation corpus", nondeterminism: "timing depends on device/browser; metrics seeded within this implementation" },
    });
  } catch (error) { run.status = "failed"; run.error_type = error instanceof Error ? error.message : "Browser execution failed"; }
}

export function browserPredict(id: string, text: string): Inspection {
  const model = models.get(id);
  if (!model) throw new Error("Complete a browser experiment before predicting.");
  const words = text.trim().split(/\s+/);
  if (!text.trim() || words.length < 3 || words.length > MAX_LENGTH || words.some(word => !/^(?:[0-9]|1[01])$/.test(word))) {
    throw new Error("Enter 3–8 integer tokens in the range 0–11 (for example: 0 1 2 3).");
  }
  const { tokens, confidence, weights } = generate(model, words.map(Number));
  return { tokens: tokens.map(String), confidence, attention: { cross: [[weights]] }, source_labels: words,
    target_labels: tokens.map(String), terminated_with_eos: false,
    confidence_note: "Attention-weighted token mass, not autoregressive confidence; fixed-length routing has no EOS.", note: BROWSER_NOTE };
}

export async function browserBenchmark(): Promise<Benchmark> {
  const random = seededRandom(42);
  const matrix = (rows: number, width: number) => Array.from({ length: rows }, () => Array.from({ length: width }, () => random() - .5));
  const q = matrix(32, 16), k = matrix(32, 16), v = matrix(32, 16);
  for (let i = 0; i < 5; i++) attention(q, k, v);
  await new Promise(resolve => setTimeout(resolve, 0));
  // Batch repetitions to reduce timer-resolution noise for short kernels.
  const samples = [];
  for (let i = 0; i < 30; i++) {
    const start = performance.now();
    for (let j = 0; j < 10; j++) attention(q, k, v);
    samples.push(Math.max(1e-6, (performance.now() - start) / 10));
    if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return { runtime: "browser", paths: { javascript: { latency_ms: { p50: percentile(samples, 50), p95: percentile(samples, 95), mean }, throughput_sequences_per_second: 1000 / mean } },
    environment: { user_agent: navigator.userAgent, sequence_length: 32, key_width: 16, repeats: 30, inner_repeats: 10, warmup: 5 },
    note: "Measured JavaScript Eq.1 kernel on this browser/device. Allocation is included. No PyTorch/SDPA, GPU, peak RAM, or server throughput claim." };
}
