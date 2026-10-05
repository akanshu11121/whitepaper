/** Reduced browser experiment for LoRA: fit a known low-rank matrix update. */
import { percentile, seededRandom } from "./attention";
import type { Benchmark, LoraPrediction, Run } from "../types";

export const LORA_BROWSER_NOTE = "Browser-native low-rank matrix adaptation: a frozen base matrix, trainable A/B factors, known low-rank target update, and SGD. This is a mechanism visualization, not GLUE/GPT-2/GPT-3 reproduction.";
type LoraConfig = { seed: number; input_dim: number; output_dim: number; rank: number; target_rank: number; alpha: number; steps: number; train_size: number; eval_size: number; learning_rate: number };
type Matrix = number[][];
type State = { config: LoraConfig; base: Matrix; a: Matrix; b: Matrix; scale: number; target: Matrix };
const states = new Map<string, State>();
const runs = new Map<string, Run>();

const zeros = (rows: number, cols: number): Matrix => Array.from({ length: rows }, () => Array(cols).fill(0));
const randomMatrix = (rows: number, cols: number, random: () => number, scale = 1): Matrix => Array.from({ length: rows }, () => Array.from({ length: cols }, () => (random() - .5) * 2 * scale));
const multiply = (left: Matrix, right: Matrix): Matrix => left.map(row => right[0].map((_, j) => row.reduce((sum, value, k) => sum + value * right[k][j], 0)));
const transpose = (matrix: Matrix): Matrix => matrix[0].map((_, j) => matrix.map(row => row[j]));
const add = (left: Matrix, right: Matrix): Matrix => left.map((row, i) => row.map((value, j) => value + right[i][j]));
const scale = (matrix: Matrix, factor: number): Matrix => matrix.map(row => row.map(value => value * factor));
const mse = (left: Matrix, right: Matrix) => left.reduce((sum, row, i) => sum + row.reduce((inner, value, j) => inner + (value - right[i][j]) ** 2, 0), 0) / Math.max(1, left.length * (left[0]?.length ?? 1));

function forward(state: State, inputs: Matrix) {
  const update = scale(multiply(state.b, state.a), state.scale);
  return multiply(inputs, transpose(add(state.base, update)));
}

function trainStep(state: State, inputs: Matrix, targets: Matrix) {
  const predictions = forward(state, inputs);
  const error = scale(predictions.map((row, i) => row.map((value, j) => value - targets[i][j])), 2 / inputs.length);
  const gradWeight = multiply(transpose(error), inputs);
  const gradB = scale(multiply(gradWeight, transpose(state.a)), state.scale);
  const gradA = scale(multiply(transpose(state.b), gradWeight), state.scale);
  state.a = state.a.map((row, i) => row.map((value, j) => value - state.config.learning_rate * gradA[i][j]));
  state.b = state.b.map((row, i) => row.map((value, j) => value - state.config.learning_rate * gradB[i][j]));
  return mse(predictions, targets);
}

export function browserLoraRun(config: LoraConfig): Run {
  if (config.rank > Math.min(config.input_dim, config.output_dim) || config.steps < 1 || config.steps > 3000) throw new Error("Browser LoRA requires a valid rank and 1–3000 updates.");
  const random = seededRandom(config.seed);
  const base = randomMatrix(config.output_dim, config.input_dim, random, .3);
  const trueA = randomMatrix(config.target_rank, config.input_dim, random, .5);
  const trueB = randomMatrix(config.output_dim, config.target_rank, random, .5);
  const target = add(base, scale(multiply(trueB, trueA), config.alpha / config.target_rank));
  const state: State = { config, base, a: randomMatrix(config.rank, config.input_dim, random, .04), b: zeros(config.output_dim, config.rank), scale: config.alpha / config.rank, target };
  const inputs = randomMatrix(config.train_size + config.eval_size, config.input_dim, random, 1);
  const targets = multiply(inputs, transpose(target));
  const runId = crypto.randomUUID();
  const run: Run = { run_id: runId, status: "running", scientific_status: "browser-derived-low-rank-study", note: LORA_BROWSER_NOTE, config: config as never, history: [] };
  runs.set(runId, run);
  states.set(runId, state);
  void (async () => {
    const history = [];
    const start = performance.now();
    for (let step = 1; step <= config.steps; step++) {
      const loss = trainStep(state, inputs.slice(0, config.train_size), targets.slice(0, config.train_size));
      if (step === 1 || step % Math.max(1, Math.floor(config.steps / 20)) === 0 || step === config.steps) {
        history.push({ step, loss, learning_rate: config.learning_rate }); run.history = [...history];
      }
      if (step % 20 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }
    const testPredictions = forward(state, inputs.slice(config.train_size));
    const testTargets = targets.slice(config.train_size);
    const update = scale(multiply(state.b, state.a), state.scale);
    Object.assign(run, { status: "completed", duration_seconds: (performance.now() - start) / 1000,
      parameter_count: config.rank * (config.input_dim + config.output_dim),
      test: { lora_test_mse: mse(testPredictions, testTargets), frozen_test_mse: mse(multiply(inputs.slice(config.train_size), transpose(base)), testTargets),
        trainable_fraction: (config.rank * (config.input_dim + config.output_dim)) / (config.input_dim * config.output_dim) },
      environment: { runtime: "JavaScript Number / float64", hardware: "visitor browser CPU", seed: config.seed, training_data: "generated vectors", evaluation_data: "held-out generated vectors" },
      lora_update: update, base_matrix: base });
  })();
  return run;
}

export function browserLoraRunStatus(id: string) { const run = runs.get(id); if (!run) throw new Error("Browser LoRA run not found."); return run; }

export function browserLoraPredict(id: string, text: string): LoraPrediction {
  const state = states.get(id); if (!state) throw new Error("Complete a browser LoRA experiment before predicting.");
  const input = text.replace(/,/g, " ").trim().split(/\s+/).map(Number);
  if (input.length !== state.config.input_dim || input.some(value => !Number.isFinite(value))) throw new Error(`Enter exactly ${state.config.input_dim} finite numeric values.`);
  const vector = [input]; const unmerged = forward(state, vector)[0];
  const merged = multiply(vector, transpose(add(state.base, scale(multiply(state.b, state.a), state.scale))))[0];
  return { input, output: unmerged, merged_output: merged, update_matrix: scale(multiply(state.b, state.a), state.scale), max_merge_difference: Math.max(...unmerged.map((value, i) => Math.abs(value - merged[i]))), note: LORA_BROWSER_NOTE };
}

export async function browserLoraBenchmark(): Promise<Benchmark> {
  const random = seededRandom(12); const input = randomMatrix(1, 128, random); const base = randomMatrix(128, 128, random, .01); const a = randomMatrix(4, 128, random, .02); const b = randomMatrix(128, 4, random, .02); const merged = add(base, scale(multiply(b, a), 2));
  const times = (fn: () => void) => { const values = []; for (let i = 0; i < 30; i++) { const start = performance.now(); for (let j = 0; j < 10; j++) fn(); values.push((performance.now() - start) / 10); } return values; };
  const unmerged = times(() => { multiply(input, transpose(base)); const update = multiply(input, transpose(scale(multiply(b, a), 2))); add(multiply(input, transpose(base)), update); });
  const mergedTimes = times(() => multiply(input, transpose(merged)));
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const stats = (values: number[]) => ({ p50: percentile(values, 50), p95: percentile(values, 95), mean: mean(values) });
  return { runtime: "browser", paths: { unmerged: { latency_ms: stats(unmerged), throughput_sequences_per_second: 1000 / mean(unmerged) }, merged: { latency_ms: stats(mergedTimes), throughput_sequences_per_second: 1000 / mean(mergedTimes) } }, environment: { sequence_length: 1, input_dim: 128, output_dim: 128, rank: 4, repeats: 30 }, note: "Browser float64/JavaScript matrix benchmark; allocation and timer resolution are included. This is not a GPU or LLM benchmark." };
}
