export type Explanation = { title: string; body: string; source?: string; section?: string };
export type Equation = {
  id: string; title: string; section: string; equation: string; symbols: Record<string, string>;
  why: string; inputs: string; outputs: string; intuition: string; example: string;
  implementation: string; assumptions: string; stability: string; source: string;
};
export type Paper = {
  id: string; title: string; year: number; domain: string; task: string; readiness: number;
  authors: string[]; official_url: string; code_url: string; reported_results: Record<string, number>;
};
export type ExperimentConfig = {
  task: "copy" | "reverse"; steps: number; seed: number; train_size: number; eval_size: number;
  min_length: number; max_length: number; symbols: number; model: { d_model: number; heads: number;
  layers: number; d_ff: number; dropout: number; attention_dropout: number; positions: string;
  max_length: number; optimized: boolean }; warmup_steps: number; smoothing: number;
  model_type: "transformer"; device: "cpu"; batch_size: number;
};
export type HistoryPoint = { step: number; loss: number; learning_rate: number };
export type Example = { source: string; target: string; prediction: string[]; correct: boolean };
export type Run = {
  run_id: string; status: string; test?: Record<string, number>; history?: HistoryPoint[];
  parameter_count?: number; duration_seconds?: number; scientific_status?: string;
  config?: ExperimentConfig; baselines?: Record<string, Record<string, number>>;
  examples?: Example[]; error_type?: string; note?: string; environment?: Record<string, unknown>;
};
export type Inspection = {
  tokens: string[]; confidence: number[]; attention: Record<string, number[][][][]>;
  source_labels: string[]; target_labels: string[]; confidence_note: string;
  terminated_with_eos: boolean; unknown_tokens?: string[]; note?: string;
};
export type LoraPrediction = { input: number[]; output: number[]; merged_output: number[]; update_matrix: number[][]; max_merge_difference: number; note: string };
export type Benchmark = {
  runtime: string; paths: Record<string, { latency_ms: { p50: number; p95: number; mean: number };
  throughput_sequences_per_second: number }>; max_absolute_difference?: number;
  attention_matrix_bytes_fp32?: number; parameter_count?: number; environment: Record<string, unknown>;
  note?: string;
};
