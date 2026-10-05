// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { browserBenchmark, browserExperiment, browserPredict, browserRun } from "./browser";
import type { ExperimentConfig } from "../types";

const config: ExperimentConfig = {
  task: "reverse", steps: 3, seed: 42, train_size: 32, eval_size: 4,
  min_length: 3, max_length: 8, symbols: 12, batch_size: 8, warmup_steps: 2,
  smoothing: 0.1, model_type: "transformer", device: "cpu",
  model: { d_model: 64, heads: 4, layers: 2, d_ff: 128, dropout: 0.1,
    attention_dropout: 0, positions: "sinusoidal", max_length: 128, optimized: false },
};

describe("standalone browser research lab", () => {
  it("trains, polls, predicts, and benchmarks without the API", async () => {
    const initial = browserExperiment(config);
    let run = browserRun(initial.run_id);
    for (let attempt = 0; attempt < 100 && run.status === "running"; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 0));
      run = browserRun(initial.run_id);
    }
    expect(run.status).toBe("completed");
    expect(run.note).toContain("not the paper's full Transformer");
    const prediction = browserPredict(initial.run_id, "0 1 2");
    expect(prediction.attention.cross.length).toBe(1);
    expect(prediction.tokens).toHaveLength(3);
    const benchmark = await browserBenchmark();
    expect(benchmark.runtime).toBe("browser");
    expect(benchmark.paths.javascript.latency_ms.p50).toBeGreaterThan(0);
  });
});
