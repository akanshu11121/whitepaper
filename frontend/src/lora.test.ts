// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { browserLoraBenchmark, browserLoraPredict, browserLoraRun, browserLoraRunStatus } from "./lab/lora";

describe("standalone browser LoRA study", () => {
  it("trains a low-rank update, exposes it, and compares merge paths", async () => {
    const run = browserLoraRun({ seed: 7, input_dim: 8, output_dim: 6, rank: 2, target_rank: 2, alpha: 2, steps: 3, train_size: 16, eval_size: 4, learning_rate: .1 });
    let current = browserLoraRunStatus(run.run_id);
    for (let attempt = 0; attempt < 100 && current.status === "running"; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 0));
      current = browserLoraRunStatus(run.run_id);
    }
    expect(current.status).toBe("completed");
    const prediction = browserLoraPredict(run.run_id, "1 0 0 0 0 0 0 0");
    expect(prediction.output).toHaveLength(6);
    expect(prediction.max_merge_difference).toBeLessThan(1e-8);
    const benchmark = await browserLoraBenchmark();
    expect(benchmark.paths.merged).toBeTruthy();
  });
});
