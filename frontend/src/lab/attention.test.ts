import { describe, expect, it } from "vitest";
import { attention } from "./attention";

describe("browser Eq. 1 attention", () => {
  it("scales dot products and returns a weighted value", () => {
    const result = attention([[1, 0]], [[1, 0], [0, 1]], [[2, 0], [0, 4]]);
    expect(result.weights[0][0]).toBeCloseTo(0.6698, 3);
    expect(result.output[0][0]).toBeCloseTo(1.3395, 3);
  });

  it("respects causal masks and produces a zero all-masked row", () => {
    const allowed = [[true, false], [true, true]];
    const result = attention([[1, 0], [1, 0]], [[1, 0], [0, 1]], [[2, 0], [0, 4]], allowed);
    expect(result.weights[0][1]).toBe(0);
    const empty = attention([[1, 0]], [[1, 0]], [[2, 4]], [[false]]);
    expect(empty.output[0]).toEqual([0, 0]);
  });

  it("rejects incompatible input shapes", () => {
    expect(() => attention([[1]], [[1, 0]], [[1, 0]])).toThrow();
  });
});
