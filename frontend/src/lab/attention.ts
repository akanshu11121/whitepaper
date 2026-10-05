/** Paper Eq.1 implemented using JavaScript numbers (float64); True=allowed. */
export type Matrix = number[][];

export function attention(q: Matrix, k: Matrix, v: Matrix, allowed?: boolean[][]) {
  const width = q[0]?.length;
  const valueWidth = v[0]?.length;
  if (!width || !valueWidth || !k.length || k.length !== v.length ||
      q.some(row => row.length !== width) || k.some(row => row.length !== width) ||
      v.some(row => row.length !== valueWidth) || [q, k, v].some(m => m.flat().some(x => !Number.isFinite(x)))) {
    throw new Error("Q/K widths and K/V lengths must match; matrices must be nonempty and finite.");
  }
  if (allowed && (allowed.length !== q.length || allowed.some(row => row.length !== k.length))) {
    throw new Error("Mask must have shape [queries, keys].");
  }
  const scores = q.map((query, i) => k.map((key, j) => {
    if (allowed && !allowed[i][j]) return -Infinity;
    const score = query.reduce((sum, x, d) => sum + x * key[d], 0) / Math.sqrt(width);
    if (!Number.isFinite(score)) throw new Error("Attention dot product overflowed; reduce input magnitudes.");
    return score;
  }));
  const weights = scores.map(row => {
    const maximum = Math.max(...row);
    if (maximum === -Infinity) return row.map(() => 0);
    const values = row.map(x => Math.exp(x - maximum));
    const sum = values.reduce((a, b) => a + b, 0);
    return values.map(x => x / sum);
  });
  const output = weights.map(row => Array.from({ length: valueWidth }, (_, d) =>
    row.reduce((sum, weight, j) => sum + weight * v[j][d], 0)));
  return { scores, weights, output };
}

export function seededRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let x = value;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function percentile(samples: number[], percent: number) {
  const sorted = [...samples].sort((a, b) => a - b);
  const position = (sorted.length - 1) * percent / 100;
  const low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
