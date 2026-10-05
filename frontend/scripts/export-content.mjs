/** Bundle the existing paper registry/content without a Python build dependency. */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { parse } from "yaml";

const frontend = fileURLToPath(new URL("../", import.meta.url));
const root = path.resolve(frontend, "..");
const registry = parse(await readFile(path.join(root, "papers/registry.yaml"), "utf8"));
const papers = [];
const sourceNames = {
  attention: ["attention", "model", "training", "inference", "runner", "baseline", "ablations", "benchmark"],
  lora: ["config", "layers", "models", "runner", "module"],
};

for (const entry of registry.papers) {
  const metadata = parse(await readFile(path.join(root, entry.metadata), "utf8"));
  const content = parse(await readFile(path.join(root, entry.content), "utf8"));
  const destination = path.join(frontend, "public/research", entry.id);
  await mkdir(destination, { recursive: true });
  const docs = {};
  for (const name of ["VALIDATION", "TRACEABILITY", "README", "EXPERIMENTS", "SCORECARD"]) {
    const text = await readFile(path.join(root, "papers", entry.id, `${name}.md`), "utf8");
    const publicPath = `/research/${entry.id}/${name.toLowerCase()}.md`;
    await writeFile(path.join(destination, `${name.toLowerCase()}.md`), text);
    docs[name.toLowerCase()] = publicPath;
  }
  const source = {};
  if (sourceNames[entry.id]) {
    for (const name of sourceNames[entry.id] ?? []) {
      const sourcePath = `src/research_lab/papers/${entry.id}/${name}.py`;
      source[sourcePath] = await readFile(path.join(root, sourcePath), "utf8");
    }
  }
  papers.push({
    ...entry, ...metadata, content, docs, source,
    code_url: metadata.official_implementation,
    readiness: 80,
    assessment_note: "Engineering assessment of the Python module; not certification of a Netlify-hosted full Transformer.",
    reported_results: { wmt14_en_de_big_bleu: 28.4, wmt14_en_fr_big_bleu: 41.8, wmt14_en_fr_section_6_1_bleu: 41.0 },
  });
}

await mkdir(path.join(frontend, "src/generated"), { recursive: true });
await writeFile(path.join(frontend, "src/generated/catalog.json"), JSON.stringify(papers, null, 2));
console.log(`Bundled ${papers.length} paper(s), educational content, source, and public documentation.`);
