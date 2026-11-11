// Headless numeric harness for EdgeOptimizer.trainHead (run by tests/test_multiplatform_edge_fl.py).
// Prints a JSON report on stdout.
const path = require('path');
global.window = { crypto: require('crypto').webcrypto };
const mod = require(path.resolve(__dirname, '../src/agrifed_khaos/static/khaos-fl-pwa.js'));
const { EdgeOptimizer } = mod._internals;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

const D = 32;
const classes = ['a', 'b', 'c', 'd'];
const r = rng(7);
const baseW = classes.map(() => Array.from({ length: D }, () => (r() - 0.5) * 0.2));
const baseB = classes.map(() => 0.0);
const base = { weight: baseW, bias: baseB };

// Local data only for classes 'a' (0) and 'b' (1); embeddings are non-negative like avg-pooled features.
function sample(label) {
  const emb = Array.from({ length: D }, (_, d) => Math.abs((r() - 0.5) * 0.5) + (d % 2 === label ? 1.0 : 0.0));
  return { embedding: emb, label };
}
const dataset = [];
for (let i = 0; i < 24; i++) dataset.push(sample(i % 2));

const common = { dataset, baseWeights: base, canonicalClasses: classes, activeClasses: classes, embeddingDim: D, epochs: 5, noiseSigma: 0.0 };

function relDelta(w, b) {
  let d = 0, n = 0;
  for (let c = 0; c < classes.length; c++) {
    d += (b[c] - baseB[c]) ** 2;
    n += baseB[c] ** 2;
    for (let k = 0; k < D; k++) { d += (w[c][k] - baseW[c][k]) ** 2; n += baseW[c][k] ** 2; }
  }
  return Math.sqrt(d) / Math.sqrt(n);
}

(async () => {
  const out = {};

  // 1) Missing classes ('c','d') must stay exactly at base values; present classes must move.
  const r1 = await EdgeOptimizer.trainHead({ ...common });
  out.missing_unchanged = [2, 3].every((c) => r1.weights.weight[c].every((v, k) => Math.abs(v - baseW[c][k]) < 1e-7) && Math.abs(r1.weights.bias[c]) < 1e-7);
  out.present_moved = r1.weights.weight[0].some((v, k) => Math.abs(v - baseW[0][k]) > 1e-4);
  out.all_finite = r1.weights.weight.flat().every(Number.isFinite) && r1.weights.bias.every(Number.isFinite);
  out.accuracy = r1.stats.accuracy;

  // 2) FedProx: larger mu must pull weights closer to the global anchor.
  const lo = await EdgeOptimizer.trainHead({ ...common, fedproxMu: 0.0, epochs: 20, maxRelativeDelta: 100 });
  const hi = await EdgeOptimizer.trainHead({ ...common, fedproxMu: 5.0, epochs: 20, maxRelativeDelta: 100 });
  out.rel_mu0 = relDelta(lo.weights.weight, lo.weights.bias);
  out.rel_mu5 = relDelta(hi.weights.weight, hi.weights.bias);

  // 3) Delta clipping: relative update never exceeds the configured bound (noise disabled).
  const clipped = await EdgeOptimizer.trainHead({ ...common, fedproxMu: 0.0, epochs: 60, lr: 0.05, maxRelativeDelta: 0.1 });
  out.rel_clipped = relDelta(clipped.weights.weight, clipped.weights.bias);

  // 4) Warm start: a second session on class 'b' only must retain what session 1 learned on 'a'.
  const sess1 = await EdgeOptimizer.trainHead({ ...common, maxRelativeDelta: 100, dataset: dataset.filter((s) => s.label === 0), epochs: 10 });
  const sess2 = await EdgeOptimizer.trainHead({
    ...common,
    dataset: dataset.filter((s) => s.label === 1),
    epochs: 10,
    maxRelativeDelta: 100,
    initWeights: { weight: sess1.weights.weight, bias: sess1.weights.bias },
  });
  out.warm_start_retains_a = sess2.weights.weight[0].every((v, k) => Math.abs(v - sess1.weights.weight[0][k]) < 1e-7);
  const cold = await EdgeOptimizer.trainHead({ ...common, dataset: dataset.filter((s) => s.label === 1), epochs: 10 });
  out.cold_start_resets_a = cold.weights.weight[0].every((v, k) => Math.abs(v - baseW[0][k]) < 1e-7);

  console.log(JSON.stringify(out));
})();
