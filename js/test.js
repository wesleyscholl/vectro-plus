/**
 * Minimal Node.js smoke-test for the vectro-plus WASM entry points.
 *
 * Usage (after `wasm-pack build --target nodejs vectro_lib`):
 *   node js/test.js
 *
 * Exit codes:
 *   0  all assertions pass
 *   1  one or more assertions failed
 */

"use strict";

const path = require("path");
const assert = require("assert").strict;

// Adjust the path to match the wasm-pack output directory.
const WASM_PKG = path.join(__dirname, "..", "vectro_lib", "pkg");

let vectroLib;
try {
  vectroLib = require(WASM_PKG);
} catch (err) {
  console.error(
    `[SKIP] Could not load WASM package from ${WASM_PKG}\n` +
      `       Run: wasm-pack build --target nodejs vectro_lib\n` +
      `       Error: ${err.message}`
  );
  // Exit 0 — this is a skip, not a hard failure, when the package hasn't been built.
  process.exit(0);
}

const { cosine_similarity, quantize_batch } = vectroLib;

let failures = 0;

function check(label, fn) {
  try {
    fn();
    console.log(`  ✓ ${label}`);
  } catch (err) {
    console.error(`  ✗ ${label}: ${err.message}`);
    failures++;
  }
}

console.log("vectro-plus WASM smoke tests");
console.log("─".repeat(50));

// ── cosine_similarity ────────────────────────────────────────────────────

check("cosine_similarity: unit vector with itself = 1.0", () => {
  const a = new Float32Array([1, 0, 0]);
  const result = cosine_similarity(a, a);
  assert(
    Math.abs(result - 1.0) < 1e-6,
    `expected ≈1.0, got ${result}`
  );
});

check("cosine_similarity: orthogonal vectors = 0.0", () => {
  const a = new Float32Array([1, 0, 0]);
  const b = new Float32Array([0, 1, 0]);
  const result = cosine_similarity(a, b);
  assert(
    Math.abs(result) < 1e-6,
    `expected ≈0.0, got ${result}`
  );
});

check("cosine_similarity: opposite vectors = -1.0", () => {
  const a = new Float32Array([0, 0, 1]);
  const b = new Float32Array([0, 0, -1]);
  const result = cosine_similarity(a, b);
  assert(
    Math.abs(result - (-1.0)) < 1e-6,
    `expected ≈-1.0, got ${result}`
  );
});

check("cosine_similarity: zero vector returns 0.0 (no NaN/Inf)", () => {
  const z = new Float32Array([0, 0, 0]);
  const a = new Float32Array([1, 2, 3]);
  const result = cosine_similarity(z, a);
  assert(
    Number.isFinite(result) && result === 0.0,
    `expected 0.0, got ${result}`
  );
});

check("cosine_similarity: general similarity in range [-1, 1]", () => {
  const a = new Float32Array([0.2, 0.5, -0.1, 0.8]);
  const b = new Float32Array([-0.3, 0.4, 0.9, 0.1]);
  const result = cosine_similarity(a, b);
  assert(result >= -1.0 && result <= 1.0, `got ${result} outside [-1, 1]`);
});

// ── quantize_batch ───────────────────────────────────────────────────────

check("quantize_batch: output length == input length", () => {
  const data = new Float32Array([0, 1, 2, 3, 4, 5]);
  const out = quantize_batch(data, 3 /* dim */);
  assert.equal(out.length, data.length, `expected ${data.length}, got ${out.length}`);
});

check("quantize_batch: each byte is in [0, 255]", () => {
  const N = 5;
  const DIM = 4;
  const data = new Float32Array(N * DIM);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() - 0.5) * 10;
  }
  const out = quantize_batch(data, DIM);
  for (let i = 0; i < out.length; i++) {
    assert(out[i] >= 0 && out[i] <= 255, `byte ${i} = ${out[i]}`);
  }
});

check("quantize_batch: min dimension maps to 0, max maps to 255", () => {
  // Single 1-d 'vector' batch: two values [0.0, 1.0] → quantized [0, 255]
  const data = new Float32Array([0.0, 1.0]);
  const out = quantize_batch(data, 1 /* dim=1, n=2 rows */);
  assert.equal(out[0], 0, `min should be 0, got ${out[0]}`);
  assert.equal(out[1], 255, `max should be 255, got ${out[1]}`);
});

check("quantize_batch: empty input returns empty array", () => {
  const out = quantize_batch(new Float32Array(0), 3);
  assert.equal(out.length, 0);
});

check("quantize_batch: dim=0 returns empty array", () => {
  const out = quantize_batch(new Float32Array([1, 2, 3]), 0);
  assert.equal(out.length, 0);
});

// ── Summary ──────────────────────────────────────────────────────────────

console.log("─".repeat(50));
if (failures === 0) {
  console.log("All tests passed.");
  process.exit(0);
} else {
  console.error(`${failures} test(s) FAILED.`);
  process.exit(1);
}
