//! WASM-specific entry points for browser / edge deployments.
//!
//! Compiled only when targeting `wasm32-unknown-unknown`.
//! Exposes core compute primitives that are useful from JavaScript:
//!   - `cosine_similarity(a, b) -> f32`
//!   - `quantize_batch(data, dim) -> Vec<u8>`  (scalar INT8-style quantization)
//!
//! Build with:
//!   `wasm-pack build --target web vectro_lib`
//!   or: `wasm-pack build --target bundler vectro_lib`

use wasm_bindgen::prelude::*;

/// Compute cosine similarity between two equal-length f32 slices.
///
/// Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector is zero.
///
/// # Panics (JS `Error`)
/// Panics at the JS boundary if `a.length !== b.length`.
#[wasm_bindgen]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        wasm_bindgen::throw_str("cosine_similarity: vectors must have equal length");
    }
    if a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Scalar-quantize a flat batch of f32 vectors to u8.
///
/// `data`  – flat array of `n_vectors * dim` f32 values (row-major).
/// `dim`   – dimensionality of each vector. `data.length` must be a multiple of `dim`.
///
/// Each dimension is scaled independently using per-dimension min/max derived from
/// the entire batch. The output is `n_vectors * dim` bytes.
///
/// Returns an empty `Uint8Array` if `data` is empty or `dim == 0`.
#[wasm_bindgen]
pub fn quantize_batch(data: &[f32], dim: usize) -> Vec<u8> {
    if data.is_empty() || dim == 0 || data.len() % dim != 0 {
        return Vec::new();
    }
    let n = data.len() / dim;

    // Compute per-dimension min/max over the entire batch.
    let mut min_vals = vec![f32::INFINITY; dim];
    let mut max_vals = vec![f32::NEG_INFINITY; dim];
    for row in 0..n {
        for d in 0..dim {
            let v = data[row * dim + d];
            if v < min_vals[d] {
                min_vals[d] = v;
            }
            if v > max_vals[d] {
                max_vals[d] = v;
            }
        }
    }

    let mut out = vec![0u8; data.len()];
    for row in 0..n {
        for d in 0..dim {
            let v = data[row * dim + d];
            let lo = min_vals[d];
            let hi = max_vals[d];
            let q = if (hi - lo).abs() < 1e-12 {
                128u8
            } else {
                ((v - lo) / (hi - lo) * 255.0).round().clamp(0.0, 255.0) as u8
            };
            out[row * dim + d] = q;
        }
    }
    out
}
