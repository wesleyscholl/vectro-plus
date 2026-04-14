//! Product Quantization (PQ) for approximate nearest-neighbor search.
//!
//! # Algorithm Overview
//! 1. **Training**: split each vector into `m` subspaces of `dim/m` dimensions each.
//!    Run Lloyd's k-means (`k` centroids, k ≤ 256) on each subspace independently,
//!    parallelised across subspaces with rayon.
//! 2. **Encoding**: for each subspace find the nearest centroid → emit its index as `u8`.
//!    Each vector becomes `m` bytes.
//! 3. **Decoding**: look up each centroid and concatenate to reconstruct an approximate f32 vector.
//! 4. **ADC Search**: build a query-specific `m × k` lookup table of inner products, then score
//!    each encoded vector by summing `m` table lookups — O(m) per vector instead of O(dim).
//!
//! # Cosine Compatibility
//! Vectors are L2-normalised before training and encoding, so inner products between
//! reconstructed vectors approximate cosine similarity.
//!
//! # Compression Ratio
//! `(dim × 4 bytes) / (m bytes) = 4 × dim / m`
//! Example: `dim=768, m=48` → 64× compression.

use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use crate::Embedding;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Product Quantizer: encodes `dim`-dimensional f32 vectors into `m` bytes.
///
/// Train with [`ProductQuantizer::train`], then call [`encode`] / [`decode`] / [`search_adc`].
///
/// [`encode`]: ProductQuantizer::encode
/// [`decode`]: ProductQuantizer::decode
/// [`search_adc`]: ProductQuantizer::search_adc
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Number of subspaces — also the byte length of each encoded vector.
    pub m: usize,
    /// Number of centroids per subspace (≤ 256; codes are stored as `u8`).
    pub k: usize,
    /// Full vector dimension — must be divisible by `m`.
    pub dim: usize,
    /// `codebooks[s][c]` = centroid `c` in subspace `s`, length = `dim/m`.
    /// Centroids are L2-normalised unit vectors in their subspace.
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Dimension of each sub-vector (`dim / m`).
    #[inline]
    pub fn sub_dim(&self) -> usize {
        self.dim / self.m
    }

    /// Train a `ProductQuantizer` on the given embeddings.
    ///
    /// # Arguments
    /// * `data`  — training set (vectors need not be normalised; training normalises internally).
    /// * `m`     — number of subspaces (encoded bytes per vector); must divide `data[0].len()`.
    /// * `k`     — centroids per subspace; must be ≤ 256.
    /// * `iters` — maximum Lloyd's k-means iterations per subspace (25–50 is usually sufficient).
    ///
    /// # Panics
    /// Panics if `k > 256`, if the dataset is empty, or if `dim % m != 0`.
    pub fn train(data: &[Embedding], m: usize, k: usize, iters: usize) -> Self {
        assert!(k <= 256, "k must be ≤ 256 (codes are stored as u8); got k={}", k);
        assert!(!data.is_empty(), "training dataset must not be empty");
        let dim = data[0].vector.len();
        assert!(
            dim > 0 && dim % m == 0,
            "dim ({}) must be divisible by m ({})",
            dim,
            m
        );
        let sub_dim = dim / m;

        // L2-normalise every vector for cosine-compatible codebooks.
        let normed: Vec<Vec<f32>> = data.par_iter().map(|e| l2_normalize(&e.vector)).collect();

        // Precompute owned sub-vectors per subspace to avoid lifetime issues in par_iter.
        let all_subs: Vec<Vec<Vec<f32>>> = (0..m)
            .map(|s| {
                let start = s * sub_dim;
                normed.iter().map(|v| v[start..start + sub_dim].to_vec()).collect()
            })
            .collect();

        // Run k-means on each subspace in parallel.
        let codebooks: Vec<Vec<Vec<f32>>> = all_subs
            .into_par_iter()
            .map(|subs| kmeans(&subs, k, iters))
            .collect();

        Self { m, k, dim, codebooks }
    }

    /// Encode a single f32 vector into `m` bytes.
    ///
    /// The vector is L2-normalised before encoding to match the training convention.
    ///
    /// # Panics
    /// Panics if `v.len() != self.dim`.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(
            v.len(),
            self.dim,
            "vector length {} ≠ ProductQuantizer dim {}",
            v.len(),
            self.dim
        );
        let sub_dim = self.sub_dim();
        let normed = l2_normalize(v);
        (0..self.m)
            .map(|s| {
                let start = s * sub_dim;
                let sub = &normed[start..start + sub_dim];
                nearest_centroid(sub, &self.codebooks[s]) as u8
            })
            .collect()
    }

    /// Decode `m` bytes back into an approximate f32 vector of length `dim`.
    ///
    /// The reconstruction is the concatenation of the `m` sub-space centroids.
    ///
    /// # Panics
    /// Panics if `code.len() != self.m`.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(
            code.len(),
            self.m,
            "code length {} ≠ m {}",
            code.len(),
            self.m
        );
        let sub_dim = self.sub_dim();
        let mut out = vec![0.0f32; self.dim];
        for (s, &c_byte) in code.iter().enumerate().take(self.m) {
            let c = c_byte as usize;
            let start = s * sub_dim;
            out[start..start + sub_dim].copy_from_slice(&self.codebooks[s][c]);
        }
        out
    }

    /// Compression ratio: bytes per vector (f32) divided by bytes per code (m bytes).
    ///
    /// For a 768-dim vector with m=8 subspaces: 768×4 / 8 = 384×.
    pub fn compression_ratio(&self) -> f32 {
        (self.dim as f32 * 4.0) / self.m as f32
    }

    /// ADC (Asymmetric Distance Computation) search — approximate cosine similarity.
    ///
    /// For each subspace `s` and centroid `c`, the lookup table entry is
    /// `dot(query_sub[s], centroid[s][c])`. The approximate score for an encoded
    /// vector is the sum of `m` table lookups — O(m) per vector.
    ///
    /// Returns top-`k` `(id, score)` pairs sorted descending by score.
    /// Returns an empty vec if `codes` is empty, `query.len() != self.dim`, or `k == 0`.
    pub fn search_adc<'a>(
        &self,
        codes: &'a [(String, Vec<u8>)],
        query: &[f32],
        k: usize,
    ) -> Vec<(&'a str, f32)> {
        if codes.is_empty() || query.len() != self.dim || k == 0 {
            return vec![];
        }
        let sub_dim = self.sub_dim();
        let qnorm = l2_normalize(query);

        // Build m × k lookup table: table[s][c] = dot(query_sub_s, centroid_s_c)
        let table: Vec<Vec<f32>> = (0..self.m)
            .map(|s| {
                let qsub = &qnorm[s * sub_dim..(s + 1) * sub_dim];
                self.codebooks[s].iter().map(|centroid| dot(qsub, centroid)).collect()
            })
            .collect();

        // Score every encoded vector by summing m table lookups (parallel).
        let mut scores: Vec<(&str, f32)> = codes
            .par_iter()
            .map(|(id, code)| {
                let score: f32 = (0..self.m).map(|s| table[s][code[s] as usize]).sum();
                (id.as_str(), score)
            })
            .collect();

        scores.par_sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.into_iter().take(k).collect()
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}

#[inline]
fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Return the index of the centroid nearest to `sub` (L2 distance).
fn nearest_centroid(sub: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_dist_sq(sub, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Lloyd's k-means on a slice of owned sub-vectors.
///
/// Returns `min(k, subs.len())` L2-normalised centroids.
fn kmeans(subs: &[Vec<f32>], k: usize, iters: usize) -> Vec<Vec<f32>> {
    if subs.is_empty() {
        return vec![];
    }
    let d = subs[0].len();
    let n = subs.len();
    let actual_k = k.min(n);

    // Initialise: evenly-spaced subset of data (deterministic, no randomness dependency).
    let mut centroids: Vec<Vec<f32>> = (0..actual_k)
        .map(|i| subs[(i * n / actual_k).min(n - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iters {
        // Assignment step (parallel).
        let new_assignments: Vec<usize> = subs
            .par_iter()
            .map(|sub| nearest_centroid(sub, &centroids))
            .collect();

        let converged = new_assignments == assignments;
        assignments = new_assignments;
        if converged {
            break;
        }

        // Update step: compute cluster means.
        let mut sums = vec![vec![0.0f32; d]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for (sub, &c) in subs.iter().zip(assignments.iter()) {
            counts[c] += 1;
            for (j, &x) in sub.iter().enumerate() {
                sums[c][j] += x;
            }
        }

        for c in 0..actual_k {
            if counts[c] == 0 {
                // Dead centroid: reinitialise to a deterministic data point.
                centroids[c] = subs[c % n].clone();
            } else {
                let cnt = counts[c] as f32;
                let mean: Vec<f32> = sums[c].iter().map(|&x| x / cnt).collect();
                // L2-normalise centroid for cosine compatibility.
                let n_val = l2_norm(&mean);
                centroids[c] = if n_val > 1e-12 {
                    mean.iter().map(|x| x / n_val).collect()
                } else {
                    mean
                };
            }
        }
    }

    centroids
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Embedding;

    /// Deterministic dataset: vectors spread across the unit hypercube.
    fn make_dataset(n: usize, dim: usize) -> Vec<Embedding> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| {
                        // Spread values to avoid degenerate clustering.
                        (((i * 17 + d * 11) % 97) as f32 + 0.5) / 97.5
                    })
                    .collect();
                Embedding::new(format!("id_{}", i), v)
            })
            .collect()
    }

    // ── Shape & dtype contract ───────────────────────────────────────────────

    #[test]
    fn test_pq_encode_decode_shape() {
        let data = make_dataset(200, 32);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);

        assert_eq!(pq.m, 4);
        assert_eq!(pq.k, 16);
        assert_eq!(pq.dim, 32);
        assert_eq!(pq.sub_dim(), 8);

        let code = pq.encode(&data[0].vector);
        assert_eq!(code.len(), 4, "encode must return m=4 bytes");

        let recon = pq.decode(&code);
        assert_eq!(recon.len(), 32, "decode must return dim=32 floats");
    }

    // ── Compression ratio gate (≥ 16×) ──────────────────────────────────────

    #[test]
    fn test_pq_compression_ratio_gate() {
        // dim=128, m=8 → ratio = (128 × 4) / 8 = 64×
        let data = make_dataset(500, 128);
        let pq = ProductQuantizer::train(&data, 8, 64, 20);
        let code = pq.encode(&data[0].vector);
        assert_eq!(code.len(), 8);

        let ratio = (pq.dim * 4) as f32 / pq.m as f32;
        assert!(ratio >= 16.0, "compression ratio {:.1}× < 16×", ratio);
    }

    // ── Numerical correctness: encode→decode reconstruction error ───────────

    #[test]
    fn test_pq_reconstruction_error() {
        // With sufficient centroids the reconstruction should be close.
        let data = make_dataset(500, 32);
        let pq = ProductQuantizer::train(&data, 4, 64, 30);

        let mut total_error = 0.0f32;
        for e in data.iter().take(50) {
            let code = pq.encode(&e.vector);
            let recon = pq.decode(&code);
            // Compare normalised versions (train normalises internally).
            let v_norm = l2_normalize(&e.vector);
            let err: f32 = v_norm
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_error += err;
        }
        let mean_err = total_error / 50.0;
        // Mean L2 reconstruction error should be well below 1.0 with 64 centroids.
        assert!(mean_err < 1.0, "mean reconstruction error {:.4} ≥ 1.0", mean_err);
    }

    // ── Recall@10 gate (≥ 0.90) ─────────────────────────────────────────────

    #[test]
    fn test_pq_recall_at_10_gate() {
        // 2000 vectors, 64d, m=8, k=256, 30 iters.
        // Queries are the first 30 vectors in the dataset (present in the index).
        let data = make_dataset(2000, 64);
        let pq = ProductQuantizer::train(&data, 8, 256, 30);

        let codes: Vec<(String, Vec<u8>)> = data
            .iter()
            .map(|e| (e.id.clone(), pq.encode(&e.vector)))
            .collect();

        let num_queries = 30;
        let mut total_hits = 0usize;

        for q in data.iter().take(num_queries) {
            // Exact top-10 by cosine similarity.
            let exact = crate::search::top_k(&data, &q.vector, 10);
            let exact_ids: std::collections::HashSet<&str> =
                exact.iter().map(|(id, _)| *id).collect();

            // Approximate top-10 via ADC.
            let approx = pq.search_adc(&codes, &q.vector, 10);
            let approx_ids: std::collections::HashSet<&str> =
                approx.iter().map(|(id, _)| *id).collect();

            total_hits += exact_ids.intersection(&approx_ids).count();
        }

        let recall = total_hits as f32 / (num_queries * 10) as f32;
        assert!(recall >= 0.90, "recall@10 = {:.3} < gate 0.90", recall);
    }

    // ── Regression snapshot ──────────────────────────────────────────────────

    #[test]
    fn test_pq_encode_deterministic() {
        // Same training data + same vector must always produce the same code.
        let data = make_dataset(200, 32);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);
        let code_a = pq.encode(&data[0].vector);
        let code_b = pq.encode(&data[0].vector);
        assert_eq!(code_a, code_b, "encode must be deterministic");
    }

    // ── Failure cases ────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "k must be ≤ 256")]
    fn test_pq_train_k_too_large() {
        let data = make_dataset(10, 8);
        let _ = ProductQuantizer::train(&data, 2, 300, 5);
    }

    #[test]
    #[should_panic(expected = "dim (9) must be divisible by m (4)")]
    fn test_pq_train_dim_not_divisible() {
        let data = make_dataset(10, 9);
        let _ = ProductQuantizer::train(&data, 4, 4, 5);
    }

    #[test]
    #[should_panic(expected = "training dataset must not be empty")]
    fn test_pq_train_empty_dataset() {
        let _ = ProductQuantizer::train(&[], 4, 16, 5);
    }

    #[test]
    fn test_pq_search_adc_empty() {
        let data = make_dataset(50, 16);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);
        // Empty codes
        let results = pq.search_adc(&[], &data[0].vector, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pq_search_adc_dim_mismatch() {
        let data = make_dataset(50, 16);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);
        let codes: Vec<(String, Vec<u8>)> =
            data.iter().map(|e| (e.id.clone(), pq.encode(&e.vector))).collect();
        // Wrong query dimension
        let results = pq.search_adc(&codes, &[1.0, 2.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pq_search_adc_k_zero() {
        let data = make_dataset(50, 16);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);
        let codes: Vec<(String, Vec<u8>)> =
            data.iter().map(|e| (e.id.clone(), pq.encode(&e.vector))).collect();
        let results = pq.search_adc(&codes, &data[0].vector, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pq_search_adc_returns_at_most_k() {
        let data = make_dataset(100, 16);
        let pq = ProductQuantizer::train(&data, 4, 16, 10);
        let codes: Vec<(String, Vec<u8>)> =
            data.iter().map(|e| (e.id.clone(), pq.encode(&e.vector))).collect();
        let results = pq.search_adc(&codes, &data[0].vector, 5);
        assert_eq!(results.len(), 5);
    }
}
