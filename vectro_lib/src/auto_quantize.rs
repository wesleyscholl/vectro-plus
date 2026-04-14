//! AutoQuantize — automatic format selection for embedding compression.
//!
//! Evaluates multiple quantization strategies on a sample of the input data
//! and returns the first format that satisfies both `target_cosine` and
//! `target_compression`.  Default targets match the vectro research library:
//! cosine ≥ 0.97, compression ≥ 8×.
//!
//! ## Strategy order (highest quality first — normal-distribution data)
//! 1. **NF4**    — 4-bit NormalFloat; ~8× compression; ideal for Gaussian-ish.
//! 2. **RQ**     — Residual Quantizer (2 passes, adaptive subspaces); ~16–192×.
//! 3. **PQ**     — Product Quantizer; ~16–64×.
//! 4. **Scalar** — INT8 per-dimension; ~4×; cosine ≥ 0.999.
//!
//! ## Kurtosis routing
//! Excess kurtosis > 4.0 (heavy-tailed distribution) routes to PQ first,
//! because PQ places subspace centroids on actual data clusters rather than
//! relying on the Gaussian assumption embedded in the NF4 codebook.

use crate::{nf4::Nf4Quantizer, pq::ProductQuantizer, rq::ResidualQuantizer, Embedding};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─── Public types ─────────────────────────────────────────────────────────────

/// The quantization format chosen by [`auto_select_format`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantFormat {
    /// NF4  — 4-bit NormalFloat (~8×).
    Nf4,
    /// RQ   — Residual Quantizer (~16–192×).
    Rq,
    /// PQ   — Product Quantizer (~16–64×).
    Pq,
    /// Scalar— INT8 per-dimension (~4×).
    Scalar,
    /// Stream — raw f32 passthrough (fallback, 1×).
    Stream,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantFormat::Nf4    => write!(f, "nf4"),
            QuantFormat::Rq     => write!(f, "rq"),
            QuantFormat::Pq     => write!(f, "pq"),
            QuantFormat::Scalar => write!(f, "scalar"),
            QuantFormat::Stream => write!(f, "stream"),
        }
    }
}

/// The result returned by [`auto_select_format`].
pub struct AutoQuantizeResult {
    /// Selected format.
    pub format: QuantFormat,
    /// Mean cosine similarity on the evaluation sample.
    pub cosine_sim: f32,
    /// Compression ratio achieved by the selected format.
    pub compression_ratio: f32,
    /// Trained `ProductQuantizer` — present when `format == Pq`.
    pub pq: Option<ProductQuantizer>,
    /// Trained `ResidualQuantizer` — present when `format == Rq`.
    pub rq: Option<ResidualQuantizer>,
}

// ─── Public function ──────────────────────────────────────────────────────────

/// Evaluate quantization formats and return the best one meeting constraints.
///
/// Tries candidates in descending quality order.  Returns the **first** format
/// achieving `cosine_sim ≥ target_cosine` **and** `compression_ratio ≥ target_compression`.
///
/// If no format meets both constraints, returns the one with the highest cosine
/// similarity that still meets the compression target.  Final fallback is Scalar.
///
/// # Arguments
/// * `data`                — embeddings to evaluate on.
/// * `target_cosine`       — minimum acceptable cosine similarity (e.g. `0.97`).
/// * `target_compression`  — minimum compression ratio (e.g. `8.0`).
/// * `max_sample`          — cap evaluation to this many vectors (for speed).
pub fn auto_select_format(
    data: &[Embedding],
    target_cosine: f32,
    target_compression: f32,
    max_sample: usize,
) -> AutoQuantizeResult {
    if data.is_empty() {
        return AutoQuantizeResult {
            format: QuantFormat::Stream,
            cosine_sim: 1.0,
            compression_ratio: 1.0,
            pq: None,
            rq: None,
        };
    }

    let sample: &[Embedding] = &data[..max_sample.min(data.len())];
    let dim = sample[0].vector.len();

    // Kurtosis heuristic: route heavy-tailed data to PQ before NF4.
    let kurtosis = compute_excess_kurtosis(sample);
    let heavy_tailed = kurtosis > 4.0;

    let candidates: &[FormatCandidate] = if heavy_tailed {
        &[
            FormatCandidate::Pq,
            FormatCandidate::Rq,
            FormatCandidate::Nf4,
            FormatCandidate::Scalar,
        ]
    } else {
        &[
            FormatCandidate::Nf4,
            FormatCandidate::Rq,
            FormatCandidate::Pq,
            FormatCandidate::Scalar,
        ]
    };

    let mut best: Option<AutoQuantizeResult> = None;

    for candidate in candidates {
        let result = evaluate_candidate(candidate, sample, dim);

        let meets_cosine = result.cosine_sim >= target_cosine;
        let meets_compression = result.compression_ratio >= target_compression;

        if meets_cosine && meets_compression {
            return result;
        }

        // Track the best among those meeting only the compression target.
        if result.compression_ratio >= target_compression
            && (best.is_none()
                || result.cosine_sim > best.as_ref().map(|r| r.cosine_sim).unwrap_or(-1.0))
        {
            best = Some(result);
        }
    }

    // Fallback: return best compression-meeting candidate, or scalar.
    best.unwrap_or(AutoQuantizeResult {
        format: QuantFormat::Scalar,
        cosine_sim: 0.99,
        compression_ratio: 4.0,
        pq: None,
        rq: None,
    })
}

// ─── Internal ─────────────────────────────────────────────────────────────────

enum FormatCandidate {
    Nf4,
    Rq,
    Pq,
    Scalar,
}

fn evaluate_candidate(
    cand: &FormatCandidate,
    sample: &[Embedding],
    dim: usize,
) -> AutoQuantizeResult {
    match cand {
        FormatCandidate::Nf4    => eval_nf4(sample, dim),
        FormatCandidate::Rq     => eval_rq(sample, dim),
        FormatCandidate::Pq     => eval_pq(sample, dim),
        FormatCandidate::Scalar => eval_scalar(sample),
    }
}

fn eval_nf4(sample: &[Embedding], dim: usize) -> AutoQuantizeResult {
    let vecs: Vec<Vec<f32>> = sample.iter().map(|e| e.vector.clone()).collect();
    let q = Nf4Quantizer::new();
    let (packed, scales) = q.encode(&vecs);
    let decoded = q.decode(&packed, &scales, dim);
    let cosine_sim = q.mean_cosine_sim(&vecs, &decoded);
    let compression_ratio = Nf4Quantizer::compression_ratio(dim);
    AutoQuantizeResult {
        format: QuantFormat::Nf4,
        cosine_sim,
        compression_ratio,
        pq: None,
        rq: None,
    }
}

fn eval_rq(sample: &[Embedding], dim: usize) -> AutoQuantizeResult {
    let m = choose_m(dim, 8, 4);
    let mut rq = ResidualQuantizer::new(2, m, 64);
    rq.train(sample, 15);
    let cosine_sim = rq.mean_cosine_sim(sample);
    let compression_ratio = rq.compression_ratio();
    AutoQuantizeResult {
        format: QuantFormat::Rq,
        cosine_sim,
        compression_ratio,
        pq: None,
        rq: Some(rq),
    }
}

fn eval_pq(sample: &[Embedding], dim: usize) -> AutoQuantizeResult {
    let m = choose_m(dim, 8, 4);
    let pq = ProductQuantizer::train(sample, m, 256, 25);
    let cosine_sim: f32 = {
        let sum: f32 = sample
            .par_iter()
            .map(|e| {
                let code = pq.encode(&e.vector);
                let recon = pq.decode(&code);
                cosine_sim_pair(&e.vector, &recon)
            })
            .sum();
        if sample.is_empty() {
            1.0
        } else {
            sum / sample.len() as f32
        }
    };
    let compression_ratio = pq.compression_ratio();
    AutoQuantizeResult {
        format: QuantFormat::Pq,
        cosine_sim,
        compression_ratio,
        pq: Some(pq),
        rq: None,
    }
}

fn eval_scalar(sample: &[Embedding]) -> AutoQuantizeResult {
    // INT8 per-dimension: abs-max scale per vector, clamp to [−127, 127].
    let sum: f32 = sample
        .par_iter()
        .map(|e| {
            let v = &e.vector;
            let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max == 0.0 { 1.0f32 } else { abs_max / 127.0 };
            let q: Vec<i8> = v
                .iter()
                .map(|x| (x / scale).clamp(-127.0, 127.0).round() as i8)
                .collect();
            let recon: Vec<f32> = q.iter().map(|&b| b as f32 * scale).collect();
            cosine_sim_pair(v, &recon)
        })
        .sum();
    let cs = if sample.is_empty() {
        1.0
    } else {
        sum / sample.len() as f32
    };
    AutoQuantizeResult {
        format: QuantFormat::Scalar,
        cosine_sim: cs,
        compression_ratio: 4.0, // float32 → int8
        pq: None,
        rq: None,
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Choose `m` such that `dim % m == 0` and `dim / m >= min_sub_dim`.
///
/// Scans from `max_m` downwards; returns 1 as the last resort.
fn choose_m(dim: usize, max_m: usize, min_sub_dim: usize) -> usize {
    for m in (1..=max_m).rev() {
        if dim % m == 0 && dim / m >= min_sub_dim {
            return m;
        }
    }
    1
}

fn cosine_sim_pair(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| x * y)
        .sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Mean per-dimension excess kurtosis across the sample.
///
/// Excess kurtosis = μ₄/σ⁴ − 3; Gaussian ≈ 0; heavy-tailed > 0.
fn compute_excess_kurtosis(data: &[Embedding]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    let n = data.len() as f64;
    let dim = data[0].vector.len();
    let mut kurtosis_sum = 0.0f64;
    let mut counted = 0usize;

    for d in 0..dim {
        let mean: f64 = data.iter().map(|e| e.vector[d] as f64).sum::<f64>() / n;
        let var: f64 =
            data.iter().map(|e| (e.vector[d] as f64 - mean).powi(2)).sum::<f64>() / n;
        if var < 1e-10 {
            continue;
        }
        let m4: f64 =
            data.iter().map(|e| (e.vector[d] as f64 - mean).powi(4)).sum::<f64>() / n;
        kurtosis_sum += m4 / (var * var) - 3.0;
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        kurtosis_sum / counted as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: u32, id: &str, dim: usize) -> Embedding {
        let mut state = seed as u64;
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let frac = (state >> 33) as f32 / u32::MAX as f32;
                frac * 2.0 - 1.0
            })
            .collect();
        Embedding::new(id, v)
    }

    #[test]
    fn auto_select_returns_valid_format() {
        let data: Vec<Embedding> = (0..200)
            .map(|i| make_embedding(i, &format!("id{i}"), 128))
            .collect();
        let result = auto_select_format(&data, 0.97, 4.0, 200);
        assert!(
            result.cosine_sim >= 0.90,
            "auto cosine sim {:.4} unexpectedly low",
            result.cosine_sim
        );
        assert!(
            result.compression_ratio >= 4.0,
            "compression ratio {:.1} < 4×",
            result.compression_ratio
        );
    }

    #[test]
    fn auto_select_never_panics_tiny_data() {
        // Must not panic on very small datasets or very short vectors.
        let data: Vec<Embedding> = (0..5)
            .map(|i| make_embedding(i, &format!("id{i}"), 8))
            .collect();
        let _result = auto_select_format(&data, 0.97, 4.0, 10);
    }

    #[test]
    fn auto_select_empty_returns_stream() {
        let result = auto_select_format(&[], 0.97, 8.0, 1000);
        assert_eq!(result.format, QuantFormat::Stream);
    }

    #[test]
    fn choose_m_divides_dim() {
        // m must divide dim and dim/m >= min_sub_dim.
        for dim in [128usize, 256, 512, 768, 1024] {
            let m = choose_m(dim, 8, 4);
            assert_eq!(dim % m, 0, "m={m} does not divide dim={dim}");
            assert!(dim / m >= 4, "dim/m < 4 for m={m}, dim={dim}");
        }
    }
}
