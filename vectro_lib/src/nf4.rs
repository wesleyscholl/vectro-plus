//! NF4 (NormalFloat 4-bit) quantization — Vectro+ Research Port.
//!
//! Implements the NF4 encoding scheme from Dettmers et al. 2023 (QLoRA).
//! Each f32 value is mapped to the nearest of 16 quantile levels drawn
//! from N(0,1), rescaled to [−1, +1], then packed 2-per-byte with a
//! per-vector abs-max scale factor.
//!
//! # Compression
//! `dim × 4 bytes → ceil(dim/2) + 4 bytes` ≈ **7.9× compression**.
//!
//! # Accuracy
//! Cosine similarity ≥ 0.98 for Gaussian-ish embedding distributions
//! (typical of Transformer hidden states and sentence embeddings).
//!
//! # Binary format (`VECTRO+NF4STREAM1\n`)
//! ```text
//! [header]  bytes: b"VECTRO+NF4STREAM1\n"
//! [u32 LE]  dim   — original vector dimension
//! [u32 LE]  count — number of records that follow
//! [repeat]  u32-LE len + bincode((id: String, packed: Vec<u8>, scale: f32))
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─── NF4 codebook ─────────────────────────────────────────────────────────────

/// The 16 NF4 quantile levels (Dettmers et al. 2023 — QLoRA, Table 1).
///
/// These are the quantiles of the standard normal distribution N(0,1) mapped
/// symmetrically onto [−1, +1].  Level 7 (index 7) is exactly 0.0.
pub const NF4_LEVELS: [f32; 16] = [
    -1.000_000_0,
    -0.696_192_8,
    -0.525_073,
    -0.394_900_3,
    -0.284_467_7,
    -0.184_874_5,
    -0.091_050_04,
    0.000_000_0,
    0.079_580_31,
    0.160_939_08,
    0.246_114_96,
    0.337_915_24,
    0.440_709_83,
    0.562_667_55,
    0.722_957_6,
    1.000_000_0,
];

/// Midpoint thresholds between adjacent NF4 levels (for nearest-neighbour lookup).
///
/// `THRESHOLDS[i]` = (NF4_LEVELS[i] + NF4_LEVELS[i + 1]) / 2.0
///
/// A value `v` maps to level index `partition_point(THRESHOLDS, |&t| t < v)`.
const THRESHOLDS: [f32; 15] = {
    let mut t = [0.0f32; 15];
    let mut i = 0usize;
    while i < 15 {
        t[i] = (NF4_LEVELS[i] + NF4_LEVELS[i + 1]) / 2.0;
        i += 1;
    }
    t
};

// ─── Encode helpers ──────────────────────────────────────────────────────────

/// Map a single f32 in [−1, 1] to its nearest NF4 level index (0..=15).
///
/// Uses a linear scan of the 15 midpoint thresholds — constant time, branchless.
#[inline]
fn quantize_scalar(v: f32) -> u8 {
    THRESHOLDS.partition_point(|&t| t < v) as u8
}

/// Pack normalised slice `normed` into 4-bit nibbles, two per byte.
///
/// Byte layout:  `byte[i]` = `lo_nibble(dim 2i) | hi_nibble(dim 2i+1)`
fn pack_nibbles(normed: &[f32]) -> Vec<u8> {
    let d = normed.len();
    let bytes = d.div_ceil(2);
    let mut out = vec![0u8; bytes];
    for i in 0..d / 2 {
        let lo = quantize_scalar(normed[2 * i]);
        let hi = quantize_scalar(normed[2 * i + 1]);
        out[i] = lo | (hi << 4);
    }
    if d % 2 == 1 {
        out[d / 2] = quantize_scalar(normed[d - 1]);
    }
    out
}

// ─── Decode helper ───────────────────────────────────────────────────────────

/// Unpack 4-bit nibbles back to f32 and multiply by `scale`.
fn unpack_nibbles(packed: &[u8], scale: f32, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let full_pairs = dim / 2;
    for i in 0..full_pairs {
        let lo = (packed[i] & 0x0F) as usize;
        let hi = ((packed[i] >> 4) & 0x0F) as usize;
        out[2 * i] = NF4_LEVELS[lo] * scale;
        out[2 * i + 1] = NF4_LEVELS[hi] * scale;
    }
    if dim % 2 == 1 {
        let lo = (packed[dim / 2] & 0x0F) as usize;
        out[dim - 1] = NF4_LEVELS[lo] * scale;
    }
    out
}

// ─── Struct ──────────────────────────────────────────────────────────────────

/// Stateless NF4 quantizer.
///
/// No codebook training is required — all state is stored per-vector
/// (a single `f32` scale and `ceil(dim/2)` packed bytes).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Nf4Quantizer;

impl Nf4Quantizer {
    pub fn new() -> Self {
        Self
    }

    /// Encode a batch of float32 vectors to NF4.
    ///
    /// Returns `(packed, scales)`:
    /// - `packed[i]` — `ceil(dim/2)` packed bytes for vector `i`.
    /// - `scales[i]` — abs-max scale for vector `i` (required for decode).
    ///
    /// Runs in parallel across vectors via rayon.
    pub fn encode(&self, vectors: &[Vec<f32>]) -> (Vec<Vec<u8>>, Vec<f32>) {
        vectors
            .par_iter()
            .map(|v| {
                let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = if abs_max == 0.0 { 1.0f32 } else { abs_max };
                let normed: Vec<f32> = v.iter().map(|x| x / scale).collect();
                (pack_nibbles(&normed), scale)
            })
            .unzip()
    }

    /// Decode NF4 packed bytes back to float32.
    ///
    /// # Arguments
    /// - `packed` — one `Vec<u8>` per vector (length `ceil(dim/2)`).
    /// - `scales` — per-vector abs-max scales from `encode`.
    /// - `dim`    — original vector dimension.
    pub fn decode(&self, packed: &[Vec<u8>], scales: &[f32], dim: usize) -> Vec<Vec<f32>> {
        packed
            .par_iter()
            .zip(scales.par_iter())
            .map(|(p, &scale)| unpack_nibbles(p, scale, dim))
            .collect()
    }

    /// Decode a single packed NF4 vector (used by `EmbeddingDataset::load`).
    pub fn decode_single(&self, packed: &[u8], scale: f32, dim: usize) -> Vec<f32> {
        unpack_nibbles(packed, scale, dim)
    }

    /// Compression ratio of NF4 vs raw f32 storage for a given dimension.
    ///
    /// Numerator: `dim × 4` bytes (f32).
    /// Denominator: `ceil(dim/2)` nibble bytes + 4 bytes for the f32 scale.
    pub fn compression_ratio(dim: usize) -> f32 {
        let packed_bytes = dim.div_ceil(2);
        let stored = packed_bytes + 4; // +4 for the f32 scale
        (dim as f32 * 4.0) / stored as f32
    }

    /// Mean cosine similarity between `original` and `decoded` vector batches.
    pub fn mean_cosine_sim(&self, original: &[Vec<f32>], decoded: &[Vec<f32>]) -> f32 {
        assert_eq!(original.len(), decoded.len());
        let n = original.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = original
            .par_iter()
            .zip(decoded.par_iter())
            .map(|(a, b)| cosine_sim(a, b))
            .sum();
        sum / n as f32
    }
}

// ─── Cosine helper ────────────────────────────────────────────────────────────

pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic "Gaussian-ish" vector from a simple LCG.
    fn make_vector(seed: u32, dim: usize) -> Vec<f32> {
        let mut state = seed as u64;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let frac = (state >> 33) as f32 / u32::MAX as f32;
                frac * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn encode_decode_shape_even_dim() {
        let q = Nf4Quantizer::new();
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| make_vector(i, 128)).collect();
        let (packed, scales) = q.encode(&vecs);
        assert_eq!(packed.len(), 10);
        // ceil(128/2) = 64 bytes per vector
        for p in &packed {
            assert_eq!(p.len(), 64);
        }
        assert_eq!(scales.len(), 10);
        let decoded = q.decode(&packed, &scales, 128);
        assert_eq!(decoded.len(), 10);
        for d in &decoded {
            assert_eq!(d.len(), 128);
        }
    }

    #[test]
    fn encode_decode_shape_odd_dim() {
        let q = Nf4Quantizer::new();
        let vecs: Vec<Vec<f32>> = (0..5).map(|i| make_vector(i, 127)).collect();
        let (packed, scales) = q.encode(&vecs);
        // ceil(127/2) = 64 bytes
        for p in &packed {
            assert_eq!(p.len(), 64);
        }
        let decoded = q.decode(&packed, &scales, 127);
        for d in &decoded {
            assert_eq!(d.len(), 127);
        }
    }

    #[test]
    fn cosine_sim_meets_nf4_contract() {
        // Contract: cosine ≥ 0.98 for Gaussian-ish embeddings.
        let q = Nf4Quantizer::new();
        let vecs: Vec<Vec<f32>> = (0..200).map(|i| make_vector(i, 768)).collect();
        let (packed, scales) = q.encode(&vecs);
        let decoded = q.decode(&packed, &scales, 768);
        let sim = q.mean_cosine_sim(&vecs, &decoded);
        assert!(
            sim >= 0.98,
            "NF4 cosine sim {:.4} < 0.98 contract (dim=768)",
            sim
        );
    }

    #[test]
    fn compression_ratio_approx_8x() {
        // dim=512: float32 = 2048 bytes; packed = 256 + 4 = 260 bytes → ~7.9×
        let ratio = Nf4Quantizer::compression_ratio(512);
        assert!(ratio >= 7.5, "compression ratio {:.2} < 7.5", ratio);
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let q = Nf4Quantizer::new();
        let vecs = vec![vec![0.0f32; 64]];
        let (packed, scales) = q.encode(&vecs);
        let decoded = q.decode(&packed, &scales, 64);
        assert_eq!(decoded[0].len(), 64);
    }
}
