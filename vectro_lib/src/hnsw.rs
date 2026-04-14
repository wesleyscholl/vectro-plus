//! Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor index.
//!
//! # Algorithm
//! HNSW builds a layered proximity graph. Layer 0 contains all nodes with the densest
//! connections; higher layers are increasingly sparse sub-graphs that act as "highways"
//! to speed up the graph traversal during search.
//!
//! ## Construction — INSERT(q)
//! 1. Assign a random max layer `l = floor(-ln(U(0,1)) * m_l)` where `m_l = 1/ln(M)`.
//! 2. Phase 1 — greedy descent: for each layer `L` down to `l+1`, navigate with ef=1
//!    to refine the entry point.
//! 3. Phase 2 — beam insertion: for each layer `min(L, l)` down to 0 search with
//!    ef_construction candidates, select M nearest neighbors, add bidirectional edges.
//! 4. If `l > L`, update the entry point.
//!
//! ## Search — KNN(q, k, ef)
//! 1. Greedy descent from the top layer to layer 1 (ef=1 per layer).
//! 2. Layer-0 beam search with ef candidates.
//! 3. Return top-k from the result window.
//!
//! # Cosine Similarity
//! Vectors are L2-normalised on insertion. The inner product of two unit vectors equals
//! their cosine similarity; higher score = more similar.
//!
//! # Complexity
//! * Build: O(N · M · log N) average-case
//! * Query: O(log N) average-case with ef-proportional constant
//!
//! # Reference
//! Malkov & Yashunin 2016, "Efficient and robust approximate nearest neighbor search
//! using Hierarchical Navigable Small World graphs". arXiv:1603.09320

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use serde::{Deserialize, Serialize};
use crate::Embedding;

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// `f32` wrapper providing a total order via `total_cmp` (NaN sorts last).
/// Used as the priority key in BinaryHeap so that the heap is well-defined even
/// when similarity scores contain edge-case floats.
#[derive(Clone, Copy, PartialEq)]
struct Score(f32);

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ─── Graph Node ───────────────────────────────────────────────────────────────

/// A single node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    /// Original embedding identifier.
    id: String,
    /// L2-normalised vector (unit length). Used for dot-product similarity.
    vector: Vec<f32>,
    /// `edges[layer]` = indices of connected neighbor nodes at that layer.
    edges: Vec<Vec<usize>>,
}

// ─── Public Index ─────────────────────────────────────────────────────────────

/// HNSW Approximate Nearest-Neighbor Index.
///
/// Build with [`HnswIndex::build`], query with [`HnswIndex::search`],
/// persist with [`HnswIndex::save`] / [`HnswIndex::load`].
///
/// # Example
/// ```
/// use vectro_lib::{Embedding, hnsw::HnswIndex};
///
/// let data = vec![
///     Embedding::new("a", vec![1.0_f32, 0.0]),
///     Embedding::new("b", vec![0.0_f32, 1.0]),
///     Embedding::new("c", vec![-1.0_f32, 0.0]),
/// ];
/// let idx = HnswIndex::build(&data, 4, 200, 50);
/// let results = idx.search(&[1.0, 0.0], 1);
/// assert_eq!(results[0].0, "a");
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    /// All inserted nodes, in insertion order.
    nodes: Vec<HnswNode>,
    /// Index into `nodes` for the entry point (highest-layer node).
    entry: Option<usize>,
    /// Highest layer currently present in the graph (−1 if empty).
    max_layer: i32,
    /// Maximum connections per node per non-zero layer (M in the paper).
    m: usize,
    /// Maximum connections per node at layer 0 (= 2 × M per paper).
    m0: usize,
    /// Dynamic candidate list size during index construction.
    ef_construction: usize,
    /// Default dynamic candidate list size during queries.
    pub ef_search: usize,
    /// Layer probability normalisation: `1 / ln(M)`.
    m_l: f64,
    /// xorshift64 RNG state — updated on each insert for reproducible builds.
    rng_state: u64,
}

impl HnswIndex {
    /// Create an empty index.
    ///
    /// # Arguments
    /// * `m`               – max connections per node, per layer (paper default: 16).
    /// * `ef_construction` – beam width during build (paper default: 200).
    /// * `ef_search`       – beam width during queries (tune for recall/speed trade-off).
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        assert!(m >= 2, "m must be ≥ 2; got {}", m);
        Self {
            nodes: Vec::new(),
            entry: None,
            max_layer: -1,
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            m_l: 1.0 / (m as f64).ln(),
            rng_state: 0xdead_beef_cafe_1234,
        }
    }

    /// Build an index from a slice of embeddings.
    ///
    /// Equivalent to calling [`HnswIndex::new`] and then calling [`insert`] for each element.
    ///
    /// [`insert`]: HnswIndex::insert
    pub fn build(data: &[Embedding], m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let mut idx = Self::new(m, ef_construction, ef_search);
        for e in data {
            idx.insert(e);
        }
        idx
    }

    /// Insert one embedding into the index.
    ///
    /// The vector is L2-normalised before storage. Zero-norm vectors are stored as-is and
    /// will score 0 against all queries.
    pub fn insert(&mut self, e: &Embedding) {
        let norm = l2_norm(&e.vector);
        let vector: Vec<f32> = if norm > 0.0 {
            e.vector.iter().map(|v| v / norm).collect()
        } else {
            e.vector.clone()
        };

        let new_idx = self.nodes.len();
        let new_layer = self.random_layer();

        // Initialise edges for every layer up to new_layer (all empty).
        self.nodes.push(HnswNode {
            id: e.id.clone(),
            vector,
            edges: vec![Vec::new(); new_layer as usize + 1],
        });

        // First node — becomes the entry point.
        if self.entry.is_none() {
            self.entry = Some(new_idx);
            self.max_layer = new_layer;
            return;
        }

        let mut ep = self.entry.unwrap();
        let cur_max_layer = self.max_layer;

        // ── Phase 1: greedy descent from top layer down to new_layer+1 ──────────
        // Use ef=1 (greedy best-first) to find a tight entry point near the new node
        // at each layer above the new node's max layer.
        for lc in ((new_layer + 1)..=cur_max_layer).rev() {
            let results = self.search_layer(new_idx, ep, 1, lc as usize);
            if let Some(&(_, closest)) = results.first() {
                ep = closest;
            }
        }

        // ── Phase 2: insert at each layer from min(L, new_layer) down to 0 ──────
        for lc in (0..=std::cmp::min(cur_max_layer, new_layer)).rev() {
            let layer = lc as usize;
            let ef = self.ef_construction;

            // Find ef_construction nearest neighbors at this layer.
            let candidates = self.search_layer(new_idx, ep, ef, layer);

            // Refine ep for the next (lower) layer.
            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }

            // Maximum connections for this layer (layer 0 allows 2×M).
            let m_max = if layer == 0 { self.m0 } else { self.m };

            // Simple neighbor selection: take the closest m_max candidates.
            let neighbors: Vec<usize> = candidates.iter().take(m_max).map(|&(_, i)| i).collect();

            // Wire new_node → each neighbor.
            self.nodes[new_idx].edges[layer] = neighbors.clone();

            // Wire each neighbor → new_node (bidirectional), then prune if oversized.
            for &nb in &neighbors {
                // Extend neighbor's edge list if it was inserted at a lower layer.
                while self.nodes[nb].edges.len() <= layer {
                    self.nodes[nb].edges.push(Vec::new());
                }
                self.nodes[nb].edges[layer].push(new_idx);

                // Prune back to m_max if the neighbor's list is now too long.
                if self.nodes[nb].edges[layer].len() > m_max {
                    self.shrink_edges(nb, layer, m_max);
                }
            }
        }

        // ── Update entry point if new node resides at a higher layer ──────────
        if new_layer > cur_max_layer {
            self.entry = Some(new_idx);
            self.max_layer = new_layer;
        }
    }

    /// Find the top-k approximate nearest neighbors for `query`.
    ///
    /// Returns `Vec<(id, score)>` sorted by descending cosine similarity.
    /// Uses `self.ef_search` as the beam width.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        self.search_with_ef(query, k, self.ef_search)
    }

    /// Find the top-k approximate nearest neighbors with an explicit ef.
    ///
    /// A larger `ef` increases recall at the cost of additional comparisons.
    /// `ef` is clamped to `max(ef, k)` to guarantee at least k candidates.
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(String, f32)> {
        if self.entry.is_none() || k == 0 {
            return Vec::new();
        }

        let norm = l2_norm(query);
        if norm == 0.0 {
            return Vec::new();
        }
        let q: Vec<f32> = query.iter().map(|v| v / norm).collect();

        let mut ep = self.entry.unwrap();

        // Phase 1: greedy descent from the top layer to layer 1 (ef=1).
        for lc in (1..=self.max_layer as usize).rev() {
            let results = self.search_layer_query(&q, ep, 1, lc);
            if let Some(&(_, closest)) = results.first() {
                ep = closest;
            }
        }

        // Phase 2: beam search at layer 0 with the user-supplied ef.
        let ef_clamped = ef.max(k);
        let results = self.search_layer_query(&q, ep, ef_clamped, 0);

        results
            .into_iter()
            .take(k)
            .map(|(score, node_idx)| (self.nodes[node_idx].id.clone(), score))
            .collect()
    }

    /// Number of indexed embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Serialise the index to a file (bincode encoding).
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let bytes = bincode::serialize(self)?;
        let mut f = File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }

    /// Deserialise an index from a file previously written by [`save`].
    ///
    /// [`save`]: HnswIndex::save
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let idx: Self = bincode::deserialize(&bytes)?;
        Ok(idx)
    }

    // ─── Private helpers ────────────────────────────────────────────────────

    /// Draw a random layer for a new node.
    ///
    /// Layer assignment: `floor(-ln(U) * m_l)` where U ~ Uniform(0, 1).
    /// The probability of assigning layer `l` is `(1 - 1/M)^l / M`, giving an
    /// exponentially decaying distribution that keeps higher layers sparse.
    fn random_layer(&mut self) -> i32 {
        let u = self.xorshift_f64();
        let layer = (-u.ln() * self.m_l).floor() as i32;
        layer.max(0)
    }

    /// xorshift64 pseudo-random number generator.
    ///
    /// Produces a uniform float in (0, 1) — never exactly 0 to avoid ln(0) = −∞.
    fn xorshift_f64(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        // 53-bit mantissa → uniform in [0, 1); clamp away from 0.
        let f = (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        f.max(1e-15_f64)
    }

    /// Beam search in the graph at `layer`, starting from entry point `ep`.
    ///
    /// The query is the **normalised** vector of node `query_node_idx`.
    /// Returns `Vec<(score, node_idx)>` sorted by descending score (up to `ef` items).
    fn search_layer(&self, query_node_idx: usize, ep: usize, ef: usize, layer: usize) -> Vec<(f32, usize)> {
        let query = self.nodes[query_node_idx].vector.clone();
        self.search_layer_query(&query, ep, ef, layer)
    }

    /// Beam search in the graph at `layer`, starting from entry point `ep`.
    ///
    /// `query` must already be L2-normalised.
    /// Returns `Vec<(score, node_idx)>` sorted by descending score (up to `ef` items).
    fn search_layer_query(&self, query: &[f32], ep: usize, ef: usize, layer: usize) -> Vec<(f32, usize)> {
        let ep_score = dot(&self.nodes[ep].vector, query);

        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(ep);

        // Max-heap for the candidates to explore next (best = highest score = top of heap).
        let mut candidates: BinaryHeap<(Score, usize)> = BinaryHeap::new();
        candidates.push((Score(ep_score), ep));

        // Min-heap as the result window of size ef (we evict the worst when full).
        // BinaryHeap is a max-heap, so we use Reverse<Score> to get a min-heap.
        let mut window: BinaryHeap<(Reverse<Score>, usize)> = BinaryHeap::new();
        window.push((Reverse(Score(ep_score)), ep));

        while let Some((Score(c_score), c)) = candidates.pop() {
            // If the best remaining candidate is worse than the worst item already in the
            // window (and window is full), no further expansion can improve the result.
            let worst_in_window = window
                .peek()
                .map(|&(Reverse(Score(s)), _)| s)
                .unwrap_or(f32::NEG_INFINITY);

            if window.len() >= ef && c_score < worst_in_window {
                break;
            }

            // Expand c: visit its neighbors at this layer.
            let neighbors: Vec<usize> = if layer < self.nodes[c].edges.len() {
                self.nodes[c].edges[layer].clone()
            } else {
                Vec::new()
            };

            for nb in neighbors {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);

                let nb_score = dot(&self.nodes[nb].vector, query);
                let worst = window
                    .peek()
                    .map(|&(Reverse(Score(s)), _)| s)
                    .unwrap_or(f32::NEG_INFINITY);

                if window.len() < ef || nb_score > worst {
                    candidates.push((Score(nb_score), nb));
                    window.push((Reverse(Score(nb_score)), nb));
                    if window.len() > ef {
                        window.pop(); // evict the worst
                    }
                }
            }
        }

        // Drain window and sort descending by score.
        let mut result: Vec<(f32, usize)> = window
            .into_iter()
            .map(|(Reverse(Score(s)), i)| (s, i))
            .collect();
        result.sort_by(|a, b| b.0.total_cmp(&a.0));
        result
    }

    /// Shrink the edge list of `node_idx` at `layer` to at most `max_edges` neighbors.
    ///
    /// Uses simple selection: keep the `max_edges` neighbors with the highest cosine
    /// similarity to the node (i.e., the closest ones in the normalised vector space).
    fn shrink_edges(&mut self, node_idx: usize, layer: usize, max_edges: usize) {
        let query = self.nodes[node_idx].vector.clone();
        let edges = self.nodes[node_idx].edges[layer].clone();

        let mut scored: Vec<(f32, usize)> = edges
            .iter()
            .map(|&nb| (dot(&self.nodes[nb].vector, &query), nb))
            .collect();

        // Keep only the max_edges most similar neighbors.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(max_edges);

        self.nodes[node_idx].edges[layer] = scored.into_iter().map(|(_, i)| i).collect();
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a vector.
#[inline]
fn l2_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic dataset: N random-ish unit-norm vectors in R^dim.
    fn make_embeddings(n: usize, dim: usize) -> Vec<Embedding> {
        let mut rng = 0xdead_beef_u64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                v.push((rng >> 33) as f32 / u32::MAX as f32 - 0.5);
            }
            out.push(Embedding::new(format!("id_{}", i), v));
        }
        out
    }

    #[test]
    fn test_empty_index_returns_nothing() {
        let idx = HnswIndex::new(16, 200, 50);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        let r = idx.search(&[1.0_f32, 0.0], 10);
        assert!(r.is_empty());
    }

    #[test]
    fn test_single_insert_is_found() {
        let mut idx = HnswIndex::new(16, 200, 50);
        idx.insert(&Embedding::new("solo", vec![1.0_f32, 0.0]));
        assert_eq!(idx.len(), 1);
        let r = idx.search(&[1.0, 0.0], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "solo");
    }

    #[test]
    fn test_nearest_of_four_cardinal_vectors() {
        let data = vec![
            Embedding::new("north", vec![ 0.0_f32,  1.0]),
            Embedding::new("east",  vec![ 1.0_f32,  0.0]),
            Embedding::new("south", vec![ 0.0_f32, -1.0]),
            Embedding::new("west",  vec![-1.0_f32,  0.0]),
        ];
        let idx = HnswIndex::build(&data, 4, 200, 50);
        // A query pointing slightly east should retrieve "east" as the first result.
        let r = idx.search(&[1.0, 0.1], 1);
        assert_eq!(r[0].0, "east");
        // A query pointing slightly north should retrieve "north" first.
        let r = idx.search(&[0.1, 1.0], 1);
        assert_eq!(r[0].0, "north");
    }

    #[test]
    fn test_scores_are_in_valid_range() {
        let data = make_embeddings(100, 32);
        let idx = HnswIndex::build(&data, 16, 200, 50);
        let query = data[0].vector.clone();
        let results = idx.search(&query, 10);
        for &(_, score) in &results {
            assert!(
                score >= -1.0 - 1e-5 && score <= 1.0 + 1e-5,
                "cosine score out of [-1, 1]: {}",
                score
            );
        }
    }

    #[test]
    fn test_k_larger_than_corpus_returns_all() {
        let data = make_embeddings(5, 8);
        let idx = HnswIndex::build(&data, 4, 200, 50);
        let r = idx.search(&data[0].vector, 100);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn test_results_are_sorted_descending() {
        let data = make_embeddings(200, 32);
        let idx = HnswIndex::build(&data, 16, 200, 50);
        let query = data[0].vector.clone();
        let results = idx.search(&query, 10);
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results not sorted: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    /// Gate: recall@10 ≥ 0.95 vs brute-force on a 1 000-vector, 64-dim dataset.
    ///
    /// This is the v1.3.0 ship gate.  If this fails the recall floor is not met —
    /// tune M or ef_construction before merging.
    #[test]
    fn test_recall_at_10_gate() {
        use crate::search::SearchIndex;

        let data = make_embeddings(1000, 64);
        let hnsw = HnswIndex::build(&data, 16, 200, 50);
        let brute = SearchIndex::from_dataset(&data);

        const N_QUERIES: usize = 100;
        const K: usize = 10;
        let mut hits = 0usize;

        for i in 0..N_QUERIES {
            let q = &data[i].vector;
            let exact: std::collections::HashSet<String> = brute
                .top_k(q, K)
                .into_iter()
                .map(|(id, _)| id.to_string())
                .collect();
            let approx = hnsw.search(q, K);
            hits += approx.iter().filter(|(id, _)| exact.contains(id)).count();
        }

        let recall = hits as f64 / (N_QUERIES * K) as f64;
        assert!(
            recall >= 0.95,
            "recall@10 = {:.4} — below the 0.95 gate (N={}, K={})",
            recall,
            N_QUERIES,
            K
        );
    }

    /// Gate: save → load round-trip produces identical search results.
    #[test]
    fn test_save_load_roundtrip() {
        let data = make_embeddings(200, 32);
        let idx = HnswIndex::build(&data, 16, 200, 50);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hnsw");
        idx.save(path.to_str().unwrap()).unwrap();

        let loaded = HnswIndex::load(path.to_str().unwrap()).unwrap();
        assert_eq!(idx.len(), loaded.len());

        let query = vec![1.0_f32; 32];
        let orig = idx.search(&query, 5);
        let from_disk = loaded.search(&query, 5);
        assert_eq!(orig, from_disk, "results differ after save/load round-trip");
    }

    #[test]
    fn test_zero_norm_query_returns_empty() {
        let data = make_embeddings(10, 8);
        let idx = HnswIndex::build(&data, 4, 200, 50);
        // A zero-norm query cannot be normalised; search returns empty.
        let r = idx.search(&vec![0.0_f32; 8], 5);
        assert!(r.is_empty(), "expected empty result for zero-norm query");
    }

    #[test]
    fn test_ef_search_controls_candidate_pool() {
        // Verify that a higher ef value can only equal or improve recall.
        use crate::search::SearchIndex;
        let data = make_embeddings(500, 32);
        let idx = HnswIndex::build(&data, 16, 200, 10);
        let brute = SearchIndex::from_dataset(&data);
        let q = &data[0].vector;
        let exact: std::collections::HashSet<String> = brute
            .top_k(q, 10)
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect();

        let low_ef = idx.search_with_ef(q, 10, 10);
        let high_ef = idx.search_with_ef(q, 10, 200);
        let low_hits = low_ef.iter().filter(|(id, _)| exact.contains(id)).count();
        let high_hits = high_ef.iter().filter(|(id, _)| exact.contains(id)).count();
        assert!(
            high_hits >= low_hits,
            "higher ef ({}) should produce recall ≥ lower ef ({}): {} vs {}",
            200, 10, high_hits, low_hits
        );
    }
}
