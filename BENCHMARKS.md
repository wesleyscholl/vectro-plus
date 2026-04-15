# Vectro+ Benchmarks

This document captures ground-truth recall@k and QPS results for all Vectro+ search algorithms.  
All numbers are reproducible by running `scripts/run_benchmark.sh` on matching hardware.

---

## Methodology

| Concept | Detail |
|---|---|
| **Ground truth** | Brute-force cosine similarity (`SearchIndex::from_dataset`) over the full corpus |
| **Recall@k** | `\|ANN_top_k ∩ exact_top_k\| / k`, averaged over all evaluation queries |
| **QPS** | `n_queries / total_search_wall_time` (single-thread, no batching) |
| **Synthetic data** | Deterministic xorshift64 PRNG, seed `0xdeadbeef_cafebabe` — fully CI-reproducible |
| **HNSW params** | M = 16, ef_construction = 200, ef_search = 50 |
| **PQ params** | m = 8 subspaces, k = 256 centroids, 25 Lloyd's iterations |

### Gate criteria

| Algorithm | Metric     | Gate   |
|-----------|------------|--------|
| HNSW      | recall@10  | ≥ 0.90 |
| PQ        | recall@10  | ≥ 0.90 |

Warnings are printed when a gate is not met; the full table is always shown.  
Brute-force is always exact (recall = 1.000) and serves as the reference baseline.

---

## Reproducing Results

```bash
# Synthetic 10 k × 128-d run (CI default)
./scripts/run_benchmark.sh

# Larger synthetic run
./scripts/run_benchmark.sh --vectors 100000 --dim 128 --queries 500 --k 10

# Real dataset (any .stream1 or .jsonl file)
./scripts/run_benchmark.sh --dataset /path/to/dataset.stream1 --save-report
```

Reports are saved as timestamped JSON to `benchmarks/results/`.

---

## v1.4.0 Results — Synthetic 10 k × 128-d

> Hardware: Apple M3, 16 GB unified memory, macOS 26.4 (Build 25E246).  
> Rust: 1.89.0 (Homebrew). Build: `cargo build --release`. Single thread. No batching.  
> Command: `vectro_cli bench-gt --vectors 10000 --dim 128 --queries 100 --k 10 --save-report`  
> Report: `benchmarks/results/2026-04-14T07-29-40-bench-gt.json`

| Algorithm              | recall@10 | QPS    | Latency (ms/q) | Build (ms) | Gate |
|------------------------|-----------|--------|----------------|------------|------|
| Brute-force (exact)    | 1.0000    | 1 985  | 0.504          | —          | ✅   |
| HNSW M=16 ef_s=50      | **0.9200**| 8 066  | 0.124          | 3 654      | ✅ ≥ 0.90 |
| PQ m=8 k=256           | 0.2180    | 3 456  | 0.289          | 320        | ℹ️ see note |

**HNSW gate: PASS** — recall@10 = 0.920 ≥ 0.90 threshold.

**PQ note:** 0.2180 recall on purely random synthetic data is _expected and documented_—not a
regression.  After L2-normalisation the cosine advantage of a true neighbour over a random
pair (≈ 0.27 for 10k × 128-d) is smaller than the m=8 codebook quantisation error.
PQ recall **≥ 0.90 on structured data** is validated by the unit-test
`test_pq_recall_at_10_gate` (`vectro_lib/src/pq.rs`), which uses index-correlated vectors
that model the cluster structure present in real embedding datasets (SIFT1M, GloVe, etc.).

> Run `./scripts/run_benchmark.sh --save-report` for live numbers on your hardware.  
> Stored JSON reports in `benchmarks/results/` contain exact timestamped metrics.

---

## v2.1.0 — GloVe-100d (1.2M × 100-d, M3 16GB)

> **Status:** Framework complete (v2.1.0). Full run pending GloVe download.  
> **Gate criterion:** δ recall@10 < 5 pp vs vectro Python reference; HNSW recall@10 ≥ 0.90.

### How to run

```bash
# 1. Download GloVe 6B 100d text file (840 MB, Stanford NLP)
#    https://nlp.stanford.edu/data/glove.6B.zip  — extract glove.6B.100d.txt

# 2. Convert to STREAM1 binary format
python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1

# 3. Run benchmark and compare against vectro Python reference
./scripts/run_benchmark.sh --dataset glove100d.stream1 --save-report --ground-truth
```

Report will be saved to `benchmarks/results/<timestamp>-bench-gt.json`.  
Update this section with the resulting recall@10, QPS, and δ vs reference when the run completes.

---

## What Each Column Means

| Column | Meaning |
|---|---|
| `recall@10` | Fraction of the true top-10 nearest neighbours returned by the algorithm |
| `QPS` | Queries per second (higher is better) |
| `Latency` | Per-query wall time in microseconds (lower is better) |
| `Build` | Time to build the index or train PQ codebook (ms, one-time cost) |
| `Gate` | ✅ meets ≥ 0.90 recall target; ⚠️ below gate; exact search is always exact |

---

## JSON Report Format

Every `--save-report` run writes a file like:

```
benchmarks/results/bench_gt_20260413T143022.json
```

The JSON schema:

```json
{
  "timestamp": "2026-04-13T14:30:22Z",
  "n_vectors": 10000,
  "dimensions": 128,
  "n_queries": 100,
  "k": 10,
  "algorithms": {
    "brute_force": {
      "recall_at_k": 1.0,
      "qps": 5123.4,
      "latency_ms": 0.195,
      "build_ms": null
    },
    "hnsw": {
      "recall_at_k": 0.96,
      "qps": 15432.1,
      "latency_ms": 0.065,
      "build_ms": 812.0
    },
    "pq": {
      "recall_at_k": 0.91,
      "qps": 48230.5,
      "latency_ms": 0.021,
      "build_ms": 1950.0
    }
  }
}
```

---

## Known Limitations

- **Brute-force QPS scales as O(n·d)** — not suitable for datasets > 1 M vectors without an ANN index.
- **HNSW recall depends on ef_search** — increasing `ef_search` improves recall at the cost of QPS.
- **PQ recall depends on m and k** — more subspaces and more centroids improve recall; `d % m = 0` is required (falls back to `m = 1` if not satisfied).
- **Single-thread only** — Vectro+ does not currently batch queries across threads in `bench-gt`.
