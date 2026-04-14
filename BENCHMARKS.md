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

> Hardware: Apple M3, 16 GB unified memory, macOS 15 (Sequoia).  
> Build: `cargo build --release`. Single thread. No batching.  
> Command: `vectro bench-gt --vectors 10000 --dim 128 --queries 100 --k 10 --save-report`

| Algorithm              | recall@10 | QPS       | Latency (µs/q) | Build (ms) | Gate |
|------------------------|-----------|-----------|----------------|------------|------|
| Brute-force (exact)    | 1.0000    | ~5,000    | ~200           | —          | ✅   |
| HNSW M=16 ef_s=50      | ≥ 0.95    | ~15,000   | ~70            | ~800       | ✅   |
| PQ m=8 k=256           | ≥ 0.90    | ~50,000   | ~20            | ~2,000     | ✅   |

> Run `./scripts/run_benchmark.sh --save-report` for live numbers on your hardware.  
> Stored JSON reports in `benchmarks/results/` contain exact timestamped metrics.

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
