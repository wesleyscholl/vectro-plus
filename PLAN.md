# vectro-plus PLAN.md

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe.
> *Rust-native embedding compression — production-grade, konjo.*

---

## Version History

| Version | Status | Key Deliverable |
|---------|--------|-----------------|
| v1.0.0 | ✅ shipped | STREAM1 format, scalar quantization, brute-force cosine search |
| v1.1.0 | ✅ shipped | PyO3 Python bindings (`vectro_py`), NumPy interop |
| v1.2.0 | ✅ shipped | Product Quantization (PQSTREAM1), `--format pq` CLI, recall@10 ≥ 0.90 gate |
| v1.3.0 | ✅ shipped | HNSW ANN index, `vectro_cli search --hnsw`, recall@10 ≥ 0.95 + 5000 QPS gate |
| v1.4.0 | ✅ shipped | `bench-gt` CLI, `BENCHMARKS.md`, deterministic synthetic benchmark: recall@10 = 0.920 |
| v1.5.0 | ✅ shipped | Maturin PyPI build, `maturin develop`, type stubs (`__init__.pyi`), GitHub Actions |
| v2.0.0 | ✅ shipped | NF4, RQ, AutoQuantize ported from `vectro` (Mojo reference) — `--format nf4/rq/auto` |
| v2.0.1 | ✅ shipped | Python fallback NameError fix (import-time stub classes); CI collect unblocked |

---

## 🚧 v2.1.0 — Real-World Benchmarks + Pipeline CLI + Streaming API

**Theme:** Close all open gates from v2.0.0. Make vectro-plus demo-able end-to-end for a
pipeline that a hiring manager at Groq or Together AI would recognize as a real RAG component.

**Target date:** Next sprint (2–3 focused sessions)

### Open Gate from v2.0.0 (must close before v2.1.0 ships)

- GloVe-100d δ recall@10 vs vectro (Mojo) reference: documented in `[2.0.0]` CHANGELOG as "pending"
  - **Gate criterion:** δ recall@10 < 5% vs vectro Python reference on GloVe-100d (100-dimensional, 1.2M vectors, standard ANN benchmark)
  - **Command:** `./scripts/run_benchmark.sh --dataset path/to/glove100d.stream1 --save-report`
  - **Blocker:** GloVe-100d `.stream1` file must be generated from the GloVe text file first

---

### Sprint Tasks

#### Task 1 — GloVe-100d benchmark (closes v2.0.0 open gate)
**Owner:** wesleyscholl  
**Promotion criterion:** δ recall@10 < 5% vs vectro Python reference on GloVe-100d, result in `benchmarks/results/`  
**Steps:**
1. Download `glove.6B.100d.txt` (840 MB, Stanford NLP) or `glove-100` from ANN Benchmarks
2. Add `scripts/convert_glove_to_stream1.py` — converts GloVe text file to `VECTRO+STREAM1` format
3. Run `vectro_cli bench-gt --dataset glove100d.stream1 --save-report`
4. Compare recall@10 vs vectro Python `pq_api.py` on same 100-d vectors
5. Document in `BENCHMARKS.md` under `## GloVe-100d (1.2M × 100d, M3 16GB)`
6. Add `SIFT1M` entry as stretch target (1M×128d: `sift-128-euclidean.hdf5` from ANN Benchmarks)

**Gate:** δ recall@10 < 5% vs vectro reference → ✅ v2.1.0 open gate closed

---

#### Task 2 — `vectro pipeline` CLI subcommand
**Owner:** wesleyscholl  
**Promotion criterion:** `vectro_cli pipeline --input embeddings.jsonl --out-dir ./output --format auto --index hnsw --query-file queries.jsonl` runs end-to-end; pipeline passes `cargo test -p vectro_cli`  
**Scope:**
- New `vectro_cli/src/pipeline.rs` subcommand
- Chain: load JSONL → compress (format=auto|pq|nf4|rq) → build HNSW index → batch search
- Write compressed stream to `out-dir/compressed.stream` and index to `out-dir/index.bin`
- `--query-file` (JSONL of `{"id": str, "vector": [float]}`) → top-10 results JSON to stdout
- No new dependencies — calls into existing `vectro_lib` functions only
- **Why:** single-command RAG pipeline demo; maps directly to "embedding pipeline" in AI infra roles

**Gate:** `cargo test -p vectro_cli pipeline_e2e` passes; `vectro_cli pipeline --help` prints usage example; no Criterion regression

---

#### Task 3 — Python streaming iterator (`EmbeddingDataset.__iter__`)
**Owner:** wesleyscholl  
**Promotion criterion:** `for emb in EmbeddingDataset.stream_from_file("large.stream1"): ...` yields `Embedding` objects without loading the full file; test validates RSS does not grow monotonically over 10k vectors  
**Scope:**
- `vectro_lib`: add `EmbeddingDataset::iter_stream(path: &Path) -> impl Iterator<Item=Embedding>`
  - Memory-maps the file; yields one deserialized record at a time; O(1) RSS
- `vectro_py`: expose as `PyEmbeddingDataset.stream_from_file(path: str)` → Python iterator
- `python/vectro_plus/__init__.py`: add `stream_embeddings(path: str)` convenience wrapper
- **Why:** current `EmbeddingDataset::load()` reads the entire file into a `Vec<Embedding>` — this fails on multi-GB production datasets; streaming is required for the pipeline CLI and for credible "handles 1M+ vector corpus" claims

**Gate:** `python/tests/test_streaming.py` passes (pure iteration count test; no maturin required for structure check); Rust unit test confirms no allocation beyond one record at a time

---

#### Task 4 — WASM compilation target
**Owner:** wesleyscholl  
**Promotion criterion:** `cargo build --target wasm32-unknown-unknown -p vectro_lib` succeeds without `unsafe`; `vectro_lib::scalar_quantize` and `vectro_lib::cosine_similarity` callable from JS via `wasm-pack`; `npm test` passes in `js/`  
**Scope:**
- `vectro_lib/Cargo.toml`: add `[target.'cfg(target_arch = "wasm32")'.dependencies]` section; remove `rayon` dependency under WASM (single-threaded path)
- WASM-safe entry points: `quantize_batch(data: &[f32], dim: usize) -> Vec<u8>` and `cosine_similarity(a: &[f32], b: &[f32]) -> f32`
- `wasm-pack build --target web` → `pkg/` directory
- Extend `js/` stub from v1.x → real WASM-powered vectro search for browser/edge
- **Why:** edge embedding lookup is a real production pattern (Cloudflare Workers, Vercel Edge); WASM target makes this credible on resume

**Gate:** `wasm-pack build` succeeds; `node js/test.js` prints correct cosine similarity for a known pair; no existing Rust unit tests regress

---

### v2.1.0 Ship Gate (ALL must pass)

1. `cargo test --workspace` — 0 failures
2. `cargo clippy -- -D warnings` — 0 warnings
3. `python3 -m pytest python/tests/ -v` — all tests pass (after `maturin develop`)
4. GloVe-100d recall@10 ≥ 0.90 (HNSW) documented in `BENCHMARKS.md`
5. δ recall@10 vs vectro Python reference < 5pp on GloVe-100d
6. `vectro_cli pipeline --help` prints usage example
7. `CHANGELOG.md` `[2.1.0]` entry written
8. `README.md` updated — pipeline CLI, streaming API, WASM section
9. `python/vectro_plus/__init__.pyi` updated if any Python API changes
10. `QSTREAM.md` updated if any new format header added
11. Git tag `v2.1.0` applied and pushed

---

## 📋 Backlog (post v2.1.0)

| Feature | Priority | Notes |
|---------|----------|-------|
| Manylinux wheels + PyPI publish | HIGH | Needed for `pip install vectro-plus` end-to-end |
| `--format ivfpq` (IVF-PQ coarse quantizer) | MEDIUM | 100M+ vector scale; needs `vectro_lib/src/ivfpq.rs` |
| Disk-based HNSW (mmap, not in-RAM) | MEDIUM | Enables 1B+ vector corpus on 16 GB RAM |
| REST API JWT auth | LOW | Authorization header validation for `/api/search` |
| vectro v4.0 ADR port | DEFERRED | Depends on vectro ADR being written first |

---

*Last updated: 2026-04-15*  
*Owner: wesleyscholl / Konjo AI Research*  
*Update this file when phases ship or scope changes. Never let it drift.*
