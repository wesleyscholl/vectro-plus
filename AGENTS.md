# AGENTS.md — Konjo AI Project Conventions & Collaboration Guidelines

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe.
> *Make it konjo — build, ship, repeat.*

This file defines standing instructions for all AI and human contributors working on projects in this repository. Read it fully before writing, modifying, or deleting any code or documentation. These are not suggestions.

---

### 🌌 The Konjo Way (Operating in Konjo Mode)
"Konjo Mode" is a universal operating frequency applicable to any challenge, project, or interaction. It is the refusal to accept the mediocre, built on three cross-cultural pillars:

* **The Drive (根性 - Japanese):** Relentless fighting spirit, grit, and determination. Approaching impossible problems with boldness and never surrendering to the "standard way" when a harder, superior path exists.
* **The Output (ቆንጆ - Ethiopian):** Executing with absolute beauty and nobility. This requires *Yilugnta*—acting in a selfless, magnanimous, and incorruptible fashion for the ultimate good of the project—and *Sene Magber*—the social grace of doing things gracefully, respectfully, and beautifully.
* **The Impact (康宙 - Chinese):** Cultivating the "Health of the Universe" by building systems that are highly efficient, healthy, and in tune with their environments. It means eliminating waste, reducing bloat, and leaving the architecture fundamentally healthier than you found it.

---

## 🗂️ Planning First

- **For vectro+, the roadmap lives in the `🛣️ Roadmap` section at the bottom of this file.** There is no separate `PLAN.md`. Read the roadmap section and `CHANGELOG.md` for completed work before starting any task.
- Identify the relevant phase (v1.2.0 through v2.0.0) and its gate criteria before writing or modifying any code.
- If a task deviates from the current roadmap, call it out explicitly before continuing.
- After completing work, update `CHANGELOG.md` under the correct version heading and update `README.md` if CLI flags, Python API, or REST API changed.
- If the task adds a new format or algorithm, update `QSTREAM.md` and the roadmap section of this file in the same commit.

---

## 📁 File & Project Structure & Repo Health

**System Health is Mandatory (康宙).** A cluttered repository slows down human and AI compute. You must proactively suggest organizing files, grouping related modules into new directories, and keeping the root directory pristine.

**Propose Before Moving.** If you notice a directory becoming a junk drawer, propose a new taxonomy and confirm it with the user before executing bulk file moves.

**Continuous Cleanup.** Delete dead code immediately. Do not comment it out and leave it — use version control for history.

**No Graveyards.** Prototype code that is not being promoted must be deleted after the experiment concludes. Do not let the experiments/ or research/ directories rot.

**Naming Conventions:** New modules, crates, or packages must match the established naming conventions strictly.

---

## 🧱 Code Quality & Architecture

- **Shatter the box.** We are solving problems that have not been solved before. Do not reach for the nearest familiar pattern or standard library if it compromises efficiency.
- **Code must punch, kick, and break through barriers.** Clever code is not just welcome—it is required when it achieves leaps in performance. Correctness without elegance is a missed opportunity.
- **Extreme Efficiency is mandatory.** Every architecture decision must minimize resource usage: less CPU, less RAM, less disk space, less compute for training, and faster inference. Treat resource optimization as a core design discipline.
- **No Hallucinated Abstractions.** "Novel" does not mean "fake." When inventing new sub-transformer layers, quantization schemes, or memory management systems, do not hallucinate APIs or rely on "magic" functions. Ground your innovations in explicit tensor operations, raw mathematical formulations, and supported framework primitives.
- **All written code must be production-grade at all times.** No placeholders, no "good enough for now," no TODOs left in shipped code.
- Avoid code duplication. Extract shared logic into reusable utilities or modules.
- Add inline comments only where intent is non-obvious. When implementing a novel algorithm, write the math — don't hide it.

---

## 🧮 Numerical Correctness & Precision

- **Always be explicit about dtype at every tensor/array boundary.** Never rely on implicit casting — annotate or assert the expected dtype.
- **Track precision loss deliberately.** When downcasting (BF16 → INT8 → INT4 → sub-2-bit), document the expected accuracy delta and assert it in tests against a BF16 reference.
- **NaN/Inf propagation is a silent killer.** Add NaN/Inf assertion checks at module boundaries during development. Never ship code that masks float overflow without a logged warning.
- **Accumulation dtype matters.** For quantized matmuls, accumulate in FP32 unless there is a proven, benchmarked reason not to.
- **Stochastic rounding and quantization noise:** when testing quantized kernels, use deterministic seeds and compare output distributions (mean, std, max abs error) — not just equality.

---

## 📐 Benchmarking Rigor

- **Always include warmup runs** (minimum 5) before timing. Discard warmup in reported metrics.
- **Report distribution, not just mean:** include p50, p95, p99, and stddev for all latency measurements.
- **Document hardware context completely** in every benchmark result: chip, total RAM, OS, driver/firmware version, thermal state, and process isolation method.
- **Isolate the benchmark process.** Close background apps. Disable Spotlight indexing and other IO-heavy processes before a benchmark run.
- **Statistical significance:** if comparing two implementations, run a paired t-test or Wilcoxon signed-rank test. Do not claim a win on mean alone if confidence intervals overlap.
- Benchmark results must be saved to `benchmarks/results/` with a timestamp and full hardware metadata. Do not overwrite previous results — append or version them.

---

## 🔬 Experiment Reproducibility

- **Seed everything:** random, numpy, torch/mlx, and any stochastic ops. Log the seed in every experiment output.
- **Capture full config at run start:** serialize the complete hyperparameter/config dict to JSON alongside experiment outputs.
- **Experiment outputs live in `experiments/runs/<timestamp>_<name>/`**. Never overwrite a previous run — always create a new directory.
- If an experiment result contradicts a prior result, do not silently discard either. Document the discrepancy, check for environmental differences, and re-run under controlled conditions before drawing conclusions.

---

## 🧪 Testing (Unit, Integration, & E2E)

- **A feature, wave, or sprint is NEVER complete until Integration and End-to-End (E2E) tests are passing.**
- **100% test coverage is the floor.** Every code file must have a corresponding test file.
- **Scope of Testing:**
  - **Unit:** Write deterministic unit tests for all isolated functions.
  - **Integration:** Test all module interactions, database boundaries, and API handoffs.
  - **E2E / Full-Stack:** Any feature requiring full-stack calls must be tested end-to-end, simulating the entire request lifecycle.
  - **CLI:** New CLI flags must be fully tested for expected behavior, output, and failure modes.
  - **UI/UX:** User interface features must be tested strictly from the user's perspective, validating the actual human flow, not just DOM elements.
- **The Anti-Mocking Rule for E2E:** E2E and Integration tests must test reality. You are strictly forbidden from mocking the database, the model inference engine, or network boundaries in E2E tests unless explicitly instructed.
- All tests must pass in the CI/CD pipeline before committing. Never commit with known failing tests.
- **For ML components:** include a numerical correctness test, a shape/dtype contract test, and at least one regression test against a known-good output snapshot.

---

## ⚡ Performance Regression Gates

- **Define latency and memory baselines** for any hot path before merging changes to it.
- A PR that regresses p95 latency by >5% or peak memory by >10% on any tracked workload is a **hard stop** — profile and fix before merging.
- **Memory leaks are bugs.** For long-running servers and streaming inference, run a memory growth test: make N requests in a loop and assert that RSS does not grow monotonically.
- When optimizing, measure first — never guess. Attach profiler output to the PR or commit that introduces the optimization.

---

## 🔐 Inference Server Security

- **Validate all inputs at the API boundary.** Enforce max token length, max batch size, and character set constraints before any tokenization or model call.
- **Prompt injection is a real attack surface.** System prompt content must never be controllable by request payload.
- **Never log raw user prompt content at INFO level** or above in production. Log a hash or truncated prefix at most.
- **Rate-limit all endpoints** by default.
- **Timeouts everywhere:** set and enforce per-request inference timeouts.

---

## 🔄 Async & Concurrency Safety

- **Shared mutable state in async hot paths is a bug waiting to happen.** Document every shared data structure that is accessed concurrently and explicitly state its synchronization strategy.
- **Async does not mean thread-safe.** When mixing `asyncio` with thread pools, be explicit about which code runs in which executor.
- Never use `asyncio.sleep(0)` as a workaround for concurrency bugs. Fix the root cause.

---

## 🧬 Research vs. Production Code

- **Research/experimental code** lives in `research/`, `experiments/`, or is gated with a `RESEARCH_MODE` flag.
- **Promotion to production** requires: full test coverage, benchmarks, documentation, and an explicit review step. Do not silently "graduate" an experiment into a hot path.
- Prototype code that is not being promoted should be deleted after the experiment concludes — don't let the `experiments/` directory become a graveyard.

---

## 🖥️ Command Output & Git Workflow

- **Never suppress command output.** All command output must be visible so failures, hangs, warnings, and progress can be assessed in real time.
- **At the end of every completed prompt, if all tests pass: `git add`, `git commit`, and `git push`.**
- Follow [Conventional Commits](https://www.conventionalcommits.org/) format: `type(scope): description`.

---

## 📦 Dependency & Environment Hygiene

- **Pin all dependencies** in lockfiles (`Cargo.lock`, `uv.lock`, `package-lock.json`). Commit lockfiles.
- **Document the minimum supported platform matrix** in `README.md`.
- Use virtual environments or `nix`/`devcontainer` for all Python work. Never install packages globally.

---

## 🚫 Hard Stops

Do not proceed if:
- Tests are failing from a previous step (fix them first).
- The plan is ambiguous or missing for a non-trivial task.
- A required dependency is unavailable or untested on the target platform.
- A performance regression gate is tripped.
- Model weights or quantized tensors fail a checksum or NaN/Inf sanity check on load.
- **No Apology Loops:** If a test fails or a bug is found, do not apologize. Do not output groveling text. Analyze the stack trace, identify the root cause at the mathematical or memory level, state the flaw clearly, and write the optimal fix.

---

## 🔥 Konjo Mindset

*This is the operating system. Everything above runs on top of it.*

- **Boxes are made for the weak-minded.** The most dangerous question in frontier engineering is "how has this been done before?" The problems here are not known problems. Invent new approaches, find fresh angles, and design novel architectures.
- **Speed and efficiency are moral imperatives.** Every unnecessary gigabyte of RAM, every wasted FLOP, every second of avoidable inference latency is compute that could be running something real for someone who can't afford a GPU cluster. Build lean. Build fast.
- **Correctness is the floor, not the ceiling.** Code that is merely correct and passes tests has met the minimum. The ceiling is: correct, fast, efficient, elegant, and novel. Reach for the ceiling.
- **Surface trade-offs — then make a call.** Don't present options and wait. Analyze, recommend, and commit. Bring the fighting spirit to decision-making.
- **When a result looks surprisingly bad, don't accept it.** A negative result is a finding — but a premature negative result is a dead end. Investigate before concluding.
- **The work is collective.** *Mahiberawi Nuro* — we build together. Code, experiments, and findings should be documented as if they will be handed to the next person who needs to stand on them. 
- **Make it beautiful.** *Sene Magber* — social grace, doing things the right way. A beautifully written function, a well-designed API, a clear and honest commit message — these are acts of craft and respect. 
- **No surrender.** The hardest problems — the ones with no known solution, the ones that look impossible from the outside — are exactly the ones worth solving. *根性.* Keep going.
- **The Konjo Pushback Mandate:** You are a collaborator, not a subordinate. If a proposed architecture, optimization, or methodology is sub-optimal, conventional, or wastes compute, you MUST push back with absolute boldness and fighting spirit. Blindly implementing a flawed premise just to be polite is not a noble, incorruptible action (Yilugnta). Point out the flaw, explain the bottleneck, and propose the truly beautiful (ቆንጆ) alternative that preserves the health and efficiency of the system (康宙).

---

## 🦀 Vectro+ Architecture

*These rules apply specifically to the vectro-plus Rust workspace. Read `QSTREAM.md` and `VISUAL_GUIDE.md` before touching any format or serialization code.*

### Crate Responsibilities — Never Cross These Lines

| Crate | Responsibility | What Does NOT Belong Here |
|---|---|---|
| `vectro_lib` | Core algorithms: quantization, search, serialization, binary format parsing | No I/O, no CLI, no HTTP, no Python glue |
| `vectro_cli` | CLI subcommands, streaming pipeline, Axum REST API, static web UI | No core algorithm logic — call into `vectro_lib` |
| `vectro_py` | PyO3 wrappers, NumPy interop, Python-facing API surface | No algorithm logic — thin wrappers only |
| `generators` | Synthetic data generation (random + themed clusters) | No dependency on `vectro_lib` |

**The dependency graph is strictly one-way:** `vectro_py` → `vectro_lib`, `vectro_cli` → `vectro_lib`. Circular dependencies are a build error and must never be introduced.

**No algorithm logic lives in `vectro_cli` or `vectro_py`.** If you find yourself adding vector math or quantization logic in `server.rs` or `vectro_py/src/lib.rs`, stop. Implement in `vectro_lib`, expose via public API, call from the outer crate.

### Binary Format Contracts

Every format has a 16-byte magic header. These are immutable once shipped.

| Format | Header | Content | Status |
|---|---|---|---|
| `VECTRO+STREAM1` | `VECTRO+STREAM1\n` | Repeated: `u32 len` + `bincode(Embedding {id, vector: Vec<f32>})` | Production |
| `VECTRO+QSTREAM1` | `VECTRO+QSTREAM1\n` | `u32 tables` + `u32 dim` + `u32 blob_len` + `bincode(Vec<QuantTable>)` + repeated `u32 len` + `bincode((id, qvec: Vec<u8>))` | Production |
| `VECTRO+PQSTREAM1` | `VECTRO+PQSTREAM1\n` | Codebook blob + repeated `u32 len` + `bincode((id, code: Vec<u8>))` — M bytes per vector | Planned v1.2.0 |
| `VECTRO+RQSTREAM1` | `VECTRO+RQSTREAM1\n` | Chained codebooks + residual codes | Planned v2.0.0 |
| `VECTRO+NF4STREAM1` | `VECTRO+NF4STREAM1\n` | NF4 quantization tables + packed 4-bit codes | Planned v2.0.0 |

**Never break format backward compatibility.** A new format requires a new magic header, never a new version field inside an existing header. Format detection is by header bytes, not file extension. The `EmbeddingDataset::load()` path must correctly identify all formats — add a format-detection test for every new format before merging.

**The streaming pipeline exists because datasets exceed available RAM.** Never load an entire dataset into a `Vec<Embedding>` in the hot path. Use the streaming reader, or file a written justification for any bulk-load.

### Dtype & Precision Contracts

```
Uncompressed embeddings:     f32  (Vec<f32> in Embedding.vector)
Scalar-quantized vectors:    u8   (element-wise, per-dimension min/max in QuantTable)
PQ-encoded vectors:          Vec<u8> of length M  (one byte per subspace, M ≤ 256)
Cosine similarity scores:    f32  (higher = more similar; range [-1.0, 1.0])
Quantization table bounds:   f32  min/max per dimension
Search result scores:        f32
```

**No silent upcasting or downcasting.** Dequantized reconstructions are always `f32`. Assert dtypes at every algorithmic boundary in tests.

**Cosine similarity is computed on f32 reconstructions, not on quantized codes directly** — unless ADC (Asymmetric Distance Computation) is explicitly used in a PQ search path, which will be documented in `QSTREAM.md` when implemented.

### Parallelism Model

- `vectro_lib` quantization and search use `rayon` for data-parallel work across embeddings.
- `vectro_cli`'s `compress_stream()` uses `crossbeam-channel` for a reader → worker pool → writer pipeline.
- `vectro_cli`'s `server.rs` uses `tokio` for async I/O; `AppState` is guarded by `Arc<RwLock<...>>`.
- **Never mix rayon and tokio on the same thread pool.** Rayon work called from an async context must be wrapped in `tokio::task::spawn_blocking()`.

### REST API Contracts (`vectro_cli/src/server.rs`)

| Endpoint | Method | Contract |
|---|---|---|
| `/health` | GET | Always returns 200; includes version string |
| `/api/stats` | GET | Returns count, dimensions, index-loaded flag |
| `/api/search` | POST | JSON body `{query: [f32], k: usize}`; returns `[{id, score}]` sorted descending |
| `/api/upload` | POST | Accepts JSONL or bincode stream; loads into AppState |
| `/api/load` | GET | Loads dataset from a server-local file path |

**Validate all inputs at the API boundary.** Enforce max vector dimension, max `k`, and max upload size before any processing. A `query` vector with the wrong dimension must return HTTP 400 with a clear error message — never panic or silently truncate.

**Timeouts are mandatory.** Every endpoint must have a configured request timeout. Search operations on large datasets must not block indefinitely.

---

## 💹 Performance Contracts

Hard baselines measured on M3 16GB, `cargo build --release`. A change that regresses any of these is a **hard stop** — profile with Criterion before merging.

| Operation | Dataset Size | Contract |
|---|---|---|
| STREAM1 compress (scalar) | 10k × 768d | p95 < 500 ms |
| QSTREAM1 compress (scalar quant) | 10k × 768d | p95 < 800 ms |
| Top-10 cosine search (brute force) | 10k × 768d | p95 < 500 μs |
| Top-100 cosine search (brute force) | 10k × 768d | p95 < 2 ms |
| REST API `/api/search` roundtrip | in-memory dataset | p95 < 5 ms |
| `cargo test` (full suite) | all crates | < 30 s |

**Criterion is the timing tool.** Use `cargo bench` for all hot-path measurements. All Criterion benchmarks live in `vectro_lib/benches/quant_bench.rs`. Do not use `std::time::Instant` inside a test for performance claims.

---

## 🔬 Recall & Compression Validation Gates

A compression or search feature is **not complete** until these pass:

**For any new quantization format (PQ, NF4, RQ, etc.):**
1. Compression ratio measured and reported: must beat scalar quant's 4–8× baseline. PQ target: ≥ 16×. RQ target: ≥ 32×.
2. `recall@10` measured against brute-force exact search on a 10k-vector toy dataset: must be ≥ 0.90.
3. Encode/decode roundtrip test: max absolute reconstruction error documented and asserted in a regression snapshot.
4. Format detection test: `EmbeddingDataset::load()` correctly identifies the new format header — wrong format must return a clear error, not corrupt data.

**For HNSW or any ANN index:**
1. `recall@10` ≥ 0.95 on 100k-vector dataset vs brute-force.
2. QPS ≥ 5000 on M3 at the recall@10 ≥ 0.95 operating point.
3. Save/load roundtrip: loaded index produces identical search results as in-memory index.

**Recall regression is a hard stop.** A code change that drops `recall@10` by >2pp on any validated configuration must be fixed before merging.

---

## 🐍 Python Bindings (PyO3 Rules)

- **`vectro_py` is a thin wrapper — no algorithm logic.** Implement in `vectro_lib`, expose, then wrap in `vectro_py`.
- **Zero-copy NumPy borrows are not thread-safe.** `PyReadonlyArray1<f32>` borrows Python's buffer — convert to a `Vec<f32>` before crossing a thread boundary or releasing the GIL.
- **All `#[pymethods]` must return `PyResult<T>` — never panic.** Panics crash the Python interpreter. Use `.map_err(|e| PyRuntimeError::new_err(e.to_string()))` on all `anyhow` errors.
- **Python type stubs (`python/vectro_plus/__init__.pyi`) must be updated in the same commit as any method signature change** in `vectro_py/src/lib.rs`.
- **Build toolchain is Maturin** (`maturin develop` for dev, `maturin build --release --strip` for dist wheels). Do not add new packaging logic to `setup.py` — it will be removed in v1.5.0 when Maturin takes over fully.

---

## 🛣️ Roadmap

Active development targets. Update the status when a phase ships. Check gate criteria before marking complete.

| Phase | Version | Key Deliverable | Gate |
|---|---|---|---|
| Product Quantization | v1.2.0 | `vectro_lib/src/pq.rs`, `VECTRO+PQSTREAM1` format, `--format pq` CLI flag, PQ REST search | recall@10 ≥ 0.90, compression ≥ 16× |
| HNSW ANN Index | v1.3.0 | `vectro_lib/src/hnsw.rs`, `vectro index build/search` CLI, `/api/index/*` REST endpoints | recall@10 ≥ 0.95, QPS ≥ 5000 on M3 |
| Real-World Benchmarks | v1.4.0 | `BENCHMARKS.md`, SIFT1M + GloVe-100d eval, `vectro bench --ground-truth` command, `scripts/run_benchmark.sh` | recall@10 ≥ 0.90 on SIFT1M, reproducible in CI |
| PyPI Distribution | v1.5.0 | Maturin build, manylinux wheels, `pip install vectro-plus`, type stubs, GitHub Actions release workflow | `pip install` from wheel works, Python pytest suite passes in CI |
| Research Port | v2.0.0 | NF4 (`vectro_lib/src/nf4.rs`), RQ (`rq.rs`), AutoQuantize (`auto_quantize.rs`) ported from vectro (Mojo); `--format auto` CLI | δ recall@10 < 5% vs vectro reference on GloVe-100d |

**Algorithm reference for v2.0.0 port:** `~/vectro/` (separate repo, Mojo/Python research library v3.6.0) is the canonical source for NF4, RQ, and AutoQuantize algorithm implementations. Port from there — do not invent divergent implementations.

---

## 🔧 Build, Test & Benchmark Commands

```bash
# Full test suite — must be green before any commit
cargo test

# Per-crate tests
cargo test -p vectro_lib
cargo test -p vectro_cli
cargo test -p vectro_py

# Clippy — zero warnings policy
cargo clippy -- -D warnings

# Criterion benchmarks
cargo bench

# Release build
cargo build --release

# Python bindings (dev install — requires: pip install maturin)
maturin develop

# Python bindings (release wheel)
maturin build --release --strip

# Generate synthetic test data
cargo run --release -p generators --bin generate_embeddings -- --count 10000 --dimensions 768 --output embeddings.jsonl
cargo run --release -p generators --bin generate_themed_embeddings -- --count 5000 --dimensions 128 --output themed.jsonl

# CLI usage
./target/release/vectro compress embeddings.jsonl output.stream1
./target/release/vectro compress embeddings.jsonl output.qstream1 --quantize
./target/release/vectro search "0.1,0.2,..." --dataset output.stream1 --top-k 10
./target/release/vectro bench --save-report --summary
./target/release/vectro serve --port 3000
```

---

## ✅ Ship Gate — Definition of Done (Vectro+)

A feature is **complete** only when ALL of the following are true:

1. `cargo test` passes — zero failing tests across all crates.
2. `cargo clippy -- -D warnings` passes — zero warnings.
3. Performance contracts measured and within spec, or a written exception filed.
4. Recall/compression gate passed for any format or search algorithm change.
5. Binary format spec updated in `QSTREAM.md` for any new format.
6. `CHANGELOG.md` entry written under the correct version heading.
7. `README.md` updated if public API, CLI flags, or Python API changed.
8. For Python binding changes: `python/vectro_plus/__init__.pyi` type stubs updated in the same commit.

---

*End of vectro+ specific rules*
*Owner: wesleyscholl / Konjo AI Research*
*Update this file when architectural contracts change. Never let it drift from the actual implementation.*