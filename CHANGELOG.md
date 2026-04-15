# Changelog

All notable changes to Vectro+ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-04-15

### Added
- `vectro pipeline` CLI subcommand — compress → HNSW index → batch search end-to-end pipeline
- `StreamIter` / `iter_stream()` in `vectro_lib` for lazy, zero-copy VECTRO+STREAM1 file iteration
- `PyStreamIter` PyO3 binding exposing the streaming iterator to Python
- `EmbeddingDataset.stream_from_file(path)` static method on the Python dataset class
- `stream_embeddings(path)` convenience wrapper in `vectro_plus`
- WASM entry points: `cosine_similarity()` and `quantize_batch()` in `vectro_lib/src/wasm.rs`
- `wasm-bindgen` conditional dependency in `vectro_lib/Cargo.toml`
- All `rayon` parallel calls gated with `#[cfg(not(target_arch = "wasm32"))]` sequential fallbacks
- `scripts/convert_glove_to_stream1.py` — converts GloVe `.txt` file to `VECTRO+STREAM1` binary format
- `python/tests/test_streaming.py` — integration tests for the streaming API
- `js/test.js` — Node.js WASM smoke tests
- `StreamIter` class stub in `python/vectro_plus/__init__.pyi`
- `stream_embeddings` type stub in `__init__.pyi`

### Changed
- `vectro_lib/Cargo.toml`: added `[lib] crate-type = ["cdylib", "rlib"]` for `wasm-pack` compatibility
- `vectro_lib/Cargo.toml`: rayon moved to `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`

## [2.0.1] - 2026-04-15

### 🐛 Fixed
- `python/vectro_plus/__init__.py` — `except ImportError` fallback path now defines stub classes (`Embedding`, `EmbeddingDataset`, `SearchIndex`, `QuantizedIndex`) and stub functions (`compress_embeddings`, `analyze_compression_quality`, `benchmark_search_performance`) so that type annotations and pytest collection do not crash with `NameError` when the Rust extension is unavailable (e.g. CI before `maturin develop` runs)
### � Added — v1.4.0: Ground-Truth Benchmarks
- `BENCHMARKS.md` — hardware-stamped benchmark results for the v1.4.0 milestone
  - HNSW M=16 ef_s=50: **recall@10 = 0.920** ✅ (gate ≥ 0.90), 8 066 QPS, 0.12 ms latency on M3 16GB
  - PQ m=8 k=256: 0.218 recall on random synthetic data (expected; documented limitation)
  - Documents PQ recall ≥ 0.90 on structured data is validated by `test_pq_recall_at_10_gate`
- `benchmarks/results/2026-04-14T07-29-40-bench-gt.json` — timestamped result artifact
- `scripts/run_benchmark.sh` — fixed binary name from `vectro` → `vectro_cli`
- `vectro_cli/src/main.rs` — bench-gt synthetic generator restored to deterministic xorshift64
  (random 128-d vectors; gives realistic HNSW recall ≥ 0.98 and honest PQ baseline)

### �🐛 Fixed — CI: Python error-handling tests
- `PySearchIndex::search_vector` now raises `ValueError` when `top_k == 0` or query dimension mismatches index dimension
- `PyQuantizedIndex::search_vector` same validation — consistent behaviour across both index types
- `PySearchIndex` and `PyQuantizedIndex` now store and expose a `dim` field (Python getter)
- `PyQuantizedIndex::from_dataset` and `compress_embeddings` updated to compute and propagate `dim`
- `create_index` and `create_quantized_index` (Python) raise `ValueError` on empty input arrays
- Fixes three `TestErrorHandling` tests that previously received silent empty results instead of exceptions

## [2.0.0] - 2026-04-13

### ✨ Added — v2.0.0: Research Port (NF4, RQ, AutoQuantize)

#### 🧮 vectro_lib — New Quantization Modules
- `vectro_lib/src/nf4.rs` — 4-bit NormalFloat quantization (`Nf4Quantizer`)
  - QLoRA-paper 16-level codebook; compile-time `THRESHOLDS` for O(1) `partition_point` lookup
  - Per-vector abs-max scaling; packed 2 nibbles/byte; ~8× compression; cosine ≥ 0.98 at d=768
  - `decode_single()` for single-record deserialization in `EmbeddingDataset::load()`
- `vectro_lib/src/rq.rs` — Residual Quantizer (`ResidualQuantizer`)
  - Multi-pass Lloyd's k-means on raw (un-normalised) residuals; `rayon`-parallel subspace training
  - Adaptive zero-padding when `dim % m != 0`; ~16–192× compression
- `vectro_lib/src/auto_quantize.rs` — Automatic format selection (`auto_select_format`)
  - Kurtosis-based routing: Gaussian → NF4→RQ→PQ→Scalar; heavy-tailed → PQ→RQ→NF4→Scalar
  - Returns first format meeting `target_cosine ≥ 0.97` AND `target_compression ≥ 8×`
  - `QuantFormat` enum (`Nf4 | Rq | Pq | Scalar | Stream`) with `Display`
  - `AutoQuantizeResult` carries the trained `ProductQuantizer` or `ResidualQuantizer` when selected

#### 🗃️ Format Detection
- `EmbeddingDataset::load()` now detects and decodes `VECTRO+NF4STREAM1\n` and `VECTRO+RQSTREAM1\n` headers
- NF4 stream record layout: `(id, packed: Vec<u8>, scale: f32)` per vector
- RQ stream record layout: `(id, codes: Vec<Vec<u8>>)` per vector (one code slice per pass)

#### 🖥️ CLI — New Formats
- `--format nf4` — encode to NF4 4-bit stream
- `--format rq` — encode to Residual Quantizer stream
- `--format auto` — evaluate and select best format automatically
- `--rq-passes N` — number of residual passes (default 2)
- `--rq-subspaces N` — RQ subspaces per pass (default 8)

#### 🛠 vectro_lib Improvements
- `ProductQuantizer::compression_ratio()` — new method returning `(dim × 4) / m`; used by `auto_quantize`

### 📋 Gate Results
- NF4 cosine similarity ≥ 0.98 at dim=768 (contract: ≥ 0.98) ✅
- RQ cosine similarity ≥ 0.75 on 100 training vectors (contract: ≥ 0.75) ✅
- RQ compression ratio ≥ 30× (contract: ≥ 16×) ✅
- δ recall@10 vs vectro Python reference: pending GloVe-100d evaluation run

## [1.5.0] - 2026-04-13

### ✨ Added — v1.5.0: PyPI Distribution

#### 📦 Maturin Build System
- `pyproject.toml` replaces `setup.py` as the authoritative build configuration
- Build backend: `maturin>=1.5,<2.0`; compiled extension lands at `vectro_plus.vectro_py`
- `python-source = "python"` — all Python source in `python/` is packaged automatically
- Supports Python 3.8–3.12; `abi3-py38` stable ABI

#### 🐍 Type Stubs
- `python/vectro_plus/__init__.pyi` — comprehensive PEP 561 stubs for the full public API
  - `Embedding`, `EmbeddingDataset`, `SearchIndex`, `QuantizedIndex`
  - `compress_embeddings`, `analyze_compression_quality`, `benchmark_search_performance`
  - `VectroConfig`, `create_index`, `create_quantized_index`, `search_similar`, `batch_search`
  - `load_embeddings_from_array`, `generate_quality_report`, `save_index`, `load_index`
  - `info`, `version`, `__version__`, `__author__`, `__description__`

#### ⚙️ GitHub Actions Workflows
- `.github/workflows/ci.yml` — runs on every push/PR:
  - `cargo test` + `cargo clippy -- -D warnings` on ubuntu-latest
  - `maturin develop` + `pytest python/tests/` on ubuntu-latest × macos-latest × Python 3.9/3.11/3.12
- `.github/workflows/release.yml` — triggered on `v*` tags:
  - Builds manylinux x86_64 + aarch64 wheels via `PyO3/maturin-action@v1`
  - Builds macOS arm64 + x86_64 wheels
  - Builds source distribution (`maturin sdist`)
  - Publishes all artifacts to PyPI via OIDC trusted publishing

### 🗑️ Removed
- `setup.py` — superseded by `pyproject.toml` + Maturin

### 📋 Changed
- `vectro_lib`, `vectro_cli`, `vectro_py` bumped to version 1.5.0
- `__version__` in the Rust extension updated to `"1.5.0"`
- `__version__` fallback in `__init__.py` updated to `"1.5.0"`

---

## [1.4.0] - 2026-04-13

### ✨ Added — v1.4.0: Real-World Benchmarks

#### 📊 New Command: `vectro bench-gt`
- Ground-truth recall@k and QPS evaluation for all search algorithms
- Evaluates brute-force (exact), HNSW, and PQ over the same query set
- Synthetic data generation: `--vectors N --dim D` with deterministic xorshift64 PRNG (seed `0xdeadbeef_cafebabe`) — fully CI-reproducible, no randomness
- External dataset support: `--dataset <path>` (any Vectro binary format or JSONL)
- `--save-report` writes timestamped JSON to `benchmarks/results/`
- Soft recall gates: recall@10 ≥ 0.90 for HNSW and PQ (warns on failure, full table always printed)

#### 📐 New Library Function: `vectro_lib::recall_at_k`
- `pub fn recall_at_k(exact: &[String], approx: &[String], k: usize) -> f64`
- Set-intersection formula: `|approx_top_k ∩ exact_top_k| / k`
- Includes doc-test

#### 📝 New Files
- `scripts/run_benchmark.sh` — build release binary and run `bench-gt` with configurable params
- `BENCHMARKS.md` — methodology, parameter documentation, result tables, and JSON report schema

### 📋 Changed
- `vectro_lib` version bumped to 1.4.0
- `vectro_cli` version bumped to 1.4.0

---

## [1.3.0] - 2026-04-13

### ✨ Added — v1.3.0: HNSW ANN Index

#### 🔍 New Module: `vectro_lib/src/hnsw.rs`
- **Hierarchical Navigable Small World** (HNSW) ANN index — `HnswIndex` public struct
- `HnswIndex::build(data, m, ef_construction, ef_search)` — batch build, O(N·M·log N) average
- `HnswIndex::insert(&mut self, embedding)` — incremental single-vector insert
- `HnswIndex::search(query, k)` → `Vec<(String, f32)>` sorted descending by cosine similarity
- `HnswIndex::search_with_ef(query, k, ef)` — explicit beam-width override at query time
- `HnswIndex::save(path)` / `HnswIndex::load(path)` — bincode serialization; save/load roundtrip verified
- Cosine similarity via L2-normalization on insert; inner product of unit vectors
- Custom xorshift64 PRNG for reproducible layer assignments — no external `rand` dep
- Default parameters: M=16, ef_construction=200, ef_search=50
- **Recall gate**: `test_recall_at_10_gate` enforces recall@10 ≥ 0.95 (1000 vectors, dim=64, 100 queries)
- `pub use hnsw::HnswIndex` re-exported from `vectro_lib` crate root

#### 🖥️ CLI
- `vectro index build <dataset> <output>` — build and persist HNSW index to disk
  - `--m N` (default 16), `--ef-construction N` (default 200), `--ef-search N` (default 50)
- `vectro index search <query> --index <path>` — ANN search against a saved index
  - `--top-k N` (default 10), `--ef N` to override search beam width at query time

#### 🌐 REST API
- `POST /api/index/build` — build HNSW from currently loaded embeddings; optional `{"m", "ef_construction", "ef_search"}` body
- `POST /api/index/search` — ANN search; same `{"query", "k"}` body as `POST /api/search`

#### 📐 Benchmarks
- `hnsw_build_1k` and `hnsw_search_1k` Criterion benchmarks added to `quant_bench.rs`

### 📋 Changed
- `vectro_lib` and `vectro_cli` bumped to version `1.3.0`

---

## [1.2.0] - 2025-07-08

### ✨ Added — v1.2.0: Product Quantization

#### 🗜️ New Format: `VECTRO+PQSTREAM1`
- **Product Quantization (PQ)** compression module — `vectro_lib/src/pq.rs`
- `ProductQuantizer` struct: train/encode/decode/search in a single serializable type
- **Compression**: ≥ 16× vs raw f32 (e.g. 768d m=8 → 32× compression)
- **Recall gate**: recall@10 ≥ 0.90 enforced by test `test_pq_recall_at_10_gate`
- **ADC search**: Asymmetric Distance Computation for sub-linear approximate cosine search
- K-means training: deterministic centroid init, rayon-parallel per-subspace Lloyd's
- Vectors L2-normalized before encode/train; centroids L2-normalized after each update
- `EmbeddingDataset::load()` reads `VECTRO+PQSTREAM1` files transparently

#### 🖥️ CLI
- `vectro compress --format pq` writes PQSTREAM1 output
- `--pq-subspaces N` (default 8): encoded bytes per vector; must divide `dim`
- `--pq-centroids K` (default 256): centroids per subspace (1–256)
- `--format scalar` / `--format stream` remain available; `--quantize` kept for back-compat

#### 📐 Benchmarks
- `pq_encode`, `pq_decode`, `pq_adc_topk` Criterion benchmarks in `quant_bench.rs`

### 📋 Changed
- `QSTREAM.md` — added `VECTRO+PQSTREAM1` format specification

## [1.1.0] - 2024-12-19

### ✨ Added - Python Bindings & Enhanced APIs

#### 🐍 Major Feature: Python Integration
- **Native Python bindings** using PyO3 for zero-copy NumPy integration
- **Complete Python package** (`vectro_plus`) with high-level API
- **Comprehensive Python test suite** with quality analysis tools
- **Performance benchmarking utilities** directly from Python
- **Example scripts and documentation** for Python workflows

#### 🔧 Python API Components
- `PyEmbedding`, `PyEmbeddingDataset` - Core data structures with Pythonic interface
- `PySearchIndex`, `PyQuantizedIndex` - Fast search indices with NumPy integration
- `compress_embeddings()` - One-line compression and indexing
- `analyze_compression_quality()` - Quality metrics and compression analysis
- `benchmark_search_performance()` - Performance profiling and timing tools

#### 📦 Build & Installation Infrastructure
- **Advanced setup.py** with Cargo extension building
- **Automatic Rust compilation** during Python package installation
- **Cross-platform support** for Python packaging on macOS/Linux/Windows
- **Build helper scripts** for streamlined development workflow
- **PyO3 configuration** optimized for performance and memory safety

### 🔧 Enhanced Features
- **Upgraded test coverage** from 89 to 93 comprehensive tests
- **Enhanced error handling** with Python-friendly error messages  
- **Improved documentation** with extensive Python integration examples
- **Version synchronization** across all crates and Python package

### 🐛 Fixed & Improved
- **API consistency** between Rust core and Python wrapper interfaces
- **Memory management** optimized for Python/Rust interoperability  
- **Type safety** with comprehensive PyO3 wrapper implementations
- **ID-to-index mapping** for efficient search result translation

### 📚 Documentation & Examples
- **Comprehensive Python examples** integrated into README
- **Step-by-step installation guide** for Python bindings
- **Quality analysis tutorials** showing compression trade-offs
- **Performance benchmarking guide** with interpretation examples

### ⚡ Technical Achievements
- **Zero-copy operations** between NumPy arrays and Rust data structures
- **Efficient serialization** using PyO3 and ndarray integration
- **Thread-safe Python bindings** supporting Python's GIL requirements
- **Memory-efficient implementations** with proper resource management

**Migration Notes:**
- Existing Rust API unchanged - full backward compatibility
- New Python package requires PyO3 and NumPy dependencies
- Python API mirrors Rust functionality with Pythonic conventions

---

## [Unreleased]

### Added
- **Expanded Test Coverage** - Increased from 68.64% to 77.64% (+9%)
  - Added 55 new unit tests for helper functions
  - Added 6 new integration tests for compression workflows
  - Comprehensive tests for delta calculation, JSON parsing, and data loading
  - Total test count: 93 tests (all passing)
- **Enhanced Test Documentation** - Updated TEST_COVERAGE_REPORT.md with latest metrics
- **Helper Function Tests** - Complete coverage for:
  - Delta percentage calculations
  - JSON parsing utilities  
  - Benchmark name extraction
  - Format delta HTML output
  - Dataset loading with fallbacks

### Changed
- Improved test reliability with comprehensive edge case coverage
- Better code quality metrics for production deployment
- Enhanced testing infrastructure for future maintainability

## [1.0.1] - 2025-11-04

### Added
- **Project Status & Roadmap** - Added comprehensive status section to README
  - v1.1 roadmap: Advanced quantization, GPU acceleration, Python bindings
  - v1.2 roadmap: Distributed search, real-time streaming, cloud deployment
  - v2.0 roadmap: Auto-tuning, federated learning, enterprise features
- **Contribution Guidelines** - Enhanced community participation guidance
- **Next Steps Documentation** - Clear guidance for developers, data engineers, and researchers

### Changed
- Updated README with production-ready status badges
- Enhanced documentation structure with roadmap sections
- Improved feature documentation and examples

## [1.0.0] - 2025-10-29

### 🎉 Production Ready Release

Vectro+ has achieved **production-ready status** with comprehensive features, optimized performance, and complete documentation.

### Highlights

- ✅ **Complete Feature Set** - Compression, quantization, search, web UI, REST API
- ✅ **High Performance** - Parallel processing, SIMD optimizations, streaming support
- ⚡ **Fast Search** - Sub-millisecond cosine similarity queries
- 📦 **Efficient Compression** - 75-90% size reduction with quantization
- 🌐 **Web Dashboard** - Beautiful interactive UI with real-time search
- 🔌 **REST API** - Production-ready HTTP endpoints
- 📊 **Benchmarking** - Criterion integration with HTML reports
- 🎨 **Beautiful CLI** - Progress bars, colored output, streaming logs
- 📖 **Complete Documentation** - Comprehensive guides and examples

### Performance Benchmarks

**Compression Performance:**
- 10K × 128d: 180ms (5 MB dataset)
- 100K × 768d: 3.2s (300 MB dataset)
- 1M × 768d: 34s (3 GB dataset)

**Search Performance:**
- Top-10 search: 45-156 μs
- Top-100 search: 420 μs - 1.8 ms
- Parallel indexing enabled

**Compression Ratios:**
- Regular format (STREAM1): Original size preserved
- Quantized format (QSTREAM1): 75-90% size reduction
- Quality: Minimal accuracy loss (<0.5%)

### Features

#### Core Library (vectro_lib)

- **Embedding Management**
  - `Embedding` struct with ID and vector data
  - Support for arbitrary dimensions
  - Efficient memory layout

- **Dataset Operations**
  - `Dataset` struct for collections of embeddings
  - Parallel processing with Rayon
  - Batch operations

- **Search Index**
  - `SearchIndex` for fast similarity search
  - Cosine similarity computation
  - Top-K results with configurable K
  - Batch query support

- **Quantization**
  - `QuantizedIndex` for compressed storage
  - Scalar quantization (Int8)
  - Per-dimension quantization tables
  - Reconstruction with minimal error

- **Binary Formats**
  - STREAM1: Full precision format
  - QSTREAM1: Quantized compressed format
  - Streaming read/write support
  - Bincode serialization

#### CLI Application (vectro_cli)

- **Compress Command**
  - Stream large datasets from JSONL
  - Parallel pipeline processing
  - Progress bars with ETA
  - Optional quantization flag
  - Multiple format support

- **Search Command**
  - Load compressed datasets
  - Parse query vectors from CSV
  - Top-K similarity search
  - Formatted results output

- **Benchmark Command**
  - Criterion integration
  - HTML report generation
  - Summary tables with delta tracking
  - Save reports to custom locations
  - Open reports in browser

- **Serve Command** (NEW in 1.0.0)
  - Web server with Axum framework
  - REST API endpoints
  - Interactive dashboard UI
  - Real-time search
  - Drag-and-drop upload
  - CORS support
  - Health checks

#### Web UI Features

- 📊 **Dashboard**
  - Real-time statistics
  - Dataset info display
  - Performance metrics
  - Beautiful gradient design

- 🔍 **Search Interface**
  - Interactive query input
  - Instant results
  - Top-K configuration
  - Result visualization

- 📤 **Dataset Management**
  - Upload embeddings
  - Load compressed datasets
  - Format validation
  - Progress tracking

### Added

- **Web Server (`serve` command)**
  - HTTP server with Axum
  - REST API for search and stats
  - Interactive web dashboard
  - Real-time search interface
  - Static file serving
  - CORS support

- **REST API Endpoints**
  - `GET /health` - Health check
  - `GET /api/stats` - Dataset statistics
  - `POST /api/search` - Search embeddings
  - `POST /api/upload` - Upload datasets
  - `POST /api/load` - Load compressed files

- **Enhanced CLI**
  - Progress bars with `indicatif`
  - Colored output
  - Streaming logs
  - ETA calculations

- **Benchmark Improvements**
  - HTML report auto-generation
  - Summary tables in terminal
  - Delta tracking vs baseline
  - Custom report locations

- **Documentation**
  - DEMO.md - Comprehensive examples
  - QSTREAM.md - Binary format specification
  - QUICKSTART_VIDEO.md - Video recording guide
  - VIDEO_DEMO.md - Presentation scripts
  - VISUAL_GUIDE.md - Web UI walkthrough

### Changed

- **Parallel Processing**
  - Multi-threaded compression pipeline
  - Rayon-based parallelism
  - Configurable worker threads
  - Optimal CPU utilization

- **Error Handling**
  - Comprehensive error types with `anyhow`
  - Graceful error messages
  - User-friendly CLI feedback

- **Performance Optimizations**
  - SIMD operations where applicable
  - Zero-copy operations
  - Efficient memory allocation
  - Streaming I/O for large files

### Architecture

```
vectro-plus/
├── vectro_lib/          # Core library
│   ├── src/lib.rs       # Embedding, Dataset, SearchIndex, QuantizedIndex
│   └── benches/         # Criterion benchmarks
├── vectro_cli/          # CLI application
│   ├── src/
│   │   ├── lib.rs       # Compression pipeline
│   │   └── main.rs      # CLI commands + web server
│   └── tests/           # Integration tests
└── docs/                # Documentation
```

### Testing

**Comprehensive Test Coverage: 77.18%** (504/653 lines)

- ✅ **vectro_lib: 100% coverage** (176/176 lines) - PERFECT
- ✅ **vectro_cli/lib.rs: 100% coverage** (129/129 lines) - PERFECT
- ✅ **server.rs: 92.4% coverage** (97/105 lines) - EXCELLENT
- ✅ **main.rs: 42.0% coverage** (102/243 lines) - Infrastructure-limited

**Test Suite:**
- **89 Total Tests** (all passing)
  - 71 Unit Tests
  - 18 Integration Tests
- Core library tests
- CLI integration tests
- Quantization roundtrip tests
- Search accuracy tests
- Format compatibility tests
- Server integration tests
- Bench command infrastructure tests

**Test Categories:**
```
vectro_lib:              18 unit tests
vectro_cli/lib.rs:        4 unit tests
vectro_cli/main.rs:      49 unit tests
integration_cli:          5 tests
integration_compress:     1 test
integration_quantize:     1 test
integration_bench:        8 tests
integration_server:       3 tests
Total:                   89 tests passing ✅
```

### Dependencies

- **Core:**
  - `ndarray` - N-dimensional arrays
  - `rayon` - Data parallelism
  - `serde` + `bincode` - Serialization
  - `nalgebra` - Linear algebra
  - `anyhow` - Error handling

- **CLI:**
  - `clap` - Command-line parsing
  - `indicatif` - Progress bars
  - `serde_json` - JSON parsing
  - `csv` - CSV parsing

- **Web:**
  - `axum` - Web framework
  - `tokio` - Async runtime
  - `tower-http` - HTTP middleware

- **Benchmarking:**
  - `criterion` - Statistical benchmarks

### Use Cases

Ready for production use in:
- 🗄️ **Vector Database Optimization** - Compress embeddings by 75%+
- 🤖 **RAG Pipeline Acceleration** - Faster retrieval with smaller indexes
- 🔍 **Semantic Search** - Sub-millisecond similarity queries
- 📱 **Edge Deployment** - Smaller model footprints
- ☁️ **Cloud Cost Reduction** - 75-90% storage savings
- 🌐 **Web Applications** - REST API for integration

### Breaking Changes

None - initial 1.0.0 release.

### Migration Guide

This is the first stable release. Installation:

```bash
# Build from source
git clone https://github.com/yourorg/vectro-plus
cd vectro-plus
cargo build --release

# Binary location
./target/release/vectro_cli
```

---

## [0.1.0] - 2025-10-15

### Initial Development Release

First working version of Vectro+ with core functionality.

### Features

- Basic compression pipeline
- STREAM1 format support
- Quantization (QSTREAM1)
- Cosine similarity search
- CLI with compress and search commands
- Demo scripts
- Basic documentation

### Performance

- Functional compression
- Search working
- Single-threaded processing
- Basic progress indicators

---

## Future Releases

### [1.1.0] - Planned

**Enhanced Performance:**
- GPU acceleration research
- Advanced SIMD optimizations
- Distributed processing support

**Additional Features:**
- Python bindings
- Additional quantization methods (PQ, OPQ)
- Approximate nearest neighbor algorithms
- Streaming search support

**Cloud Integration:**
- Docker containers
- Kubernetes deployment guides
- Cloud storage integration (S3, GCS, Azure)

### [1.2.0] - Planned

**Ecosystem:**
- Vector database integrations (Qdrant, Weaviate, Pinecone)
- LangChain/LlamaIndex adapters
- OpenAI embedding format support
- Hugging Face integration

**Monitoring:**
- Prometheus metrics
- Distributed tracing
- Performance profiling tools

---

## Version History

- **1.0.0** (2025-10-29) - Production ready
- **0.1.0** (2025-10-15) - Initial development release

---

## Links

- **Homepage**: https://github.com/yourorg/vectro-plus
- **Documentation**: See README.md and docs/
- **Issues**: https://github.com/yourorg/vectro-plus/issues

---

**For detailed usage examples, see [DEMO.md](DEMO.md) and [QUICKSTART_VIDEO.md](QUICKSTART_VIDEO.md).**
