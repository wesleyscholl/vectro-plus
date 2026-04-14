//! Vectro+ CLI - Command-line interface for embedding compression and search
//!
//! # Examples
//!
//! ```no_run
//! // Compress embeddings
//! // vectro compress input.jsonl output.bin
//!
//! // Search for similar vectors
//! // vectro search "1.0,2.0,3.0" --top-k 10 --dataset output.bin
//!
//! // Run benchmarks
//! // vectro bench --summary --open-report
//!
//! // Start web server
//! // vectro serve --port 8080
//! ```

use clap::{Parser, Subcommand};
use vectro_cli::{compress_stream, compress_pq};

use serde_json::Value;

type BenchRow = (String, Option<f64>, Option<f64>, Option<String>);

pub mod server;

#[derive(Parser)]
#[command(name = "vectro")]
#[command(about = "Vectro+ — Rust embedding compressor & search tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress {
        input: String,
        output: String,
        /// Compression format: `stream` (raw f32), `scalar` (u8 per-dim), or `pq` (Product Quantization).
        /// Overrides `--quantize` when specified.
        #[arg(long, default_value = "stream")]
        format: String,
        #[arg(long, default_value_t = false)]
        /// Shorthand for `--format scalar` (kept for backward compatibility).
        quantize: bool,
        /// Number of PQ subspaces (bytes per encoded vector). Used with `--format pq`.
        /// Must divide the vector dimension. Lower → higher compression, lower recall.
        #[arg(long, default_value_t = 8usize)]
        pq_subspaces: usize,
        /// Centroids per PQ subspace. Must be ≤ 256. Used with `--format pq`.
        #[arg(long, default_value_t = 256usize)]
        pq_centroids: usize,
    },
    /// Run library benchmarks (uses the `vectro_lib` bench harness).
    /// Streams benchmark output and shows a spinner while running.
    Bench {
        /// Save the Criterion HTML report to this path (directory). If omitted, report will remain under target/criterion.
        #[arg(long)]
        save_report: Option<String>,
        /// Open the HTML report after generation (macOS `open` is used).
        #[arg(long, default_value_t = false)]
        open_report: bool,
        /// Print a short JSON summary (median, mean) from Criterion's JSON output.
        #[arg(long, default_value_t = true)]
        summary: bool,
        /// Directory to copy the report into when using --save-report (default: current dir)
        #[arg(long)]
        report_dir: Option<String>,
        /// Extra arguments to pass to cargo bench (e.g., "--bench cosine_bench")
        #[arg(long)]
        bench_args: Option<String>,
    },
    Search {
        query: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
        /// Path to dataset (bincode). If omitted, uses built-in toy dataset.
        #[arg(long)]
        dataset: Option<String>,
    },
    Serve {
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    /// Build or search a persistent HNSW approximate nearest-neighbor index.
    ///
    /// Example: `vectro index build embeddings.jsonl index.hnsw`
    /// Example: `vectro index search "0.1,0.2,0.3" --index index.hnsw --top-k 5`
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },
    /// Ground-truth recall & QPS evaluation for brute-force, HNSW, and PQ search.
    ///
    /// Uses a loaded dataset (or generates synthetic data) and evaluates recall@k and
    /// queries-per-second for each algorithm against exact brute-force results.
    ///
    /// Example: `vectro bench-gt --vectors 10000 --dim 128 --queries 100 --save-report`
    /// Example: `vectro bench-gt --dataset embeddings.stream1 --k 10`
    #[command(name = "bench-gt")]
    BenchGt {
        /// Path to an existing dataset file (JSONL or any Vectro binary format).
        /// If omitted, synthetic data is generated using `--vectors` and `--dim`.
        #[arg(long)]
        dataset: Option<String>,
        /// Number of synthetic vectors to generate (ignored if `--dataset` is provided).
        #[arg(long, default_value_t = 10000usize)]
        vectors: usize,
        /// Dimension of synthetic vectors (ignored if `--dataset` is provided).
        #[arg(long, default_value_t = 128usize)]
        dim: usize,
        /// Number of random queries to evaluate.
        #[arg(long, default_value_t = 100usize)]
        queries: usize,
        /// k for recall@k metric.
        #[arg(long, default_value_t = 10usize)]
        k: usize,
        /// Save a timestamped JSON results file to `benchmarks/results/`.
        #[arg(long, default_value_t = false)]
        save_report: bool,
        /// Suppress per-query output; only print the final summary table.
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },
}

/// Sub-commands for the `vectro index` family.
#[derive(Subcommand)]
enum IndexAction {
    /// Build an HNSW index from a dataset file and write it to disk.
    ///
    /// Example: `vectro index build embeddings.jsonl index.hnsw --m 16 --ef-construction 200`
    Build {
        /// Input dataset: JSONL or any Vectro binary format (.stream1, .qstream1, .pqstream1).
        dataset: String,
        /// Output path for the HNSW index (bincode).
        output: String,
        /// Maximum connections per node per layer.  Higher = better recall, slower build.
        #[arg(long, default_value_t = 16usize)]
        m: usize,
        /// Dynamic candidate list size during construction.  Higher = better recall, slower build.
        #[arg(long, default_value_t = 200usize)]
        ef_construction: usize,
        /// Default candidate list size for queries against this index.
        #[arg(long, default_value_t = 50usize)]
        ef_search: usize,
    },
    /// Search an existing HNSW index for approximate nearest neighbors.
    ///
    /// Example: `vectro index search "0.1,0.2" --index index.hnsw --top-k 10`
    Search {
        /// Query vector as comma-separated floats (e.g. \"0.1,0.2,0.3\").
        query: String,
        /// Path to the HNSW index file produced by `vectro index build`.
        #[arg(long)]
        index: String,
        /// Number of approximate nearest neighbors to return.
        #[arg(short, long, default_value_t = 10usize)]
        top_k: usize,
        /// Override the search beam width (higher = more recall, slower).
        #[arg(long)]
        ef: Option<usize>,
    },
}

// Wrapper functions for testability
fn execute_compress_command(
    input: &str,
    output: &str,
    format: &str,
    quantize: bool,
    pq_subspaces: usize,
    pq_centroids: usize,
) -> anyhow::Result<usize> {
    if format == "pq" {
        compress_pq(input, output, pq_subspaces, pq_centroids)
    } else if format == "scalar" || quantize {
        compress_stream(input, output, true)
    } else {
        compress_stream(input, output, false)
    }
}

fn execute_serve_command(port: u16) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(async {
        server::serve(port).await
    })
}

fn parse_query_string(query: &str) -> Vec<f32> {
    query
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

fn load_dataset_or_default(dataset_path: Option<&str>) -> Vec<vectro_lib::Embedding> {
    use std::path::Path;
    
    if let Some(path) = dataset_path {
        if let Ok(ds) = vectro_lib::EmbeddingDataset::load(path) { return ds.embeddings }
    } else if Path::new("./dataset.bin").exists() {
        if let Ok(ds) = vectro_lib::EmbeddingDataset::load("./dataset.bin") {
            return ds.embeddings;
        }
    }
    
    // Default toy dataset
    vec![
        vectro_lib::Embedding::new("one", vec![1.0, 0.0]),
        vectro_lib::Embedding::new("two", vec![0.0, 1.0]),
        vectro_lib::Embedding::new("three", vec![0.707, 0.707]),
    ]
}

fn execute_search_command(query: &str, top_k: usize, dataset: Option<&str>) -> Vec<(String, f32)> {
    let vec = parse_query_string(query);
    let embeddings = load_dataset_or_default(dataset);
    let idx = vectro_lib::search::SearchIndex::from_dataset(&embeddings);
    idx.top_k(&vec, top_k)
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output, format, quantize, pq_subspaces, pq_centroids } => {
            execute_compress_command(&input, &output, &format, quantize, pq_subspaces, pq_centroids)?;
        }
        Commands::Bench { save_report, open_report, summary, report_dir: _, bench_args } => {
            // Run cargo bench for vectro_lib and stream output. Show a spinner while running.
            use indicatif::{ProgressBar, ProgressStyle};
            use std::process::Command;
            use std::io::{BufRead, BufReader};
            use std::thread;
            use std::fs;
            use std::path::PathBuf;

            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            pb.set_message("running benches...");

            let mut cmd = build_bench_command(bench_args.as_deref());
            let mut child = cmd.spawn().expect("failed to spawn cargo bench");

            // stream stdout
            if let Some(out) = child.stdout.take() {
                let pb_out = pb.clone();
                thread::spawn(move || {
                    let reader = BufReader::new(out);
                    for line in reader.lines().map_while(Result::ok) {
                        pb_out.println(line);
                    }
                });
            }

            // stream stderr
            if let Some(err) = child.stderr.take() {
                let pb_err = pb.clone();
                thread::spawn(move || {
                    let reader = BufReader::new(err);
                    for line in reader.lines().map_while(Result::ok) {
                        pb_err.println(line);
                    }
                });
            }

            let status = child.wait().expect("bench wait failed");
            pb.finish_and_clear();
            if !status.success() {
                eprintln!("bench failed: {:?}\n(bench output above)", status);
            } else {
                // After success, optionally locate Criterion report and copy/open it
                let crit_dir = PathBuf::from("target/criterion");
                if crit_dir.exists() {
                    if summary {
                        // parse JSON summaries in target/criterion/*/new/*.json and present a clean table
                        if let Ok(entries) = fs::read_dir(&crit_dir) {
                            let mut rows: Vec<BenchRow> = Vec::new();
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.is_dir() {
                                    let new_dir = p.join("new");
                                    if new_dir.exists() {
                                        if let Ok(it) = fs::read_dir(&new_dir) {
                                            for j in it.flatten() {
                                                let jp = j.path();
                                                if jp.extension().map(|s| s == "json").unwrap_or(false) {
                                                    if let Ok(txt) = fs::read_to_string(&jp) {
                                                        if let Ok(json) = serde_json::from_str::<Value>(&txt) {
                                                            let med = get_estimate(&json, "median");
                                                            let mean = get_estimate(&json, "mean");
                                                            let unit = find_string_in_json(&json, "unit");
                                                            // Use benchmark name if available, fallback to filename
                                                            let name = get_bench_name(&json)
                                                                .unwrap_or_else(|| jp.file_stem()
                                                                    .and_then(|s| s.to_str())
                                                                    .unwrap_or("unknown")
                                                                    .to_string());
                                                            rows.push((name, med, mean, unit));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if !rows.is_empty() {
                                // try to load previous history for deltas
                                let history_path = PathBuf::from(".bench_history.json");
                                let history = load_bench_history(&history_path);

                                // print pretty table
                                println!("\nBenchmark summaries:");
                                // header (include delta vs previous run)
                                println!("\x1b[1m{:<60} {:>12} {:>12} {:>8} {:>8}\x1b[0m", "benchmark", "median", "mean", "unit", "delta");
                                for (f, med, mean, unit) in &rows {
                                    let med_s = med.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
                                    let mean_s = mean.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
                                    let unit_s = unit.clone().unwrap_or_else(|| "".to_string());
                                    let delta_s = format_delta(*med, &history, f);
                                    println!("{:<60} {:>12} {:>12} {:>8} {:>8}", f, med_s, mean_s, unit_s, delta_s);
                                }

                                // update history with latest medians
                                let mut new_hist: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
                                for (f, med, _mean, _unit) in &rows {
                                    if let Some(m) = med { new_hist.insert(f.clone(), *m); }
                                }
                                let _ = save_bench_history(&history_path, &new_hist);

                                // Generate HTML summary in criterion dir
                                let html_summary = generate_html_summary(&rows, &history);
                                let summary_path = crit_dir.join("vectro_summary.html");
                                if let Err(e) = fs::write(&summary_path, html_summary) {
                                    eprintln!("Warning: couldn't write HTML summary: {}", e);
                                } else {
                                    println!("\n📊 HTML summary saved to: {}", summary_path.display());
                                }
                            }
                        }
                    }

                    if let Some(dest) = save_report {
                        let dest_dir = PathBuf::from(dest);
                        let ts = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                            Ok(d) => format!("{}", d.as_secs()),
                            Err(_) => "ts".to_string(),
                        };
                        let target_copy = dest_dir.join(format!("criterion-report-{}", ts));
                        let _ = fs::create_dir_all(&target_copy);
                        let _ = copy_dir_all(&crit_dir, &target_copy);
                        println!("Saved Criterion report to {}", target_copy.display());
                        if open_report {
                            let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                            let _ = Command::new(opener).arg(target_copy.join("index.html")).spawn();
                        }
                    } else if open_report {
                        // try to find an index.html anywhere under crit_dir
                        let mut index_opt: Option<PathBuf> = None;
                        // simple recursive search
                        let mut stack: Vec<PathBuf> = vec![crit_dir.clone()];
                        while let Some(p) = stack.pop() {
                            if let Ok(entries) = std::fs::read_dir(&p) {
                                for en in entries.flatten() {
                                    let pp = en.path();
                                    if pp.is_dir() { stack.push(pp); }
                                    else if pp.file_name().and_then(|s| s.to_str()) == Some("index.html") {
                                        index_opt = Some(pp);
                                        break;
                                    }
                                }
                            }
                            if index_opt.is_some() { break; }
                        }
                        if let Some(idx) = index_opt {
                            let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                            let _ = Command::new(opener).arg(idx).spawn();
                        }
                    }
                }
            }
        }
        Commands::Search { query, top_k, dataset } => {
            let results = execute_search_command(&query, top_k, dataset.as_deref());
            for (i, (id, score)) in results.into_iter().enumerate() {
                println!("{}. {} -> {:.6}", i + 1, id, score);
            }
        }
        Commands::Serve { port } => {
            execute_serve_command(port)?;
        }
        Commands::Index { action } => match action {
            IndexAction::Build { dataset, output, m, ef_construction, ef_search } => {
                use vectro_lib::hnsw::HnswIndex;
                let ds = vectro_lib::EmbeddingDataset::load(&dataset)
                    .map_err(|e| { eprintln!("Error loading dataset: {}", e); e })?;
                println!("Building HNSW index: {} vectors, M={}, ef_construction={}, ef_search={}",
                    ds.len(), m, ef_construction, ef_search);
                let hnsw = HnswIndex::build(&ds.embeddings, m, ef_construction, ef_search);
                hnsw.save(&output)
                    .map_err(|e| { eprintln!("Error saving index: {}", e); e })?;
                println!("Index saved → {} ({} vectors)", output, hnsw.len());
            }
            IndexAction::Search { query, index, top_k, ef } => {
                use vectro_lib::hnsw::HnswIndex;
                let mut hnsw = HnswIndex::load(&index)
                    .map_err(|e| { eprintln!("Error loading index: {}", e); e })?;
                let vec = parse_query_string(&query);
                if vec.is_empty() {
                    eprintln!("Error: could not parse query vector from {:?}", query);
                    std::process::exit(1);
                }
                if let Some(ef_val) = ef {
                    hnsw.ef_search = ef_val;
                }
                let results = hnsw.search(&vec, top_k);
                if results.is_empty() {
                    println!("No results (index may be empty or query dimension mismatch).");
                } else {
                    for (i, (id, score)) in results.iter().enumerate() {
                        println!("{:3}. {} (score: {:.6})", i + 1, id, score);
                    }
                }
            }
        },
        Commands::BenchGt { dataset, vectors, dim, queries, k, save_report, quiet } => {
            execute_bench_gt(dataset.as_deref(), vectors, dim, queries, k, save_report, quiet)?;
        }
    }

    Ok(())
}

/// Ground-truth recall & QPS evaluation for brute-force, HNSW, and PQ search.
fn execute_bench_gt(
    dataset_path: Option<&str>,
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
    save_report: bool,
    quiet: bool,
) -> anyhow::Result<()> {
    use vectro_lib::{Embedding, HnswIndex, ProductQuantizer, search::SearchIndex, recall_at_k};
    use std::time::Instant;

    // ── 1. Load or generate dataset ─────────────────────────────────────────
    let embeddings: Vec<Embedding> = if let Some(path) = dataset_path {
        if !quiet { println!("Loading dataset from {}…", path); }
        vectro_lib::EmbeddingDataset::load(path)?.embeddings
    } else {
        if !quiet { println!("Generating {} × {}-d synthetic vectors…", n_vectors, dim); }
        // Deterministic xorshift64 — same seed guarantees reproducible CI runs.
        let mut state: u64 = 0xdeadbeef_cafebabe;
        let mut next = move || -> f32 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        (0..n_vectors)
            .map(|i| Embedding::new(format!("v{}", i), (0..dim).map(|_| next()).collect()))
            .collect()
    };

    let n = embeddings.len();
    let d = embeddings.first().map(|e| e.vector.len()).unwrap_or(0);
    if n == 0 || d == 0 {
        anyhow::bail!("Dataset is empty or has zero-dimensional vectors.");
    }
    let effective_queries = n_queries.min(n);
    let effective_k = k.min(n);

    if !quiet {
        println!("Dataset: {} vectors × {} dimensions", n, d);
        println!("Queries: {}  k: {}", effective_queries, effective_k);
        println!();
    }

    // Pick query vectors: evenly spaced indices into the corpus.
    let queries: Vec<&[f32]> = (0..effective_queries)
        .map(|i| embeddings[i * n / effective_queries].vector.as_slice())
        .collect();

    // ── 2. Brute-force (exact) ground truth ─────────────────────────────────
    if !quiet { println!("Running brute-force (exact) search…"); }
    let brute_idx = SearchIndex::from_dataset(&embeddings);

    let bf_start = Instant::now();
    let exact_results: Vec<Vec<String>> = queries
        .iter()
        .map(|q| brute_idx.top_k(q, effective_k).into_iter().map(|(id, _)| id.to_string()).collect())
        .collect();
    let bf_elapsed = bf_start.elapsed().as_secs_f64();
    let bf_qps = effective_queries as f64 / bf_elapsed;
    let bf_lat_ms = bf_elapsed / effective_queries as f64 * 1000.0;

    // ── 3. HNSW approximate search ───────────────────────────────────────────
    if !quiet { println!("Building HNSW index (M=16, ef_construction=200, ef_search=50)…"); }
    let hnsw_build_start = Instant::now();
    let hnsw = HnswIndex::build(&embeddings, 16, 200, 50);
    let hnsw_build_ms = hnsw_build_start.elapsed().as_secs_f64() * 1000.0;

    let hnsw_start = Instant::now();
    let hnsw_results: Vec<Vec<String>> = queries
        .iter()
        .map(|q| hnsw.search(q, effective_k).into_iter().map(|(id, _)| id).collect())
        .collect();
    let hnsw_elapsed = hnsw_start.elapsed().as_secs_f64();
    let hnsw_qps = effective_queries as f64 / hnsw_elapsed;
    let hnsw_lat_ms = hnsw_elapsed / effective_queries as f64 * 1000.0;
    let hnsw_recall: f64 = exact_results.iter().zip(hnsw_results.iter())
        .map(|(ex, ap)| recall_at_k(ex, ap, effective_k))
        .sum::<f64>() / effective_queries as f64;

    // ── 4. PQ approximate search ─────────────────────────────────────────────
    // m=8 subspaces; fall back to 1 if dimension is not divisible.
    let pq_m = if d % 8 == 0 { 8 } else { 1 };
    let pq_k_centroids = 256usize.min(n);
    if !quiet { println!("Training PQ ({} subspaces, {} centroids per subspace)…", pq_m, pq_k_centroids); }
    let pq_build_start = Instant::now();
    let pq = ProductQuantizer::train(&embeddings, pq_m, pq_k_centroids, 25);
    let pq_build_ms = pq_build_start.elapsed().as_secs_f64() * 1000.0;
    let codes: Vec<(String, Vec<u8>)> = embeddings.iter()
        .map(|e| (e.id.clone(), pq.encode(&e.vector)))
        .collect();

    let pq_start = Instant::now();
    let pq_results: Vec<Vec<String>> = queries.iter().map(|q| {
        pq.search_adc(&codes, q, effective_k)
            .into_iter().map(|(id, _)| id.to_string()).collect()
    }).collect();
    let pq_elapsed = pq_start.elapsed().as_secs_f64();
    let pq_qps = effective_queries as f64 / pq_elapsed;
    let pq_lat_ms = pq_elapsed / effective_queries as f64 * 1000.0;
    let pq_recall: f64 = exact_results.iter().zip(pq_results.iter())
        .map(|(ex, ap)| recall_at_k(ex, ap, effective_k))
        .sum::<f64>() / effective_queries as f64;

    // ── 5. Print summary table ───────────────────────────────────────────────
    let hnsw_gate = hnsw_recall >= 0.90;
    let pq_gate   = pq_recall   >= 0.90;

    println!();
    println!("Ground-Truth Recall & QPS Benchmark");
    println!("  dataset : {} × {}-d | queries: {} | k: {}", n, d, effective_queries, effective_k);
    println!();
    println!("{:<24} {:>10} {:>14} {:>12} {:>12}",
        "algorithm", "recall@k", "QPS", "lat_ms", "build_ms");
    println!("{}", "-".repeat(76));
    println!("{:<24} {:>10} {:>14.0} {:>12.3} {:>12}",
        "brute-force (exact)", "1.0000", bf_qps, bf_lat_ms, "—");
    println!("{:<24} {:>10.4} {:>14.0} {:>12.3} {:>12.0}  {}",
        "HNSW M=16 ef_s=50",
        hnsw_recall, hnsw_qps, hnsw_lat_ms, hnsw_build_ms,
        if hnsw_gate { "✅ ≥0.90" } else { "❌ <0.90" });
    println!("{:<24} {:>10.4} {:>14.0} {:>12.3} {:>12.0}  {}",
        &format!("PQ m={} k={}", pq_m, pq_k_centroids),
        pq_recall, pq_qps, pq_lat_ms, pq_build_ms,
        if pq_gate { "✅ ≥0.90" } else { "❌ <0.90" });
    println!();

    if !hnsw_gate {
        eprintln!("WARNING: HNSW recall@{} = {:.4} — below 0.90 gate.", effective_k, hnsw_recall);
    }
    if !pq_gate {
        eprintln!("WARNING: PQ recall@{} = {:.4} — below 0.90 gate.", effective_k, pq_recall);
    }

    // ── 6. Save JSON report ──────────────────────────────────────────────────
    if save_report {
        use std::fs;
        let dir = std::path::PathBuf::from("benchmarks/results");
        fs::create_dir_all(&dir)?;
        let ts = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
        let filename = dir.join(format!("{}-bench-gt.json", ts));
        let report = serde_json::json!({
            "timestamp": chrono::Local::now().to_rfc3339(),
            "version": env!("CARGO_PKG_VERSION"),
            "dataset": { "source": dataset_path.unwrap_or("synthetic"), "vectors": n, "dimensions": d },
            "queries": effective_queries,
            "k": effective_k,
            "brute_force": { "qps": bf_qps, "latency_ms": bf_lat_ms },
            "hnsw": {
                "m": 16, "ef_construction": 200, "ef_search": 50,
                "build_ms": hnsw_build_ms,
                "qps": hnsw_qps, "latency_ms": hnsw_lat_ms,
                "recall_at_k": hnsw_recall, "gate_pass": hnsw_gate
            },
            "pq": {
                "m_subspaces": pq_m, "k_centroids": pq_k_centroids, "iterations": 25,
                "build_ms": pq_build_ms,
                "qps": pq_qps, "latency_ms": pq_lat_ms,
                "recall_at_k": pq_recall, "gate_pass": pq_gate
            }
        });
        fs::write(&filename, serde_json::to_string_pretty(&report)?)?;
        println!("📄 Report saved → {}", filename.display());
    }

    Ok(())
}

/// Build cargo bench command with optional extra args
fn build_bench_command(bench_args: Option<&str>) -> std::process::Command {
    use std::process::{Command, Stdio};
    let mut cmd = Command::new("cargo");
    cmd.arg("bench").arg("-p").arg("vectro_lib");
    
    if let Some(extra) = bench_args {
        for arg in extra.split_whitespace() {
            cmd.arg(arg);
        }
    }
    
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
}

/// Load benchmark history from file
fn load_bench_history(history_path: &std::path::Path) -> std::collections::HashMap<String, f64> {
    use std::fs;
    let mut history: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    if let Ok(txt) = fs::read_to_string(history_path) {
        if let Ok(hm) = serde_json::from_str::<std::collections::HashMap<String, f64>>(&txt) {
            history = hm;
        }
    }
    history
}

/// Save benchmark history to file
fn save_bench_history(history_path: &std::path::Path, history: &std::collections::HashMap<String, f64>) -> std::io::Result<()> {
    use std::fs;
    let out = serde_json::to_string_pretty(history)?;
    fs::write(history_path, out)
}

/// Calculate delta percentage between current and previous values
///
/// # Examples
///
/// ```
/// # use vectro_cli::*;
/// let delta = vectro_cli::calculate_delta_pub(110.0, 100.0);
/// assert_eq!(delta, Some(10.0));
///
/// let delta2 = vectro_cli::calculate_delta_pub(90.0, 100.0);
/// assert_eq!(delta2, Some(-10.0));
/// ```
fn calculate_delta(current: f64, previous: f64) -> Option<f64> {
    if previous != 0.0 {
        Some((current - previous) / previous * 100.0)
    } else {
        None
    }
}

/// Public wrapper for calculate_delta to enable doc tests
pub fn calculate_delta_pub(current: f64, previous: f64) -> Option<f64> {
    calculate_delta(current, previous)
}

/// Format delta for display
fn format_delta(med: Option<f64>, history: &std::collections::HashMap<String, f64>, name: &str) -> String {
    if let Some(prev) = history.get(name) {
        if let Some(curr) = med {
            if let Some(pct) = calculate_delta(curr, *prev) {
                format!("{:+.2}%", pct)
            } else {
                "n/a".to_string()
            }
        } else {
            "-".to_string()
        }
    } else {
        "-".to_string()
    }
}

/// Calculate delta class for HTML styling
fn get_delta_class(pct: f64) -> &'static str {
    if pct > 0.5 {
        "delta-positive"
    } else if pct < -0.5 {
        "delta-negative"
    } else {
        "delta-neutral"
    }
}

/// Format delta with class for HTML
fn format_delta_html(med: Option<f64>, history: &std::collections::HashMap<String, f64>, name: &str) -> (String, &'static str) {
    if let Some(prev) = history.get(name) {
        if let Some(curr) = med {
            if *prev != 0.0 {
                let pct = (curr - *prev) / *prev * 100.0;
                let class = get_delta_class(pct);
                (format!("{:+.2}%", pct), class)
            } else {
                ("n/a".to_string(), "delta-neutral")
            }
        } else {
            ("-".to_string(), "delta-neutral")
        }
    } else {
        ("-".to_string(), "delta-neutral")
    }
}

/// Recursively search a serde_json::Value for the first numeric value keyed by `key` and return it as f64.
fn find_number_in_json(v: &Value, key: &str) -> Option<f64> {
    match v {
        Value::Object(map) => {
            if let Some(val) = map.get(key) {
                if let Some(n) = val.as_f64() { return Some(n); }
            }
            for (_k, vv) in map.iter() {
                if let Some(n) = find_number_in_json(vv, key) { return Some(n); }
            }
            None
        }
        Value::Array(arr) => {
            for item in arr { if let Some(n) = find_number_in_json(item, key) { return Some(n); } }
            None
        }
        _ => None,
    }
}

/// Recursively find a string field in JSON by key
fn find_string_in_json(v: &Value, key: &str) -> Option<String> {
    match v {
        Value::Object(map) => {
            if let Some(val) = map.get(key) {
                if let Some(s) = val.as_str() { return Some(s.to_string()); }
            }
            for (_k, vv) in map.iter() {
                if let Some(s) = find_string_in_json(vv, key) { return Some(s); }
            }
            None
        }
        Value::Array(arr) => {
            for item in arr { if let Some(s) = find_string_in_json(item, key) { return Some(s); } }
            None
        }
        _ => None,
    }
}

/// Attempt to find an estimate value which may be nested in several known fields
fn get_estimate(v: &Value, key: &str) -> Option<f64> {
    // common shapes: { "estimates": { "median": {"point_estimate": 0.1 } } } or direct
    if let Some(direct) = find_number_in_json(v, key) { return Some(direct); }
    // try path: estimates -> key -> point_estimate
    if let Value::Object(map) = v {
        if let Some(Value::Object(est_map)) = map.get("estimates") {
            if let Some(Value::Object(kmap)) = est_map.get(key) {
                if let Some(pe) = kmap.get("point_estimate") {
                    return pe.as_f64();
                }
            }
        }
    }
    None
}

/// Extract a short benchmark name from Criterion JSON (tries "group_id", "function_id", or fallback)
fn get_bench_name(v: &Value) -> Option<String> {
    // Try common Criterion fields
    if let Some(name) = find_string_in_json(v, "group_id") {
        return Some(name);
    }
    if let Some(name) = find_string_in_json(v, "function_id") {
        return Some(name);
    }
    if let Some(name) = find_string_in_json(v, "title") {
        return Some(name);
    }
    None
}

/// Generate a compact HTML summary from benchmark results
fn generate_html_summary(rows: &[BenchRow], history: &std::collections::HashMap<String, f64>) -> String {
    let mut html = String::from(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vectro+ Benchmark Summary</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
               padding: 2rem; max-width: 1200px; margin: 0 auto; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #4a90e2; padding-bottom: 0.5rem; }
        .timestamp { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
        table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #4a90e2; color: white; font-weight: 600; }
        tr:hover { background: #f9f9f9; }
        .number { text-align: right; font-family: 'Monaco', 'Courier New', monospace; }
        .delta-positive { color: #d32f2f; }
        .delta-negative { color: #388e3c; }
        .delta-neutral { color: #666; }
        .footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #666; font-size: 0.85rem; }
        .link { color: #4a90e2; text-decoration: none; }
        .link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🚀 Vectro+ Benchmark Results</h1>
    <div class="timestamp">Generated: "#);
    
    html.push_str(&format!("{}</div>\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    html.push_str("    <table>\n        <thead>\n            <tr>\n");
    html.push_str("                <th>Benchmark</th><th class=\"number\">Median</th><th class=\"number\">Mean</th><th>Unit</th><th class=\"number\">Δ vs Previous</th>\n");
    html.push_str("            </tr>\n        </thead>\n        <tbody>\n");
    
    for (name, med, mean, unit) in rows {
        let med_str = med.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let mean_str = mean.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "-".to_string());
        let unit_str = unit.clone().unwrap_or_else(|| "".to_string());
        
        let (delta_str, delta_class) = format_delta_html(*med, history, name);
        
        html.push_str(&format!("            <tr>\n                <td>{}</td><td class=\"number\">{}</td><td class=\"number\">{}</td><td>{}</td><td class=\"number {}\">  {}</td>\n            </tr>\n",
            name, med_str, mean_str, unit_str, delta_class, delta_str));
    }
    
    html.push_str(r#"        </tbody>
    </table>
    <div class="footer">
        Generated by <a href="https://github.com/yourorg/vectro-plus" class="link">Vectro+</a> — 
        <a href="./report/index.html" class="link">View Full Criterion Report</a>
    </div>
</body>
</html>"#);
    
    html
}
// Simple recursive directory copy used to copy Criterion reports
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_find_number_in_json_simple() {
        let v = json!({"median": 0.123, "mean": 0.2});
        assert_eq!(find_number_in_json(&v, "median"), Some(0.123));
        assert_eq!(find_number_in_json(&v, "mean"), Some(0.2));
    }

    #[test]
    fn test_find_number_in_json_nested() {
        let v = json!({"estimates": {"median": {"point_estimate": 0.5}, "mean": {"point_estimate": 0.6}}});
        assert_eq!(get_estimate(&v, "median"), Some(0.5));
        assert_eq!(get_estimate(&v, "mean"), Some(0.6));
    }

    #[test]
    fn test_find_string_in_json() {
        let v = json!({"unit": "ns", "nested": {"unit": "ms"}});
        assert_eq!(find_string_in_json(&v, "unit"), Some("ns".to_string()));
        let v2 = json!({"outer": {"inner": {"unit": "us"}}});
        assert_eq!(find_string_in_json(&v2, "unit"), Some("us".to_string()));
    }

    #[test]
    fn test_get_bench_name() {
        let v1 = json!({"group_id": "search/cosine", "function_id": "top_k"});
        assert_eq!(get_bench_name(&v1), Some("search/cosine".to_string()));
        
        let v2 = json!({"function_id": "quantize_dataset", "title": "Quantization Bench"});
        assert_eq!(get_bench_name(&v2), Some("quantize_dataset".to_string()));
        
        let v3 = json!({"title": "Simple Bench"});
        assert_eq!(get_bench_name(&v3), Some("Simple Bench".to_string()));
    }

    #[test]
    fn test_bench_summary_parsing() {
        use std::fs;
        use tempfile::TempDir;

        // Create fake Criterion-like JSON output
        let tmp = TempDir::new().unwrap();
        let crit_dir = tmp.path().join("criterion");
        let bench_dir = crit_dir.join("cosine_search").join("new");
        fs::create_dir_all(&bench_dir).unwrap();

        let fake_json = json!({
            "group_id": "cosine_search",
            "function_id": "top_k_100",
            "estimates": {
                "median": {"point_estimate": 123.456},
                "mean": {"point_estimate": 125.789}
            },
            "unit": "ns"
        });

        fs::write(bench_dir.join("estimates.json"), serde_json::to_string_pretty(&fake_json).unwrap()).unwrap();

        // Parse the fake structure
        let mut found = false;
        if let Ok(entries) = fs::read_dir(&crit_dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let new_dir = p.join("new");
                    if new_dir.exists() {
                        if let Ok(it) = fs::read_dir(&new_dir) {
                            for j in it.flatten() {
                                let jp = j.path();
                                if jp.extension().map(|s| s == "json").unwrap_or(false) {
                                    if let Ok(txt) = fs::read_to_string(&jp) {
                                        if let Ok(json) = serde_json::from_str::<Value>(&txt) {
                                            let med = get_estimate(&json, "median");
                                            let mean = get_estimate(&json, "mean");
                                            let unit = find_string_in_json(&json, "unit");
                                            let name = get_bench_name(&json);

                                            assert_eq!(med, Some(123.456));
                                            assert_eq!(mean, Some(125.789));
                                            assert_eq!(unit, Some("ns".to_string()));
                                            assert_eq!(name, Some("cosine_search".to_string()));
                                            found = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found, "Should have parsed the fake Criterion JSON");
    }

    #[test]
    fn test_execute_compress_command() {
        use tempfile::NamedTempFile;
        
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap();
        std::fs::write(in_path, r#"{"id":"test","vector":[1.0,2.0,3.0]}"#).unwrap();
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command(in_path, out_path, "stream", false, 8, 256);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_execute_compress_command_quantized() {
        use tempfile::NamedTempFile;
        
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap();
        std::fs::write(in_path, r#"{"id":"a","vector":[1.0,0.0]}
{"id":"b","vector":[0.0,1.0]}"#).unwrap();
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command(in_path, out_path, "stream", true, 8, 256);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_execute_compress_command_invalid_input() {
        use tempfile::NamedTempFile;
        
        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap();
        
        let result = execute_compress_command("/nonexistent/file.jsonl", out_path, "stream", false, 8, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_query_string() {
        let result = parse_query_string("1.0,2.0,3.0");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        
        let result = parse_query_string("1.0, 2.0, 3.0");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        
        let result = parse_query_string("1.0,invalid,3.0");
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn test_load_dataset_or_default() {
        // Test with None - should return default
        let result = load_dataset_or_default(None);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, "one");
        
        // Test with invalid path - should return default
        let result = load_dataset_or_default(Some("/nonexistent/path.bin"));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_load_dataset_or_default_valid_file() {
        use tempfile::NamedTempFile;
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("test1", vec![1.0, 0.0]));
        ds.add(vectro_lib::Embedding::new("test2", vec![0.0, 1.0]));
        ds.save(path).unwrap();
        
        let result = load_dataset_or_default(Some(path));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "test1");
    }

    #[test]
    fn test_execute_search_command() {
        let results = execute_search_command("1.0,0.0", 2, None);
        assert!(results.len() <= 2);
        assert!(!results.is_empty());
        
        // First result should be "one" with highest similarity
        assert_eq!(results[0].0, "one");
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn test_execute_search_command_with_dataset() {
        use tempfile::NamedTempFile;
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("apple", vec![1.0, 0.0, 0.0]));
        ds.add(vectro_lib::Embedding::new("banana", vec![0.0, 1.0, 0.0]));
        ds.save(path).unwrap();
        
        let results = execute_search_command("1.0,0.0,0.0", 1, Some(path));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");
    }

    #[test]
    fn test_cli_parsing_compress() {
        // Test that CLI can parse compress command
        use clap::Parser;
        
        let args = vec!["vectro", "compress", "input.jsonl", "output.bin"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Compress { input, output, quantize, .. } => {
                    assert_eq!(input, "input.jsonl");
                    assert_eq!(output, "output.bin");
                    assert!(!quantize);
                }
                _ => panic!("Expected Compress command"),
            }
        }
    }

    #[test]
    fn test_cli_parsing_compress_quantized() {
        use clap::Parser;
        
        let args = vec!["vectro", "compress", "in.jsonl", "out.bin", "--quantize"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Compress { quantize, .. } => {
                assert!(quantize);
            }
            _ => panic!("Expected Compress command"),
        }
    }

    #[test]
    fn test_cli_parsing_search() {
        use clap::Parser;
        
        let args = vec!["vectro", "search", "1.0,0.0,0.0"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Search { query, top_k, dataset } => {
                assert_eq!(query, "1.0,0.0,0.0");
                assert_eq!(top_k, 10); // default
                assert!(dataset.is_none());
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_parsing_search_with_options() {
        use clap::Parser;
        
        let args = vec!["vectro", "search", "1.0,0.0", "--top-k", "5", "--dataset", "data.bin"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Search { query, top_k, dataset } => {
                assert_eq!(query, "1.0,0.0");
                assert_eq!(top_k, 5);
                assert_eq!(dataset.as_deref(), Some("data.bin"));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve() {
        use clap::Parser;
        
        let args = vec!["vectro", "serve"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Serve { port } => {
                assert_eq!(port, 8080); // default
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve_custom_port() {
        use clap::Parser;
        
        let args = vec!["vectro", "serve", "--port", "3000"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Serve { port } => {
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_bench() {
        use clap::Parser;
        
        let args = vec!["vectro", "bench"];
        let cli = Cli::try_parse_from(args).unwrap();
        
        match cli.command {
            Commands::Bench { save_report, open_report, summary, .. } => {
                assert!(save_report.is_none());
                assert!(!open_report);
                assert!(summary); // default true
            }
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_build_bench_command() {
        let cmd = build_bench_command(None);
        let program = cmd.get_program();
        assert_eq!(program, "cargo");
    }

    #[test]
    fn test_build_bench_command_with_args() {
        let cmd = build_bench_command(Some("--verbose"));
        let program = cmd.get_program();
        assert_eq!(program, "cargo");
    }

    #[test]
    fn test_load_bench_history_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let history_path = temp_dir.path().join("nonexistent.json");
        let history = load_bench_history(&history_path);
        assert!(history.is_empty());
    }

    #[test]
    fn test_load_save_bench_history() {
        use std::collections::HashMap;
        let temp_dir = tempfile::tempdir().unwrap();
        let history_path = temp_dir.path().join("history.json");
        
        let mut history = HashMap::new();
        history.insert("test_bench".to_string(), 123.456);
        
        save_bench_history(&history_path, &history).unwrap();
        let loaded = load_bench_history(&history_path);
        
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("test_bench"), Some(&123.456));
    }

    #[test]
    fn test_calculate_delta() {
        // Normal case
        let delta = calculate_delta(110.0, 100.0);
        assert_eq!(delta, Some(10.0));
        
        // Decrease
        let delta = calculate_delta(90.0, 100.0);
        assert_eq!(delta, Some(-10.0));
        
        // Zero previous
        let delta = calculate_delta(100.0, 0.0);
        assert_eq!(delta, None);
    }

    #[test]
    fn test_format_delta() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 100.0);
        
        // With history and current value
        let delta_str = format_delta(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "+10.00%");
        
        // No history
        let delta_str = format_delta(Some(110.0), &history, "bench2");
        assert_eq!(delta_str, "-");
        
        // No current value
        let delta_str = format_delta(None, &history, "bench1");
        assert_eq!(delta_str, "-");
    }

    #[test]
    fn test_format_delta_zero_previous() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 0.0);
        
        let delta_str = format_delta(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "n/a");
    }

    #[test]
    fn test_get_delta_class() {
        assert_eq!(get_delta_class(1.0), "delta-positive");
        assert_eq!(get_delta_class(-1.0), "delta-negative");
        assert_eq!(get_delta_class(0.3), "delta-neutral");
        assert_eq!(get_delta_class(-0.3), "delta-neutral");
    }

    #[test]
    fn test_format_delta_html() {
        use std::collections::HashMap;
        
        let mut history = HashMap::new();
        history.insert("bench1".to_string(), 100.0);
        
        // Positive delta
        let (delta_str, class) = format_delta_html(Some(110.0), &history, "bench1");
        assert_eq!(delta_str, "+10.00%");
        assert_eq!(class, "delta-positive");
        
        // Negative delta
        let (delta_str, class) = format_delta_html(Some(90.0), &history, "bench1");
        assert_eq!(delta_str, "-10.00%");
        assert_eq!(class, "delta-negative");
        
        // Neutral delta
        let (delta_str, class) = format_delta_html(Some(100.3), &history, "bench1");
        assert_eq!(delta_str, "+0.30%");
        assert_eq!(class, "delta-neutral");
        
        // No history
        let (delta_str, class) = format_delta_html(Some(110.0), &history, "bench2");
        assert_eq!(delta_str, "-");
        assert_eq!(class, "delta-neutral");
    }

    #[test]
    fn test_calculate_delta_comprehensive() {
        // Test improvement (faster is negative delta)
        let delta = calculate_delta(90.0, 100.0);
        assert_eq!(delta, Some(-10.0));
        
        // Test regression (slower is positive delta)
        let delta = calculate_delta(110.0, 100.0);
        assert_eq!(delta, Some(10.0));
        
        // Test zero previous value
        let delta = calculate_delta(50.0, 0.0);
        assert_eq!(delta, None);
        
        // Test equal values
        let delta = calculate_delta(100.0, 100.0);
        assert_eq!(delta, Some(0.0));
        
        // Test small change
        let delta = calculate_delta(100.5, 100.0);
        assert_eq!(delta, Some(0.5));
    }

    #[test]
    fn test_find_number_in_json_deeply_nested() {
        let json = json!({
            "level1": {
                "level2": {
                    "level3": {
                        "target": 42.5
                    }
                }
            }
        });
        assert_eq!(find_number_in_json(&json, "target"), Some(42.5));
    }

    #[test]
    fn test_find_string_in_json_array() {
        let json = json!({
            "items": [
                {"name": "first"},
                {"name": "second"},
                {"unit": "milliseconds"}
            ]
        });
        assert_eq!(find_string_in_json(&json, "unit"), Some("milliseconds".to_string()));
    }

    #[test]
    fn test_get_estimate_criterion_format() {
        // Test realistic Criterion JSON structure
        let criterion_json = json!({
            "estimates": {
                "median": {
                    "point_estimate": 123.456,
                    "confidence_interval": {
                        "lower_bound": 120.0,
                        "upper_bound": 130.0
                    }
                },
                "mean": {
                    "point_estimate": 125.789
                }
            }
        });
        
        assert_eq!(get_estimate(&criterion_json, "median"), Some(123.456));
        assert_eq!(get_estimate(&criterion_json, "mean"), Some(125.789));
    }

    #[test]
    fn test_parse_query_string_edge_cases() {
        // Empty string
        assert_eq!(parse_query_string(""), Vec::<f32>::new());
        
        // Single value
        assert_eq!(parse_query_string("1.0"), vec![1.0]);
        
        // Multiple values with spaces
        assert_eq!(parse_query_string("1.0, 2.0 , 3.0"), vec![1.0, 2.0, 3.0]);
        
        // Invalid values (should be filtered out)
        assert_eq!(parse_query_string("1.0,invalid,3.0"), vec![1.0, 3.0]);
        
        // Negative and decimal values
        assert_eq!(parse_query_string("-1.5,0.0,2.5"), vec![-1.5, 0.0, 2.5]);
    }

    #[test]
    fn test_load_dataset_or_default_with_valid_file() {
        use tempfile::NamedTempFile;
        
        // Create a temporary dataset file
        let mut ds = vectro_lib::EmbeddingDataset::new();
        ds.add(vectro_lib::Embedding::new("test1", vec![1.0, 2.0, 3.0]));
        ds.add(vectro_lib::Embedding::new("test2", vec![4.0, 5.0, 6.0]));
        
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        ds.save(path).unwrap();
        
        // Test loading the dataset
        let loaded = load_dataset_or_default(Some(path));
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "test1");
        assert_eq!(loaded[1].id, "test2");
    }

    #[test]
    fn test_load_dataset_or_default_with_invalid_path() {
        // Test with invalid path - should return default dataset
        let embeddings = load_dataset_or_default(Some("/nonexistent/path.bin"));
        assert!(!embeddings.is_empty()); // Should return default toy dataset
    }

    #[test]
    fn test_load_dataset_or_default_none() {
        // Test with None - should return default dataset
        let embeddings = load_dataset_or_default(None);
        assert!(!embeddings.is_empty()); // Should return default toy dataset
    }

    #[test]
    fn test_get_bench_name_priority() {
        // Test that group_id takes priority
        let json = json!({
            "group_id": "group_name",
            "function_id": "function_name",
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json), Some("group_name".to_string()));
        
        // Test fallback to function_id when group_id is missing
        let json2 = json!({
            "function_id": "function_name",
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json2), Some("function_name".to_string()));
        
        // Test fallback to title when both are missing
        let json3 = json!({
            "title": "Some Title"
        });
        assert_eq!(get_bench_name(&json3), Some("Some Title".to_string()));
        
        // Test None when all are missing
        let json4 = json!({
            "other_field": "value"
        });
        assert_eq!(get_bench_name(&json4), None);
    }
}
