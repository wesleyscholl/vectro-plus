//! End-to-end pipeline: compress → build HNSW index → optional batch search.
//!
//! Invoked via `vectro pipeline --input data.jsonl --out-dir ./pipeline_out`.
//! Steps:
//!   1. Compress the input file to the chosen format (default: stream1 passthrough).
//!   2. Load the compressed dataset.
//!   3. Build an HNSW index over it.
//!   4. Persist the index to <out_dir>/index.bin.
//!   5. If --query-file is provided, run brute-force search for every query vector
//!      and emit JSON lines to stdout.
use anyhow::{Context, Result};
use std::fs;

/// Run the full pipeline.
///
/// # Arguments
/// * `input`          – Path to the source JSONL or STREAM1 file.
/// * `out_dir`        – Directory that receives compressed.stream1 + index.bin.
/// * `format`         – One of "stream1", "pq", "nf4", "rq", "auto".
/// * `m`              – HNSW M parameter (number of bi-directional links per element).
/// * `ef_construction`– HNSW ef during index construction.
/// * `ef_search`      – HNSW ef at query time.
/// * `query_file`     – Optional JSONL file with `{"id":"…","vector":[…]}` objects.
/// * `top_k`          – Number of results per query.
/// * `quiet`          – Suppress progress output.
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    input: &str,
    out_dir: &str,
    format: &str,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    query_file: Option<&str>,
    top_k: usize,
    quiet: bool,
) -> Result<()> {
    // ── Step 1: create output directory ────────────────────────────────────
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {out_dir}"))?;

    // ── Step 2: compress ───────────────────────────────────────────────────
    let compressed_path = format!("{out_dir}/compressed.stream1");

    if !quiet {
        eprintln!("→ compressing [{format}] {input} → {compressed_path}");
    }

    let n = match format {
        "pq" => crate::compress_pq(input, &compressed_path, 8, 256)
            .with_context(|| format!("compress_pq failed for {input}"))?,
        "nf4" => crate::compress_nf4(input, &compressed_path)
            .with_context(|| format!("compress_nf4 failed for {input}"))?,
        "rq" => crate::compress_rq(input, &compressed_path, 2, 8, 256)
            .with_context(|| format!("compress_rq failed for {input}"))?,
        "auto" => crate::compress_auto(input, &compressed_path, 0.97, 8.0)
            .with_context(|| format!("compress_auto failed for {input}"))?,
        _ => {
            // "stream1" or any unknown format falls through to lossless passthrough
            crate::compress_stream(input, &compressed_path, false)
                .with_context(|| format!("compress_stream failed for {input}"))?
        }
    };

    if !quiet {
        eprintln!("  ✓ compressed {n} vectors");
    }

    // ── Step 3: load dataset ───────────────────────────────────────────────
    if !quiet {
        eprintln!("→ loading dataset from {compressed_path}");
    }

    let dataset = vectro_lib::EmbeddingDataset::load(&compressed_path)
        .with_context(|| format!("failed to load dataset from {compressed_path}"))?;

    if !quiet {
        eprintln!("  ✓ loaded {} embeddings", dataset.embeddings.len());
    }

    // ── Step 4: build HNSW index ───────────────────────────────────────────
    let index_path = format!("{out_dir}/index.bin");

    if !quiet {
        eprintln!("→ building HNSW index (M={m}, ef_construction={ef_construction}, ef_search={ef_search})");
    }

    let index = vectro_lib::HnswIndex::build(&dataset.embeddings, m, ef_construction, ef_search);

    if !quiet {
        eprintln!("  ✓ built index with {} nodes", index.len());
    }

    index
        .save(&index_path)
        .with_context(|| format!("failed to save index to {index_path}"))?;

    if !quiet {
        eprintln!("  ✓ saved index → {index_path}");
    }

    // ── Step 5: optional batch search ─────────────────────────────────────
    if let Some(qf) = query_file {
        run_queries(&index, qf, top_k, quiet)?;
    }

    Ok(())
}

/// Load JSONL query vectors from `query_file`, search against `index`, and write
/// JSON results to stdout (one line per query).
fn run_queries(
    index: &vectro_lib::HnswIndex,
    query_file: &str,
    top_k: usize,
    quiet: bool,
) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};

    if !quiet {
        eprintln!("→ running queries from {query_file}");
    }

    let f = fs::File::open(query_file)
        .with_context(|| format!("failed to open query file: {query_file}"))?;
    let reader = BufReader::new(f);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut query_count = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("I/O error reading {query_file} at line {line_no}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Expect: {"id":"<name>","vector":[f32,...]}
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("JSON parse error at line {line_no}: {line}"))?;

        let query_id = v["id"].as_str().unwrap_or("?").to_owned();
        let vector: Vec<f32> = v["vector"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'vector' array at line {line_no}"))?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();

        let results = index.search(&vector, top_k);

        // Emit one JSON line per query
        let result_json: Vec<serde_json::Value> = results
            .iter()
            .map(|(id, score)| serde_json::json!({"id": id, "score": score}))
            .collect();

        let row = serde_json::json!({
            "query_id": query_id,
            "results": result_json,
        });

        writeln!(out, "{}", row)?;
        query_count += 1;
    }

    if !quiet {
        eprintln!("  ✓ processed {query_count} queries");
    }

    Ok(())
}
