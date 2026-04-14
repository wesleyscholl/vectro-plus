use std::io::{BufRead, BufReader, Write};
use indicatif::{ProgressBar, ProgressStyle};

pub fn compress_stream(input: &str, output: &str, quantize: bool) -> anyhow::Result<usize> {
    use crossbeam_channel::{bounded, Sender, Receiver};
    use std::thread;

    let header = b"VECTRO+STREAM1\n";
    let infile = std::fs::File::open(input)?;
    let reader = BufReader::new(infile);

    let outfile = std::fs::File::create(output)?;
    let writer_buf = std::io::BufWriter::new(outfile);

    // channels
    let (item_tx, item_rx): (Sender<vectro_lib::Embedding>, Receiver<vectro_lib::Embedding>) = bounded(1024);
    let (bytes_tx, bytes_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = bounded(1024);

    // writer thread (non-quantized path will spawn writer now; quantized path spawns writer after tables computed)
    let out_clone = output.to_string();
    let qheader = b"VECTRO+QSTREAM1\n";
    let mut writer_handle_opt = None;
    // prepare worker handles container
    let mut worker_handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
    if !quantize {
        let mut w = writer_buf;
        let rx_for_writer = bytes_rx.clone();
        let out_for_writer = out_clone.clone();
        let header_local = *header;
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            w.write_all(&header_local)?;
            let mut written = 0usize;
            while let Ok(bytes) = rx_for_writer.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_for_writer);
            Ok(())
        });
        writer_handle_opt = Some(handle);
        // spawn workers for non-quantized path
        let workers = num_cpus::get().max(1);
        for _ in 0..workers {
            let r = item_rx.clone();
            let tx = bytes_tx.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    if let Ok(bytes) = bincode::serialize(&e) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }
    }

    // don't spawn workers yet; will spawn depending on quantize mode

    // progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    if quantize {
        pb.set_message("parsing and computing quant tables...");
    } else {
        pb.set_message("compressing (streaming bincode)...");
    }

    // reader: parse lines and collect embeddings
    let mut parsed = 0usize;
    // collect embeddings when quantizing
    let mut collected_embeddings: Vec<vectro_lib::Embedding> = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() { continue; }

        // try JSON
        let mut pushed = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(vec)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec.as_array()) {
                    let mut v = Vec::with_capacity(arr.len());
                    for x in arr { if let Some(flt) = x.as_f64() { v.push(flt as f32); } }
                    let emb = vectro_lib::Embedding::new(id_str, v.clone());
                    if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                    parsed += 1;
                    pushed = true;
                }
            }
        }
        if !pushed {
            // CSV
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let id = parts[0].to_string();
                let mut v = Vec::new();
                for p in &parts[1..] { if let Ok(f) = p.trim().parse::<f32>() { v.push(f); } }
                let emb = vectro_lib::Embedding::new(id, v.clone());
                if quantize { collected_embeddings.push(emb.clone()); } else { let _ = item_tx.send(emb); }
                parsed += 1;
            }
        }

        if parsed % 100 == 0 { pb.set_message(format!("parsed {} entries", parsed)); }
    }

    if quantize {
        // compute tables using vectro_lib::search::quant::quantize_dataset
        let vectors: Vec<Vec<f32>> = collected_embeddings.iter().map(|e| e.vector.clone()).collect();
        let (tables, _qvecs) = vectro_lib::search::quant::quantize_dataset(&vectors);
        // serialize tables to bincode
        let tables_blob = bincode::serialize(&tables)?;

        // write header + tables to file, then spawn writer thread to append entries
        {
            // overwrite/create file and write header+tables
            let mut f = std::fs::File::create(output)?;
            let mut w = std::io::BufWriter::new(&mut f);
            w.write_all(qheader)?;
            let table_count = (tables.len() as u32).to_le_bytes();
            let dim = (if !tables.is_empty() { tables.len() as u32 } else { 0u32 }).to_le_bytes();
            let tables_len = (tables_blob.len() as u32).to_le_bytes();
            w.write_all(&table_count)?;
            w.write_all(&dim)?;
            w.write_all(&tables_len)?;
            w.write_all(&tables_blob)?;
            w.flush()?;
        }

        // spawn writer that appends entries
        let outfile = std::fs::OpenOptions::new().append(true).open(output)?;
        let writer_buf = std::io::BufWriter::new(outfile);
        let out_clone2 = out_clone.clone();
        let handle = thread::spawn(move || -> anyhow::Result<()> {
            let mut w = writer_buf;
            let mut written = 0usize;
            while let Ok(bytes) = bytes_rx.recv() {
                let len = (bytes.len() as u32).to_le_bytes();
                w.write_all(&len)?;
                w.write_all(&bytes)?;
                written += 1;
            }
            w.flush()?;
            eprintln!("wrote {} entries to {}", written, out_clone2);
            Ok(())
        });
        writer_handle_opt = Some(handle);

        // spawn workers to quantize embeddings
        let workers = num_cpus::get().max(1);
        use crossbeam_channel::bounded;
        let (item_tx2, item_rx2) = bounded::<vectro_lib::Embedding>(1024);
        // worker threads
        for _ in 0..workers {
            let r = item_rx2.clone();
            let tx = bytes_tx.clone();
            let tables = tables.clone();
            worker_handles.push(thread::spawn(move || {
                while let Ok(e) = r.recv() {
                    // quantize vector
                    let qv: Vec<u8> = e.vector.iter().enumerate().map(|(i, &x)| tables[i].quantize(x)).collect();
                    let rec = (e.id.clone(), qv);
                    if let Ok(bytes) = bincode::serialize(&rec) {
                        let _ = tx.send(bytes);
                    }
                }
            }));
        }

        // feed collected embeddings into item_tx2
        for emb in collected_embeddings {
            let _ = item_tx2.send(emb);
        }
        drop(item_tx2);

        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }

    } else {
        // close item_tx to signal workers to finish
        drop(item_tx);
        // wait for workers
        drop(bytes_tx);
        for h in worker_handles { let _ = h.join(); }
        // wait for writer
        if let Some(h) = writer_handle_opt { let _ = h.join(); }
    }
    if quantize {
    // If quantized, show a short summary including table count (attempt to read tables from file)
    // variable intentionally unused; underscore prefix to silence warnings
    let _table_count = 0usize;
        if let Ok(mut f) = std::fs::File::open(output) {
            use std::io::Read;
            let mut hdr = vec![0u8; 16];
            let _ = f.read(&mut hdr);
            // crude: read table_count at offset header.len()
            // header 'VECTRO+QSTREAM1\n' length is 14
            if hdr.len() >= 16 {
                // no-op; we will just display quantized
            }
        }
        pb.finish_with_message(format!("wrote {} quantized entries to {}", parsed, output));
    } else {
        pb.finish_with_message(format!("wrote {} entries to {}", parsed, output));
    }
    Ok(parsed)
}

/// Compress embeddings to the `VECTRO+PQSTREAM1` binary format using Product Quantization.
///
/// # Arguments
/// * `input`  — path to JSONL or CSV file (`{"id":..,"vector":[..]}` per line or `id,f,f,..`).
/// * `output` — path for the `.pqstream1` output file.
/// * `m`      — number of PQ subspaces (encoded bytes per vector); must divide vector dimension.
/// * `k`      — centroids per subspace; must be ≤ 256 (default: 256).
///
/// Returns the number of embeddings written.
pub fn compress_pq(input: &str, output: &str, m: usize, k: usize) -> anyhow::Result<usize> {
    use std::io::BufRead;

    let pqheader = b"VECTRO+PQSTREAM1\n";

    let infile = std::fs::File::open(input)?;
    let reader = std::io::BufReader::new(infile);

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message("reading input for PQ training...");

    // Collect all embeddings (PQ requires a full pass for training before encoding).
    let mut embeddings: Vec<vectro_lib::Embedding> = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Try JSON first; fall back to CSV.
        let mut added = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(vec)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec.as_array()) {
                    let v: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    embeddings.push(vectro_lib::Embedding::new(id_str, v));
                    added = true;
                }
            }
        }
        if !added {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let id = parts[0].to_string();
                let v: Vec<f32> = parts[1..]
                    .iter()
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if !v.is_empty() {
                    embeddings.push(vectro_lib::Embedding::new(id, v));
                }
            }
        }

        if embeddings.len() % 500 == 0 && !embeddings.is_empty() {
            pb.set_message(format!("read {} embeddings...", embeddings.len()));
        }
    }

    let count = embeddings.len();
    if count == 0 {
        pb.finish_with_message("no embeddings found — nothing written");
        return Ok(0);
    }

    pb.set_message(format!(
        "training PQ on {} vectors (m={}, k={})…",
        count, m, k
    ));
    let pq = vectro_lib::ProductQuantizer::train(&embeddings, m, k, 25);

    pb.set_message("encoding and writing…");
    let pq_blob = bincode::serialize(&pq)?;

    let f = std::fs::File::create(output)?;
    let mut w = std::io::BufWriter::new(f);

    // Write header.
    w.write_all(pqheader)?;
    // Write ProductQuantizer blob with a 4-byte LE length prefix.
    let pq_len = (pq_blob.len() as u32).to_le_bytes();
    w.write_all(&pq_len)?;
    w.write_all(&pq_blob)?;

    // Write each record: 4-byte LE length prefix + bincode((id, code)).
    for emb in &embeddings {
        let code = pq.encode(&emb.vector);
        let rec = (emb.id.clone(), code);
        let bytes = bincode::serialize(&rec)?;
        let len = (bytes.len() as u32).to_le_bytes();
        w.write_all(&len)?;
        w.write_all(&bytes)?;
    }
    w.flush()?;

    pb.finish_with_message(format!(
        "wrote {} PQ-encoded entries to {} (m={}, k={})",
        count, output, m, k
    ));
    Ok(count)
}

// ─── NF4 ──────────────────────────────────────────────────────────────────────

/// Compress a JSONL embedding file to `VECTRO+NF4STREAM1` format.
///
/// Each vector is scaled to abs-max=1 then quantized to 4-bit NormalFloat.
/// Storage per vector: `ceil(dim/2)` bytes of packed nibbles + 4 bytes for the f32 scale.
pub fn compress_nf4(input: &str, output: &str) -> anyhow::Result<usize> {
    let nf4header = b"VECTRO+NF4STREAM1\n";
    let embeddings = read_jsonl(input)?;
    let count = embeddings.len();
    if count == 0 {
        return Ok(0);
    }

    let pb = ProgressBar::new(count as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message("encoding NF4…");

    let q = vectro_lib::Nf4Quantizer::new();
    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();
    let (packed_list, scales) = q.encode(&vecs);

    let dim = embeddings[0].vector.len() as u32;
    let f = std::fs::File::create(output)?;
    let mut w = std::io::BufWriter::new(f);

    w.write_all(nf4header)?;
    w.write_all(&dim.to_le_bytes())?;
    w.write_all(&(count as u32).to_le_bytes())?;

    for (i, emb) in embeddings.iter().enumerate() {
        let rec = (emb.id.clone(), packed_list[i].clone(), scales[i]);
        let bytes = bincode::serialize(&rec)?;
        let len = (bytes.len() as u32).to_le_bytes();
        w.write_all(&len)?;
        w.write_all(&bytes)?;
        pb.inc(1);
    }
    w.flush()?;

    pb.finish_with_message(format!(
        "wrote {} NF4-encoded entries to {}",
        count, output
    ));
    Ok(count)
}

// ─── RQ ───────────────────────────────────────────────────────────────────────

/// Compress a JSONL embedding file to `VECTRO+RQSTREAM1` format.
///
/// Trains a `ResidualQuantizer` with `n_passes` and `m` subspaces, then encodes
/// each embedding as a list of code arrays (one per pass, one byte per subspace).
pub fn compress_rq(
    input: &str,
    output: &str,
    n_passes: usize,
    m: usize,
    k: usize,
) -> anyhow::Result<usize> {
    let rqheader = b"VECTRO+RQSTREAM1\n";
    let embeddings = read_jsonl(input)?;
    let count = embeddings.len();
    if count == 0 {
        return Ok(0);
    }

    let pb = ProgressBar::new(count as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(format!("training RQ (passes={}, m={}, k={})…", n_passes, m, k));

    let mut rq = vectro_lib::ResidualQuantizer::new(n_passes, m, k);
    rq.train(&embeddings, 15);

    pb.set_message("encoding RQ…");
    let rq_blob = bincode::serialize(&rq)?;

    let f = std::fs::File::create(output)?;
    let mut w = std::io::BufWriter::new(f);

    w.write_all(rqheader)?;
    let rq_len = (rq_blob.len() as u32).to_le_bytes();
    w.write_all(&rq_len)?;
    w.write_all(&rq_blob)?;

    for emb in &embeddings {
        let codes = rq.encode(&emb.vector);
        let rec = (emb.id.clone(), codes);
        let bytes = bincode::serialize(&rec)?;
        let len = (bytes.len() as u32).to_le_bytes();
        w.write_all(&len)?;
        w.write_all(&bytes)?;
        pb.inc(1);
    }
    w.flush()?;

    pb.finish_with_message(format!("wrote {} RQ-encoded entries to {}", count, output));
    Ok(count)
}

// ─── AUTO-QUANTIZE ────────────────────────────────────────────────────────────

/// Compress a JSONL embedding file using the automatically-selected best format.
///
/// Evaluates NF4, RQ, PQ, and Scalar on a sample of up to 1 000 vectors and
/// picks the first format that meets `target_cosine` (default 0.97) and
/// `target_compression` (default 8×).
pub fn compress_auto(
    input: &str,
    output: &str,
    target_cosine: f32,
    target_compression: f32,
) -> anyhow::Result<usize> {
    let embeddings = read_jsonl(input)?;
    if embeddings.is_empty() {
        return Ok(0);
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message("evaluating formats for auto-select…");

    let result =
        vectro_lib::auto_select_format(&embeddings, target_cosine, target_compression, 1000);

    pb.set_message(format!(
        "selected format={} (cosine={:.4}, compression={:.1}x); compressing…",
        result.format, result.cosine_sim, result.compression_ratio
    ));

    let n = match result.format {
        vectro_lib::QuantFormat::Nf4    => compress_nf4(input, output)?,
        vectro_lib::QuantFormat::Rq     => compress_rq(input, output, 2, 8, 64)?,
        vectro_lib::QuantFormat::Pq     => compress_pq(
            input, output,
            result.pq.as_ref().map(|_| 8).unwrap_or(8),
            256,
        )?,
        vectro_lib::QuantFormat::Scalar | vectro_lib::QuantFormat::Stream => {
            compress_stream(input, output, true)?
        }
    };

    pb.finish_with_message(format!(
        "auto-compress complete: format={fmt} -> {n} records in {output}",
        fmt = result.format
    ));
    Ok(n)
}

// ─── shared JSONL reader ──────────────────────────────────────────────────────

fn read_jsonl(input: &str) -> anyhow::Result<Vec<vectro_lib::Embedding>> {
    use std::io::BufRead;
    let infile = std::fs::File::open(input)?;
    let reader = std::io::BufReader::new(infile);
    let mut embeddings = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let mut added = false;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if let (Some(id), Some(vec)) = (val.get("id"), val.get("vector")) {
                if let (Some(id_str), Some(arr)) = (id.as_str(), vec.as_array()) {
                    let v: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    embeddings.push(vectro_lib::Embedding::new(id_str, v));
                    added = true;
                }
            }
        }
        if !added {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let id = parts[0].to_string();
                let v: Vec<f32> = parts[1..]
                    .iter()
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if !v.is_empty() {
                    embeddings.push(vectro_lib::Embedding::new(id, v));
                }
            }
        }
    }
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn compress_small_file() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,0.0]}
{"id":"two","vector":[0.0,1.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_quantized() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"{"id":"one","vector":[1.0,2.0,3.0]}
{"id":"two","vector":[4.0,5.0,6.0]}
{"id":"three","vector":[7.0,8.0,9.0]}"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, true).expect("compress quantized");
        assert_eq!(n, 3);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 3);
        // Quantized embeddings may not preserve order
        let ids: Vec<&str> = ds.embeddings.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"one"));
        assert!(ids.contains(&"two"));
        assert!(ids.contains(&"three"));
    }

    #[test]
    fn compress_csv_format() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, "id1,1.0,2.0,3.0\nid2,4.0,5.0,6.0\n").unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress csv");
        assert_eq!(n, 2);

        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load");
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn compress_with_empty_lines() {
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        std::fs::write(&in_path, r#"
{"id":"one","vector":[1.0,0.0]}

{"id":"two","vector":[0.0,1.0]}

"#).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_stream(&in_path, &out_path, false).expect("compress");
        assert_eq!(n, 2);
    }

    #[test]
    fn compress_pq_roundtrip() {
        // Verify compress_pq writes a valid PQSTREAM1 file that EmbeddingDataset::load can read.
        let tmp_in = NamedTempFile::new().unwrap();
        let in_path = tmp_in.path().to_str().unwrap().to_string();
        // 32-dim vectors, m=4 divides 32 cleanly.
        let mut lines = String::new();
        for i in 0..100 {
            let v: Vec<String> = (0..32)
                .map(|d| format!("{:.3}", ((i * 17 + d * 11) % 97) as f32 / 97.0))
                .collect();
            lines.push_str(&format!("{{\"id\":\"id_{}\",\"vector\":[{}]}}\n", i, v.join(",")));
        }
        std::fs::write(&in_path, &lines).unwrap();

        let tmp_out = NamedTempFile::new().unwrap();
        let out_path = tmp_out.path().to_str().unwrap().to_string();

        let n = compress_pq(&in_path, &out_path, 4, 16).expect("compress_pq");
        assert_eq!(n, 100, "should write 100 entries");

        // Round-trip: load reconstructs approximate f32 vectors.
        let ds = vectro_lib::EmbeddingDataset::load(&out_path).expect("load PQSTREAM1");
        assert_eq!(ds.len(), 100);
        // Check IDs are preserved.
        let ids: std::collections::HashSet<String> =
            ds.embeddings.iter().map(|e| e.id.clone()).collect();
        assert!(ids.contains("id_0"));
        assert!(ids.contains("id_99"));
        // Reconstructed vectors should have correct dimension.
        for e in &ds.embeddings {
            assert_eq!(e.vector.len(), 32, "reconstructed dim must be 32");
        }
    }
}
