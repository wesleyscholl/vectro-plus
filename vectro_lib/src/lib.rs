use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};

pub mod hnsw;
pub use hnsw::HnswIndex;

pub mod pq;
pub use pq::ProductQuantizer;

pub mod nf4;
pub use nf4::Nf4Quantizer;

pub mod rq;
pub use rq::ResidualQuantizer;

pub mod auto_quantize;
pub use auto_quantize::{auto_select_format, AutoQuantizeResult, QuantFormat};

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Compute recall@k between an exact (ground-truth) result set and an approximate result set.
///
/// Both slices should contain the top-k IDs in descending similarity order. Only the first
/// `k` entries of each slice are considered; if a slice has fewer than `k` entries the
/// missing entries count as misses.
///
/// # Formula
/// recall@k = |approx_top_k ∩ exact_top_k| / k
///
/// # Examples
/// ```
/// use vectro_lib::recall_at_k;
/// let exact  = ["a", "b", "c", "d", "e"].map(String::from);
/// let approx = ["a", "c", "e", "f", "g"].map(String::from);
/// assert!((recall_at_k(&exact, &approx, 5) - 0.6).abs() < 1e-6);
/// ```
pub fn recall_at_k(exact: &[String], approx: &[String], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let gt: std::collections::HashSet<&str> =
        exact.iter().take(k).map(String::as_str).collect();
    let hits = approx
        .iter()
        .take(k)
        .filter(|id| gt.contains(id.as_str()))
        .count();
    hits as f64 / k as f64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingDataset {
    pub embeddings: Vec<Embedding>,
}

impl Default for EmbeddingDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingDataset {
    pub fn new() -> Self {
        Self { embeddings: vec![] }
    }

    pub fn add(&mut self, e: Embedding) {
        self.embeddings.push(e);
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut f = File::create(path)?;
        let data = bincode::serialize(self)?;
        f.write_all(&data)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let mut f = File::open(path)?;
        // detect if file is our streaming format by checking header
        let header  = b"VECTRO+STREAM1\n";
        let qheader = b"VECTRO+QSTREAM1\n";
        let pqheader = b"VECTRO+PQSTREAM1\n";
        // pqheader is longest at 17 bytes
        let max_len = pqheader.len();
        let mut sig = vec![0u8; max_len];
        let n = f.read(&mut sig)?;
        // reset cursor so each branch can read from the start as needed
        f.seek(SeekFrom::Start(0))?;
        if n >= header.len() && sig.as_slice()[..header.len()] == *header {
            // streaming format: multiple length-prefixed bincode Embedding entries
            let mut embeddings = Vec::new();
            // consume header
            let mut _hdr = vec![0u8; header.len()];
            let _ = f.read_exact(&mut _hdr);
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let e: Embedding = bincode::deserialize(&buf)?;
                        embeddings.push(e);
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // maybe quantized stream
        // rewind and check for qheader
        f.seek(SeekFrom::Start(0))?;
        let mut qsig = vec![0u8; qheader.len()];
        if f.read_exact(&mut qsig).is_ok()
            && qsig.as_slice() == qheader {
                // we've consumed the header already; proceed (tables follow)
                // quantized stream layout: u32(table_count) u32(dim) [tables serialized as bincode] then repeated len-prefixed records: bincode((id:String, qvec:Vec<u8>))
                let mut buf4 = [0u8; 4];
                f.read_exact(&mut buf4)?;
                let _table_count = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4)?;
                let _dim = u32::from_le_bytes(buf4) as usize;
                // read tables blob length
                f.read_exact(&mut buf4)?;
                let tables_len = u32::from_le_bytes(buf4) as usize;
                let mut tblbuf = vec![0u8; tables_len];
                f.read_exact(&mut tblbuf)?;
                let _tables: Vec<crate::search::quant::QuantTable> = bincode::deserialize(&tblbuf)?;

                // now read quantized entries
                let mut embeddings = Vec::new();
                loop {
                    let mut lenbuf = [0u8; 4];
                    match f.read_exact(&mut lenbuf) {
                        Ok(_) => {
                            let len = u32::from_le_bytes(lenbuf) as usize;
                            let mut buf = vec![0u8; len];
                            f.read_exact(&mut buf)?;
                            let rec: (String, Vec<u8>) = bincode::deserialize(&buf)?;
                            // dequantize
                            let id = rec.0;
                            let qv = rec.1;
                            // naive dequantize each value using tables sequentially
                            // if tables length mismatch, we'll skip
                            let mut v = Vec::with_capacity(qv.len());
                            for (i, &b) in qv.iter().enumerate() {
                                let val = _tables[i].dequantize(b);
                                v.push(val);
                            }
                            embeddings.push(Embedding::new(id, v));
                        }
                        Err(_) => break,
                    }
                }
                return Ok(EmbeddingDataset { embeddings });
            }

        // ─── NF4STREAM1 ───────────────────────────────────────────────────
        let nf4header = b"VECTRO+NF4STREAM1\n";
        f.seek(SeekFrom::Start(0))?;
        let mut nf4sig = vec![0u8; nf4header.len()];
        if f.read_exact(&mut nf4sig).is_ok() && nf4sig.as_slice() == nf4header {
            // Layout: [u32 dim][u32 n_vecs] then records [bincode((id, packed: Vec<u8>, scale: f32))]
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let dim = u32::from_le_bytes(buf4) as usize;
            f.read_exact(&mut buf4)?;
            let _n_vecs = u32::from_le_bytes(buf4) as usize;
            let q = crate::nf4::Nf4Quantizer::new();
            let mut embeddings = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let (id, packed, scale): (String, Vec<u8>, f32) =
                            bincode::deserialize(&buf)?;
                        let v = q.decode_single(&packed, scale, dim);
                        embeddings.push(Embedding::new(id, v));
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // ─── RQSTREAM1 ────────────────────────────────────────────────────
        let rqheader = b"VECTRO+RQSTREAM1\n";
        f.seek(SeekFrom::Start(0))?;
        let mut rqsig = vec![0u8; rqheader.len()];
        if f.read_exact(&mut rqsig).is_ok() && rqsig.as_slice() == rqheader {
            // Layout: [u32 rq_blob_len][rq_blob] then records [u32 rec_len][bincode((id, codes))]
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let rq_blob_len = u32::from_le_bytes(buf4) as usize;
            let mut rqbuf = vec![0u8; rq_blob_len];
            f.read_exact(&mut rqbuf)?;
            let rq: crate::rq::ResidualQuantizer = bincode::deserialize(&rqbuf)?;
            let mut embeddings = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let (id, codes): (String, Vec<Vec<u8>>) =
                            bincode::deserialize(&buf)?;
                        let v = rq.decode(&codes);
                        embeddings.push(Embedding::new(id, v));
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // fallback: rewind and read whole-file bincode
        f.seek(SeekFrom::Start(0))?;

        // ─── PQSTREAM1 ────────────────────────────────────────────────────
        let mut pqsig = vec![0u8; pqheader.len()];
        if f.read_exact(&mut pqsig).is_ok() && pqsig.as_slice() == pqheader {
            // Layout: [u32 pq_blob_len][pq_blob] then records [u32 rec_len][bincode((id, code))]
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let pq_blob_len = u32::from_le_bytes(buf4) as usize;
            let mut pqbuf = vec![0u8; pq_blob_len];
            f.read_exact(&mut pqbuf)?;
            let pq: crate::pq::ProductQuantizer = bincode::deserialize(&pqbuf)?;

            let mut embeddings = Vec::new();
            loop {
                let mut lenbuf = [0u8; 4];
                match f.read_exact(&mut lenbuf) {
                    Ok(_) => {
                        let len = u32::from_le_bytes(lenbuf) as usize;
                        let mut buf = vec![0u8; len];
                        f.read_exact(&mut buf)?;
                        let (id, code): (String, Vec<u8>) = bincode::deserialize(&buf)?;
                        // Reconstruct approximate f32 vector from PQ code.
                        let v = pq.decode(&code);
                        embeddings.push(Embedding::new(id, v));
                    }
                    Err(_) => break,
                }
            }
            return Ok(EmbeddingDataset { embeddings });
        }

        // ─── bincode fallback ─────────────────────────────────────────────
        f.seek(SeekFrom::Start(0))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let ds: EmbeddingDataset = bincode::deserialize(&buf)?;
        Ok(ds)
    }
}

// ── Streaming iterator over STREAM1 files ─────────────────────────────────

/// Lazy iterator that yields one [`Embedding`] at a time from a STREAM1 file.
/// Avoids loading the entire dataset into memory.
pub struct StreamIter {
    reader: std::io::BufReader<std::fs::File>,
}

impl Iterator for StreamIter {
    type Item = anyhow::Result<Embedding>;

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::Read;
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(anyhow::anyhow!("stream read: {}", e))),
            Ok(()) => {}
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(anyhow::anyhow!("stream record: {}", e)));
        }
        Some(bincode::deserialize(&payload).map_err(|e| anyhow::anyhow!("deserialize: {}", e)))
    }
}

/// Open a STREAM1 file and return a lazy [`StreamIter`].
///
/// Reads and validates the 15-byte magic header, then yields one
/// [`Embedding`] per call to `next()` without buffering the entire file.
pub fn iter_stream(path: &str) -> anyhow::Result<StreamIter> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 15];
    f.read_exact(&mut hdr)?;
    anyhow::ensure!(&hdr == b"VECTRO+STREAM1\n", "not a STREAM1 file: {}", path);
    Ok(StreamIter { reader: std::io::BufReader::new(f) })
}

/// Search utilities
pub mod search {
    use crate::Embedding;
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;
    

    /// Compute dot product between two same-length slices
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute L2 norm of a vector
    fn norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity between two vectors (returns -1..1)
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return -1.0;
        }
        let denom = norm(a) * norm(b);
        if denom == 0.0 {
            return -1.0;
        }
        dot(a, b) / denom
    }

    /// Naive top-k nearest neighbors by cosine similarity.
    /// Returns a Vec of (id, score) sorted by descending score.
    pub fn top_k<'a>(
        dataset: &'a [Embedding],
        query: &[f32],
        k: usize,
    ) -> Vec<(&'a str, f32)> {
        #[cfg(not(target_arch = "wasm32"))]
        let mut scores: Vec<(&str, f32)> = dataset
            .par_iter()
            .map(|e| (e.id.as_str(), cosine(&e.vector, query)))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let mut scores: Vec<(&str, f32)> = dataset
            .iter()
            .map(|e| (e.id.as_str(), cosine(&e.vector, query)))
            .collect();

        // sort descending by score
        #[cfg(not(target_arch = "wasm32"))]
        scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        #[cfg(target_arch = "wasm32")]
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter().take(k).collect()
    }

    /// A simple search index that caches normalized vectors for fast cosine scoring.
    /// It owns a normalized copy of all vectors and the ids.
    pub struct SearchIndex {
        ids: Vec<String>,
        normalized: Vec<Vec<f32>>,
        dim: usize,
    }

    impl SearchIndex {
        /// Build an index from an embedding slice by normalizing each vector.
        pub fn from_dataset(dataset: &[Embedding]) -> Self {
            let mut ids = Vec::with_capacity(dataset.len());
            let mut normalized = Vec::with_capacity(dataset.len());
            let mut dim = 0usize;

            for e in dataset {
                if dim == 0 {
                    dim = e.vector.len();
                }
                ids.push(e.id.clone());
                // normalize; handle zero-norm vectors
                let n = norm(&e.vector);
                if n == 0.0 {
                    normalized.push(vec![0.0; e.vector.len()]);
                } else {
                    normalized.push(e.vector.iter().map(|v| v / n).collect());
                }
            }

            Self { ids, normalized, dim }
        }

        /// Single query top-k using the cached normalized vectors. Query will be normalized.
        pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
            if query.len() != self.dim {
                return vec![];
            }
            let qnorm = norm(query);
            if qnorm == 0.0 {
                return vec![];
            }
            let q: Vec<f32> = query.iter().map(|v| v / qnorm).collect();

            #[cfg(not(target_arch = "wasm32"))]
            let mut scores: Vec<(&str, f32)> = self
                .normalized
                .par_iter()
                .zip(self.ids.par_iter())
                .map(|(vec, id)| (id.as_str(), dot(vec, &q)))
                .collect();
            #[cfg(target_arch = "wasm32")]
            let mut scores: Vec<(&str, f32)> = self
                .normalized
                .iter()
                .zip(self.ids.iter())
                .map(|(vec, id)| (id.as_str(), dot(vec, &q)))
                .collect();

            #[cfg(not(target_arch = "wasm32"))]
            scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            #[cfg(target_arch = "wasm32")]
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.into_iter().take(k).collect()
        }

        /// Batch top-k: accept multiple queries and return a Vec per query.
        pub fn batch_top_k(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(&str, f32)>> {
            // Parallelize across queries
            #[cfg(not(target_arch = "wasm32"))]
            return queries
                .par_iter()
                .map(|q| self.top_k(q, k))
                .collect();
            #[cfg(target_arch = "wasm32")]
            return queries
                .iter()
                .map(|q| self.top_k(q, k))
                .collect();
        }
    }

    /// Scalar quantization (per-dimension min/max -> u8)
    pub mod quant {
        use serde::{Deserialize, Serialize};
        /// Quantization table per-dimension
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct QuantTable {
            pub min: f32,
            pub max: f32,
        }

        impl QuantTable {
            pub fn new(min: f32, max: f32) -> Self {
                Self { min, max }
            }

            /// Quantize a float in [min, max] to u8
            pub fn quantize(&self, v: f32) -> u8 {
                if self.max <= self.min {
                    return 0u8;
                }
                let t = ((v - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
                (t * 255.0).round() as u8
            }

            /// Dequantize a u8 back to float
            pub fn dequantize(&self, q: u8) -> f32 {
                if self.max <= self.min {
                    return self.min;
                }
                let t = (q as f32) / 255.0;
                self.min + t * (self.max - self.min)
            }
        }

        /// Quantizes a dataset of vectors per-dimension using min/max across dataset
        pub fn quantize_dataset(vectors: &[Vec<f32>]) -> (Vec<QuantTable>, Vec<Vec<u8>>) {
            if vectors.is_empty() {
                return (vec![], vec![]);
            }
            let dim = vectors[0].len();
            let mut mins = vec![f32::INFINITY; dim];
            let mut maxs = vec![f32::NEG_INFINITY; dim];
            for v in vectors {
                for (i, x) in v.iter().enumerate() {
                    if *x < mins[i] { mins[i] = *x }
                    if *x > maxs[i] { maxs[i] = *x }
                }
            }
            let tables: Vec<QuantTable> = mins.into_iter().zip(maxs).map(|(min, max)| QuantTable::new(min, max)).collect();

            let qvecs: Vec<Vec<u8>> = vectors.iter().map(|v| {
                v.iter().enumerate().map(|(i, x)| tables[i].quantize(*x)).collect()
            }).collect();

            (tables, qvecs)
        }
    }

    /// Quantized index that stores u8 vectors with per-dimension quant tables.
    pub struct QuantizedIndex {
        ids: Vec<String>,
        tables: Vec<quant::QuantTable>,
        qvecs: Vec<Vec<u8>>,
        dim: usize,
        // optional cache of normalized dequantized vectors
        normalized_cache: Option<Vec<Vec<f32>>>,
    }

    impl QuantizedIndex {
        pub fn from_dataset(dataset: &[Embedding]) -> Self {
            let ids: Vec<String> = dataset.iter().map(|e| e.id.clone()).collect();
            let vectors: Vec<Vec<f32>> = dataset.iter().map(|e| e.vector.clone()).collect();
            let (tables, qvecs) = quant::quantize_dataset(&vectors);
            let dim = tables.len();
            Self { ids, tables, qvecs, dim, normalized_cache: None }
        }

        /// Dequantize a u8 vector into f32 vector
        fn dequantize_vec(&self, q: &[u8]) -> Vec<f32> {
            q.iter().enumerate().map(|(i, &b)| self.tables[i].dequantize(b)).collect()
        }

        /// Top-k: dequantize vectors lazily and compute cosine with normalized query
        pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
            if query.len() != self.dim { return vec![]; }
            let qnorm = norm(query);
            if qnorm == 0.0 { return vec![]; }
            let qnormed: Vec<f32> = query.iter().map(|v| v / qnorm).collect();

            #[cfg(not(target_arch = "wasm32"))]
            let mut scores: Vec<(&str, f32)> = match &self.normalized_cache {
                Some(cache) => cache.par_iter().zip(self.ids.par_iter()).map(|(v, id)| {
                    (id.as_str(), dot(v, &qnormed))
                }).collect(),
                None => self.qvecs.par_iter().zip(self.ids.par_iter()).map(|(qv, id)| {
                    let v = self.dequantize_vec(qv);
                    let n = norm(&v);
                    let score = if n == 0.0 { -1.0 } else { dot(&v, &qnormed) / n };
                    (id.as_str(), score)
                }).collect(),
            };
            #[cfg(target_arch = "wasm32")]
            let mut scores: Vec<(&str, f32)> = match &self.normalized_cache {
                Some(cache) => cache.iter().zip(self.ids.iter()).map(|(v, id)| {
                    (id.as_str(), dot(v, &qnormed))
                }).collect(),
                None => self.qvecs.iter().zip(self.ids.iter()).map(|(qv, id)| {
                    let v = self.dequantize_vec(qv);
                    let n = norm(&v);
                    let score = if n == 0.0 { -1.0 } else { dot(&v, &qnormed) / n };
                    (id.as_str(), score)
                }).collect(),
            };

            #[cfg(not(target_arch = "wasm32"))]
            scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            #[cfg(target_arch = "wasm32")]
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.into_iter().take(k).collect()
        }

        pub fn batch_top_k(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(&str, f32)>> {
            #[cfg(not(target_arch = "wasm32"))]
            return queries.par_iter().map(|q| self.top_k(q, k)).collect();
            #[cfg(target_arch = "wasm32")]
            return queries.iter().map(|q| self.top_k(q, k)).collect();
        }

        /// Precompute and cache normalized dequantized vectors to accelerate scoring.
        pub fn precompute_normalized(&mut self) {
            let cache: Vec<Vec<f32>> = self.qvecs.iter().map(|qv| {
                let v = self.dequantize_vec(qv);
                let n = norm(&v);
                if n == 0.0 { v.into_iter().map(|_| 0.0).collect() } else { v.into_iter().map(|x| x / n).collect() }
            }).collect();
            self.normalized_cache = Some(cache);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn roundtrip_save_load() {
        let mut ds = EmbeddingDataset::new();
        ds.add(Embedding::new("one", vec![0.1, 0.2]));
        ds.add(Embedding::new("two", vec![1.0, 2.0]));

        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();
        ds.save(&path).expect("save");

        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "one");
    }

    #[test]
    fn test_embedding_dataset_new_and_len() {
        let ds = EmbeddingDataset::new();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
        
        let mut ds2 = EmbeddingDataset::new();
        ds2.add(Embedding::new("test", vec![1.0, 2.0, 3.0]));
        assert_eq!(ds2.len(), 1);
        assert!(!ds2.is_empty());
    }

    #[test]
    fn test_embedding_dataset_default() {
        let ds: EmbeddingDataset = Default::default();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_streaming_format_load() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();
        
        // Write streaming format manually
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(b"VECTRO+STREAM1\n").expect("write header");
        
        // Write first embedding
        let e1 = Embedding::new("test1", vec![1.0, 2.0]);
        let bytes1 = bincode::serialize(&e1).expect("serialize");
        let len1 = (bytes1.len() as u32).to_le_bytes();
        f.write_all(&len1).expect("write len");
        f.write_all(&bytes1).expect("write bytes");
        
        // Write second embedding
        let e2 = Embedding::new("test2", vec![3.0, 4.0]);
        let bytes2 = bincode::serialize(&e2).expect("serialize");
        let len2 = (bytes2.len() as u32).to_le_bytes();
        f.write_all(&len2).expect("write len");
        f.write_all(&bytes2).expect("write bytes");
        
        drop(f);
        
        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "test1");
        assert_eq!(loaded.embeddings[1].id, "test2");
    }

    #[test]
    fn test_quantized_stream_format_load() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().unwrap().to_string();
        
        // Create embeddings
        let e1 = Embedding::new("qtest1", vec![1.0, 2.0, 3.0]);
        let e2 = Embedding::new("qtest2", vec![4.0, 5.0, 6.0]);
        let vectors = vec![e1.vector.clone(), e2.vector.clone()];
        
        // Quantize
        let (tables, qvecs) = search::quant::quantize_dataset(&vectors);
        
        // Write quantized stream format
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(b"VECTRO+QSTREAM1\n").expect("write header");
        
        // Write table count and dim
        let table_count = (tables.len() as u32).to_le_bytes();
        let dim = (tables.len() as u32).to_le_bytes();
        f.write_all(&table_count).expect("write table count");
        f.write_all(&dim).expect("write dim");
        
        // Serialize and write tables
        let tables_blob = bincode::serialize(&tables).expect("serialize tables");
        let tables_len = (tables_blob.len() as u32).to_le_bytes();
        f.write_all(&tables_len).expect("write tables len");
        f.write_all(&tables_blob).expect("write tables");
        
        // Write quantized embeddings
        for (i, qv) in qvecs.iter().enumerate() {
            let id = if i == 0 { "qtest1" } else { "qtest2" };
            let rec = (id.to_string(), qv.clone());
            let bytes = bincode::serialize(&rec).expect("serialize rec");
            let len = (bytes.len() as u32).to_le_bytes();
            f.write_all(&len).expect("write len");
            f.write_all(&bytes).expect("write bytes");
        }
        
        drop(f);
        
        let loaded = EmbeddingDataset::load(&path).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.embeddings[0].id, "qtest1");
        assert_eq!(loaded.embeddings[1].id, "qtest2");
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        use crate::search::cosine;
        
        // Different lengths
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), -1.0);
        
        // Zero vectors
        let zero = vec![0.0, 0.0];
        let nonzero = vec![1.0, 2.0];
        assert_eq!(cosine(&zero, &nonzero), -1.0);
        assert_eq!(cosine(&nonzero, &zero), -1.0);
        assert_eq!(cosine(&zero, &zero), -1.0);
    }

    #[test]
    fn test_searchindex_zero_norm_query() {
        use crate::search::SearchIndex;
        
        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = SearchIndex::from_dataset(&ds);
        
        // Query with zero norm
        let zero_query = vec![0.0, 0.0];
        let results = idx.top_k(&zero_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_searchindex_from_zero_norm_vectors() {
        use crate::search::SearchIndex;
        
        let a = Embedding::new("zero", vec![0.0, 0.0]);
        let b = Embedding::new("nonzero", vec![1.0, 2.0]);
        let ds = vec![a, b];
        let idx = SearchIndex::from_dataset(&ds);
        
        let query = vec![1.0, 0.0];
        let results = idx.top_k(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_quantized_index_zero_norm() {
        use crate::search::QuantizedIndex;
        
        let a = Embedding::new("zero", vec![0.0, 0.0]);
        let b = Embedding::new("nonzero", vec![1.0, 2.0]);
        let ds = vec![a, b];
        let idx = QuantizedIndex::from_dataset(&ds);
        
        let query = vec![1.0, 0.0];
        let results = idx.top_k(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_quantized_index_zero_query() {
        use crate::search::QuantizedIndex;
        
        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = QuantizedIndex::from_dataset(&ds);
        
        let zero_query = vec![0.0, 0.0];
        let results = idx.top_k(&zero_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_quantized_index_dim_mismatch() {
        use crate::search::QuantizedIndex;
        
        let a = Embedding::new("a", vec![1.0, 2.0]);
        let ds = vec![a];
        let idx = QuantizedIndex::from_dataset(&ds);
        
        let wrong_query = vec![1.0, 2.0, 3.0];
        let results = idx.top_k(&wrong_query, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_quantized_index_with_precompute() {
        use crate::search::QuantizedIndex;
        
        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let ds = vec![a, b];
        let mut idx = QuantizedIndex::from_dataset(&ds);
        
        // Test without precompute
        let query = vec![1.0, 0.0];
        let results1 = idx.top_k(&query, 1);
        assert_eq!(results1[0].0, "a");
        
        // Test with precompute
        idx.precompute_normalized();
        let results2 = idx.top_k(&query, 1);
        assert_eq!(results2[0].0, "a");
    }

    #[test]
    fn test_quant_table_edge_cases() {
        use crate::search::quant::QuantTable;
        
        // Min equals max
        let table = QuantTable::new(1.0, 1.0);
        assert_eq!(table.quantize(1.0), 0u8);
        assert_eq!(table.dequantize(128), 1.0);
        
        // Normal range
        let table2 = QuantTable::new(0.0, 10.0);
        let q = table2.quantize(5.0);
        let deq = table2.dequantize(q);
        assert!((deq - 5.0).abs() < 0.5); // Should be close
    }

    #[test]
    fn test_quantize_empty_dataset() {
        use crate::search::quant::quantize_dataset;
        
        let empty: Vec<Vec<f32>> = vec![];
        let (tables, qvecs) = quantize_dataset(&empty);
        assert!(tables.is_empty());
        assert!(qvecs.is_empty());
    }

    #[test]
    fn cosine_and_topk() {
        use crate::search;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.707, 0.707]);

        let ds = vec![a.clone(), b.clone(), c.clone()];

        // a vs c ~ 0.707
        let sim_ac = search::cosine(&a.vector, &c.vector);
        assert!(sim_ac > 0.7 && sim_ac < 0.72);

        let top2 = search::top_k(&ds, &a.vector, 2);
        assert_eq!(top2.len(), 2);
        // first should be 'a' itself with score 1.0
        assert_eq!(top2[0].0, "a");
        assert!((top2[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn searchindex_topk_and_batch() {
        use crate::search::SearchIndex;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.707, 0.707]);
        let ds = vec![a.clone(), b.clone(), c.clone()];

        let idx = SearchIndex::from_dataset(&ds);

        let q1 = vec![1.0, 0.0];
        let q2 = vec![0.0, 1.0];

        let single = idx.top_k(&q1, 2);
        let batch = idx.batch_top_k(&[q1.clone(), q2.clone()], 2);

        assert_eq!(single.len(), 2);
        assert_eq!(batch.len(), 2);
        assert_eq!(single[0].0, batch[0][0].0);
        assert!((single[0].1 - batch[0][0].1).abs() < 1e-6);
    }

    #[test]
    fn searchindex_dim_mismatch() {
        use crate::search::SearchIndex;

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let ds = vec![a.clone()];
        let idx = SearchIndex::from_dataset(&ds);

        let q = vec![1.0, 0.0, 0.0];
        let res = idx.top_k(&q, 1);
        assert!(res.is_empty());
    }

    #[test]
    fn quantize_roundtrip_and_topk() {
        use crate::search::{quant, QuantizedIndex, SearchIndex};

        let a = Embedding::new("a", vec![1.0, 0.0]);
        let b = Embedding::new("b", vec![0.0, 1.0]);
        let c = Embedding::new("c", vec![0.6, 0.8]);
        let ds = vec![a.clone(), b.clone(), c.clone()];

        // Quantize dataset
        let vectors: Vec<Vec<f32>> = ds.iter().map(|e| e.vector.clone()).collect();
        let (tables, qvecs) = quant::quantize_dataset(&vectors);
        assert_eq!(tables.len(), 2);
        assert_eq!(qvecs.len(), 3);

        // dequantize first vector
        let q0 = &qvecs[0];
        let deq0: Vec<f32> = q0.iter().enumerate().map(|(i, &b)| tables[i].dequantize(b)).collect();
        assert!((deq0[0] - 1.0).abs() < 1e-2 || (deq0[0] - 1.0).abs() < 1e-1);

        // Compare top-k between float index and quantized index
        let float_idx = SearchIndex::from_dataset(&ds);
        let q_idx = QuantizedIndex::from_dataset(&ds);

        let query = vec![0.6f32, 0.8f32];
        let ftop = float_idx.top_k(&query, 3);
        let qtop = q_idx.top_k(&query, 3);

        // top-1 should likely be the same (c)
        assert_eq!(ftop[0].0, qtop[0].0);
    }
}
