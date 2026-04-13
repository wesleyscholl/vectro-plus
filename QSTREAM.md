QSTREAM (Vectro+ Quantized Streaming Format)

This file documents the simple streaming format used by `vectro_cli --quantize`.

## VECTRO+QSTREAM1 (scalar quantization)

Header and layout (all numbers little-endian):

- ASCII header: `VECTRO+QSTREAM1\n` (16 bytes)
- u32 table_count: number of quantization tables (number of dimensions)
- u32 dim: repeated dimension count (same as table_count; reserved)
- u32 tables_blob_len: length in bytes of the following bincode blob
- tables_blob: bincode(Vec<QuantTable>) where QuantTable = { min: f32, max: f32 }
- Repeated records: each record is:
  - u32 len (bytes)
  - bincode((id: String, qvec: Vec<u8>))

Notes:
- Each quantized vector stores one u8 per original dimension. QuantTable.quantize maps f32 -> u8 using a linear min/max scaling.
- The format is intentionally simple for streaming and backwards-compatibility with the non-quantized `VECTRO+STREAM1` format, which stores repeated `u32 len + bincode(Embedding)` records after header `VECTRO+STREAM1\n`.
- The loader expects little-endian values and uses `bincode` for typed blobs.

## VECTRO+PQSTREAM1 (Product Quantization)

Header and layout (all numbers little-endian):

- ASCII header: `VECTRO+PQSTREAM1\n` (17 bytes)
- u32 pq_blob_len: byte length of the following bincode blob
- pq_blob: bincode(ProductQuantizer) — contains m, k, dim, and all codebook vectors
- Repeated records: each record is:
  - u32 len (bytes)
  - bincode((id: String, code: Vec<u8>))

Notes:
- `code` has exactly `m` bytes; `code[s] ∈ [0, k)` is the centroid index for subspace `s`.
- `m` and `k` are stored in the `ProductQuantizer` blob — no separate header fields needed.
- Decoding reconstructs a float vector of length `dim`; all decoded vectors are unit-norm.
- ADC (Asymmetric Distance Computation) skips explicit decode: builds an `m × k` dot-product
  lookup table from the query, then scores each code as `Σ table[s][code[s]]`.
- Compression ratio vs raw f32: `4 × dim / m`. E.g. dim=128, m=8 → 64×; dim=768, m=8 → 384×.
  The practical floor before recall degrades is typically m = dim/8 (≥ 16× compression).
- Training uses Lloyd's k-means with rayon-parallel per-subspace assignment (25 iterations default).
- The codebook centroid init is deterministic (evenly-spaced subset of training data) — no RNG.
