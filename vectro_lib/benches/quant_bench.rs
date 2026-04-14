use criterion::{criterion_group, criterion_main, Criterion};
use vectro_lib::{Embedding, pq::ProductQuantizer, search::{SearchIndex, QuantizedIndex}, hnsw::HnswIndex};

// synthetic dataset generator
fn make_dataset(n: usize, dim: usize) -> Vec<Embedding> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            // simple deterministic values
            v.push(((i + d) % 100) as f32 / 100.0);
        }
        out.push(Embedding::new(format!("id_{}", i), v));
    }
    out
}

fn bench_search(c: &mut Criterion) {
    let ds = make_dataset(1000, 64);
    let query = ds[0].vector.clone();

    let float_idx = SearchIndex::from_dataset(&ds);
    let mut qidx = QuantizedIndex::from_dataset(&ds);

    c.bench_function("float_topk", |b| b.iter(|| {
        let _ = float_idx.top_k(&query, 10);
    }));

    c.bench_function("quant_topk_on_the_fly", |b| b.iter(|| {
        let _ = qidx.top_k(&query, 10);
    }));

    qidx.precompute_normalized();
    c.bench_function("quant_topk_precomputed", |b| b.iter(|| {
        let _ = qidx.top_k(&query, 10);
    }));
}

fn bench_pq(c: &mut Criterion) {
    let ds = make_dataset(1000, 64);
    // m=8 subspaces, k=256 centroids, 25 iters — mirrors the recall@10 gate config
    let pq = ProductQuantizer::train(&ds, 8, 256, 25);
    let query = ds[0].vector.clone();
    let codes: Vec<(String, Vec<u8>)> = ds
        .iter()
        .map(|e| (e.id.clone(), pq.encode(&e.vector)))
        .collect();
    c.bench_function("pq_encode", |b| b.iter(|| pq.encode(&query)));
    c.bench_function("pq_decode", |b| {
        let code = pq.encode(&query);
        b.iter(|| pq.decode(&code))
    });
    c.bench_function("pq_adc_topk", |b| b.iter(|| pq.search_adc(&codes, &query, 10)));
}

fn bench_hnsw(c: &mut Criterion) {
    let ds = make_dataset(1000, 64);
    let query = ds[0].vector.clone();

    c.bench_function("hnsw_build_1k", |b| {
        b.iter(|| HnswIndex::build(&ds, 16, 200, 50))
    });

    let hnsw = HnswIndex::build(&ds, 16, 200, 50);
    c.bench_function("hnsw_search_1k", |b| {
        b.iter(|| hnsw.search(&query, 10))
    });
}

criterion_group!(benches, bench_search, bench_pq, bench_hnsw);
criterion_main!(benches);
