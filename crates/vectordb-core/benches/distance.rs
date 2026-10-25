use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vectordb_core::distance::simd;

fn bench_dot(c: &mut Criterion) {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.02).collect();
    c.bench_function("dot_1024", |bencher| {
        bencher.iter(|| simd::dot(black_box(&a), black_box(&b)))
    });
}

fn bench_l2(c: &mut Criterion) {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.02).collect();
    c.bench_function("l2_1024", |bencher| {
        bencher.iter(|| simd::l2_squared(black_box(&a), black_box(&b)))
    });
}

criterion_group!(benches, bench_dot, bench_l2);
criterion_main!(benches);
