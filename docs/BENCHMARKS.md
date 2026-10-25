# BENCHMARKS.md

## Status

Benchmarks not yet measured in this environment. Scripts are provided; results pending.

## Methodology

- Datasets: SIFT1M, GloVe-100 (download scripts in bench/scripts, not committed).
- Compare vectordb vs FAISS (HNSW, IVF-Flat) and Qdrant (Docker).
- Metrics: recall@10 vs QPS (vary ef_search), build time, index size, peak RSS, p50/p95/p99 latency.
- Single-thread and multi-thread.

## Hardware

- CPU: Apple M4 (10 cores)
- RAM: 16 GB
- OS: macOS 26.5 (Darwin 25.5.0, arm64)
- Rust: 1.95.0

## Status

ANN benchmarks vs FAISS/Qdrant: NOT MEASURED (datasets not downloaded in this environment). Scripts provided.

## Results

Distance kernel microbenchmarks (1024-dim, aarch64 NEON):
- dot_1024: ~147 ns
- l2_1024: ~164 ns

ANN benchmarks: not yet measured.
EOF