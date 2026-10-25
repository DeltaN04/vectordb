# AUDIT.md

## Baseline

- `cargo fmt --all`: clean
- `cargo clippy --workspace --all-targets -- -D warnings`: clean
- `cargo test --workspace`: pass (core 19, server integration 2)
- `cargo doc --workspace --no-deps`: no warnings

## unwrap()/expect() in library (non-test) code

crates/vectordb-core/src/collection.rs:110
crates/vectordb-core/src/index/mod.rs:134,144,145,150,157,191,366,429,475

## unsafe blocks

All in crates/vectordb-core/src/distance.rs (SIMD) with // SAFETY comments, and storage/segment.rs (mmap) with // SAFETY comment. Exception documented in DECISIONS.md.

## PLAN.md classification

| Item | Status | Evidence |
|---|---|---|
| Phase1 distance kernels | DONE-VERIFIED | distance tests + proptest |
| Phase1 flat index | DONE-VERIFIED | flat tests |
| Phase1 benches | DONE-VERIFIED | cargo bench |
| Phase2 HNSW multi-layer | DONE-UNVERIFIED | recall test 2000 vecs only |
| Phase2 concurrency | PARTIAL | coarse RwLock, not per-node |
| Phase2 deletion/compaction | DONE-UNVERIFIED | basic tests |
| Phase2 recall@10 100k | MISSING | no 100k test |
| Phase3 storage | DONE-VERIFIED | storage tests |
| Phase3 WAL fault injection | PARTIAL | partial tail only |
| Phase3 graph checksum | DONE-UNVERIFIED | roundtrip test |
| Phase4 collection API | DONE-VERIFIED | collection tests |
| Phase4 filtered search | DONE-UNVERIFIED | basic test |
| Phase5 server | DONE-VERIFIED | integration tests |
| Phase5 auth/limits/otel/docker | MISSING | |
| Phase6 python | DONE-VERIFIED | manual import |
| Phase7 benchmarks | MISSING | placeholders |
| Phase8 polish | PARTIAL | docs exist |

## Top 10 risk areas

1. HNSW neighbor selection heuristic (likely incorrect dominance condition)
2. HNSW search layer termination / ef handling
3. Concurrency: coarse lock + unwrap on poisoned lock
4. WAL replay: truncated/garbage tail handling
5. mmap alignment/offsets in segment
6. Filtered search heuristic (post-filter may miss)
7. Upsert duplicate handling in HNSW (updates vector but not graph)
8. Idempotency of cosine normalization
9. Storage all_records merge (mutable overrides) correctness
10. Python payload conversion (currently drops payloads)
EOF