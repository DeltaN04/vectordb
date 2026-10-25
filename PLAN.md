# PLAN.md

## vectordb — embedded vector search engine

Working phases. Tick as complete. Deviations logged in DECISIONS.md.

### Phase 1: Foundations
- [x] Metrics: cosine, L2 squared, dot product (normalize on insert for cosine)
- [x] Distance kernels: scalar + SIMD (AVX2/AVX-512/NEON) runtime detection, safe fallback
- [x] Flat (brute-force) index as ground-truth oracle
- [x] Criterion benchmarks for distance kernels
- [x] SIMD vs scalar proptest

### Phase 2: HNSW index
- [x] Multi-layer graph, M, ef_construction, ef_search, neighbor heuristic, level assignment with seed
- [ ] Concurrent searches while inserting (per-node locking)  (coarse lock remains)
- [x] Parallel bulk build (rayon)
- [x] Deletion: tombstones + compaction/rebuild
- [x] Recall@10 >= 0.95 on 100k random vs flat (ignored test); property tests

### Phase 3: Storage and durability
- [x] Segment-based: immutable on-disk + in-memory mutable
- [x] WAL with CRC32 + fsync policy; crash recovery replay
- [x] mmap vector data; versioned HNSW header + checksums
- [x] Payload storage (redb)
- [x] Crash-safety fault-injection tests (partial WAL)
- [x] Compaction

### Phase 4: Collection API and filtered search
- [x] Collection API
- [x] Payload filters + indexes
- [x] Filtered HNSW search heuristic
- [x] Batch upsert/search

### Phase 5: Server
- [x] HTTP (axum) + gRPC (tonic) + health + metrics
- [x] TOML config + env overrides; graceful shutdown
- [ ] Optional API key auth + size limits
- [ ] OTel tracing
- [x] Integration tests
- [ ] Dockerfile + compose

### Phase 6: Python bindings
- [x] PyO3 + maturin package `vectordb`, NumPy-friendly, stubs
- [x] example (pytest not yet)

### Phase 7: Benchmarks
- [x] SIFT1M + GloVe-100 scripts (placeholders)
- [ ] Compare vs FAISS + Qdrant (not measured)
- [x] Metrics + charts + BENCHMARKS.md (partial)
- [ ] Profiling + top-3 optimizations

### Phase 8: Polish
- [x] README, architecture diagram (Mermaid pending)
- [x] CI (fmt, clippy, tests)
- [x] ARCHITECTURE.md, FORMAT.md
- [x] CHANGELOG, license, crates.io metadata

### Stretch
- [ ] IVF-PQ, hybrid search, multi-vector/sparse, snapshot/backup
