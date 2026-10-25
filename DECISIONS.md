# DECISIONS.md

- ID type: u64 for vector IDs (performance; strings mapped at higher layers).
- Concurrency: coarse-grained RwLock (readers concurrent, writers exclusive) for HNSW; chosen for correctness/simplicity over per-node locking.
- Payload storage: redb (justified: mature embedded KV, transactional).
- Filtered search heuristic: if estimated selectivity < 5% or candidate set < 100, brute-force pre-filter; otherwise HNSW with expanded ef + post-filter and brute-force fallback.
- Cosine: vectors normalized on insert; distance = 1 - dot.
- unsafe outside SIMD: mmap slice creation in storage::segment requires unsafe; isolated with SAFETY comment (exception to rule).
- Optimizations: not yet applied; baseline measured.
EOF