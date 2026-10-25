# DEPENDENCIES.md

- thiserror: library errors
- serde/serde_json: serialization
- tracing: logging
- rand: RNG for HNSW levels
- rayon: parallel bulk build
- memmap2: mmap segments
- crc32fast: checksums
- redb: payload KV
- bytemuck: safe cast for mmap slices
- axum/tokio/tonic/prost: server
- pyo3/numpy: Python bindings
- clap: CLI
- criterion/proptest/tempfile: dev
EOF