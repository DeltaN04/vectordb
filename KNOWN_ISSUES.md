# KNOWN_ISSUES.md

- HNSW uses a coarse RwLock; concurrent inserts block searches (see DECISIONS.md).
- Python bindings drop payloads in upsert (payloads not yet wired).
- ANN benchmarks vs FAISS/Qdrant not measured.
- Filtered search post-filter path may miss results when HNSW traversal doesn't visit enough matching nodes.
EOF