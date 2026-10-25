# ARCHITECTURE.md

## Overview

vectordb is an embedded vector database. Core crate provides distance kernels, flat and HNSW indices, storage engine, collection API. Server crate exposes HTTP/gRPC. Python bindings via PyO3.

## Components

- Distance: SIMD kernels with runtime detection.
- HNSW: multi-layer graph, neighbor heuristic, tombstones, compaction.
- Storage: segments, WAL, mmap, redb payloads.
- Collection: orchestration + filtered search.

## Concurrency

Coarse-grained RwLock.

## Durability

WAL with CRC32, fsync policies, crash recovery.
EOF