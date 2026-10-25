#!/usr/bin/env bash
set -euo pipefail
# Download SIFT1M and GloVe-100 datasets (outside git).
DATA_DIR="${1:-bench/data}"
mkdir -p "$DATA_DIR/sift" "$DATA_DIR/glove"
echo "SIFT1M: download from http://corpus-texmex.irisa.fr/ and extract to $DATA_DIR/sift/"
echo "Expected: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs"
echo "GloVe: download from https://nlp.stanford.edu/projects/glove/ (glove.6B.zip) and extract 100d to $DATA_DIR/glove/"
echo "Checksums: NOT VERIFIED (provide sha256 in this script when available)"
EOF