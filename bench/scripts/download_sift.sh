#!/usr/bin/env python3
"""Download SIFT1M dataset (not committed)."""
import os, urllib.request, zipfile

URLS = {
    "sift_base.fvecs": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
}
print("Please download SIFT1M manually from http://corpus-texmex.irisa.fr/ and extract to bench/data/sift/")
print("Files expected: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs")
EOF