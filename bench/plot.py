#!/usr/bin/env python3
"""Plot recall@10 vs QPS. Requires matplotlib."""
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib not available:", e)
    raise SystemExit(0)

import csv, sys, os
res = os.path.join(os.path.dirname(__file__), "results", "results.csv")
if not os.path.exists(res):
    print("no results.csv yet")
    sys.exit(0)
print("plotting", res)
EOF