import json
from collections import defaultdict

import pandas as pd


def compute_avg_qps(filepath):
    with open(filepath) as f:
        data = json.load(f)

    groups = defaultdict(list)
    for entry in data["results"]:
        key = (entry["bucket_capacity"], entry["num_hashes"])
        groups[key].append(entry["qps"])

    return {k: (sum(v) / len(v), len(v)) for k, v in groups.items()}


if __name__ == "__main__":
    import sys

    unpacked = [(k[0], k[1], *v) for (k, v) in compute_avg_qps(sys.argv[1]).items()]
    pd.DataFrame(
        unpacked, columns=["bucket_capacity", "threads", "throughput", "seeds"]
    ).to_csv("buckets.csv", index=False)
