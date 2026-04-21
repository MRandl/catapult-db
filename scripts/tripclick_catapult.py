import json
from collections import defaultdict


def compute_avg_qps(filepath):
    with open(filepath) as f:
        data = json.load(f)

    groups = defaultdict(list)
    for entry in data["results"]:
        key = (entry["num_threads"], entry["beam_width"])
        groups[key].append(
            (
                entry["qps"],
                entry["avg_dists_computed"],
                entry["avg_nodes_visited"],
                # entry["catapult_usage_pct"],
            )
        )

    return {
        k: ([s / len(v) for s in map(sum, zip(*v))], len(v)) for k, v in groups.items()
    }


if __name__ == "__main__":
    import csv
    import sys

    result = compute_avg_qps(sys.argv[1])
    with open("tripclick_vanilla.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "num_threads",
                "beam_width",
                "count",
                "avg_qps",
                "avg_dists_computed",
                "avg_nodes_visited",
                "catapult_usage_pct",
            ]
        )
        for (threads, beam_width), (
            [avg_qps, avg_dists, avg_nodes],  # , avg_pct],
            count,
        ) in sorted(result.items()):
            writer.writerow(
                [
                    threads,
                    beam_width,
                    count,
                    f"{avg_qps:.2f}",
                    f"{avg_dists:.2f}",
                    f"{avg_nodes:.2f}",
                    # f"{avg_pct:.2f}",
                ]
            )
