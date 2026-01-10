#!/usr/bin/env python3
"""
Performance Analysis: Catapult vs No-Catapult
Parses benchmark logs and generates heatmap visualizations as png files.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_log_file(filepath):
    """
    Parse a benchmark log file and extract metrics.

    Returns:
        dict: Nested dictionary with structure {threads: {beam_width: {metric: value}}}
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Find all configuration blocks
    config_pattern = r"--- Configuration: threads=(\d+), beam_width=(\d+) ---([\s\S]*?)(?=--- Configuration:|={20,}|$)"
    configs = re.findall(config_pattern, content)

    results = {}

    for threads_str, beam_width_str, block in configs:
        threads = int(threads_str)
        beam_width = int(beam_width_str)

        # Extract nodes expanded (avg per search)
        nodes_match = re.search(r"Avg per search: ([\d.]+) nodes expanded", block)
        nodes_expanded = float(nodes_match.group(1)) if nodes_match else None

        # Extract QPS
        qps_match = re.search(r"\(([\.\d]+) QPS\)", block)
        qps = float(qps_match.group(1)) if qps_match else None

        # Store results
        if threads not in results:
            results[threads] = {}

        results[threads][beam_width] = {"nodes_expanded": nodes_expanded, "qps": qps}

    return results


def results_to_matrix(results, metric):
    """
    Convert nested dict results to a 2D numpy array for heatmap plotting.

    Args:
        results: Nested dict {threads: {beam_width: {metric: value}}}
        metric: String name of the metric to extract ('qps' or 'nodes_expanded')

    Returns:
        matrix: 2D numpy array
        threads_list: List of thread values (y-axis)
        beam_widths_list: List of beam_width values (x-axis)
    """
    threads_list = sorted(results.keys())
    beam_widths_list = sorted(
        set(bw for thread_data in results.values() for bw in thread_data.keys())
    )

    matrix = np.zeros((len(threads_list), len(beam_widths_list)))

    for i, threads in enumerate(threads_list):
        for j, beam_width in enumerate(beam_widths_list):
            if beam_width in results[threads]:
                value = results[threads][beam_width][metric]
                matrix[i, j] = value if value is not None else 0

    return matrix, threads_list, beam_widths_list


def plot_heatmap(
    matrix,
    threads_list,
    beam_widths_list,
    title,
    output_file,
    cmap="YlOrRd",
    fmt=".0f",
    cbar_label="",
    custom_fmt=None,
):
    """
    Plot a heatmap and save to png file.
    """
    plt.figure(figsize=(10, 6))

    if custom_fmt:
        # Create custom annotations
        annot_data = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                annot_data[i, j] = custom_fmt.format(matrix[i, j])

        ax = sns.heatmap(
            matrix,
            xticklabels=beam_widths_list,
            yticklabels=threads_list,
            annot=annot_data,
            fmt="",
            cmap=cmap,
            cbar_kws={"label": cbar_label},
        )
    else:
        ax = sns.heatmap(
            matrix,
            xticklabels=beam_widths_list,
            yticklabels=threads_list,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            cbar_kws={"label": cbar_label},
        )

    plt.xlabel("Beam Width", fontsize=12)
    plt.ylabel("Threads", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def main():
    # Parse both log files
    print("Parsing log files...")
    cata_results = parse_log_file("log-cata.txt")
    nocata_results = parse_log_file("log-nocata.txt")

    print(f"Catapult configs parsed: {sum(len(v) for v in cata_results.values())}")
    print(f"No-Catapult configs parsed: {sum(len(v) for v in nocata_results.values())}")
    print()

    # Convert to matrices
    qps_matrix, threads, beam_widths = results_to_matrix(cata_results, "qps")
    nodes_matrix, _, _ = results_to_matrix(cata_results, "nodes_expanded")
    qps_matrix_nc, threads_nc, beam_widths_nc = results_to_matrix(nocata_results, "qps")
    nodes_matrix_nc, _, _ = results_to_matrix(nocata_results, "nodes_expanded")

    # Generate heatmaps
    print("Generating heatmaps...")

    plot_heatmap(
        qps_matrix,
        threads,
        beam_widths,
        "Catapult: QPS vs Threads and Beam Width",
        "catapult_qps.png",
        cmap="RdYlGn",
        fmt=".0f",
        cbar_label="Queries Per Second",
    )

    plot_heatmap(
        nodes_matrix,
        threads,
        beam_widths,
        "Catapult: Nodes Expanded vs Threads and Beam Width",
        "catapult_nodes.png",
        cmap="YlOrRd",
        fmt=".2f",
        cbar_label="Avg Nodes Expanded",
    )

    plot_heatmap(
        qps_matrix_nc,
        threads_nc,
        beam_widths_nc,
        "No-Catapult: QPS vs Threads and Beam Width",
        "nocatapult_qps.png",
        cmap="RdYlGn",
        fmt=".0f",
        cbar_label="Queries Per Second",
    )

    plot_heatmap(
        nodes_matrix_nc,
        threads_nc,
        beam_widths_nc,
        "No-Catapult: Nodes Expanded vs Threads and Beam Width",
        "nocatapult_nodes.png",
        cmap="YlOrRd",
        fmt=".2f",
        cbar_label="Avg Nodes Expanded",
    )

    # Comparison plots
    speedup_matrix = qps_matrix / qps_matrix_nc
    plot_heatmap(
        speedup_matrix,
        threads,
        beam_widths,
        "QPS Speedup: Catapult vs No-Catapult",
        "speedup_qps.png",
        cmap="RdYlGn",
        fmt=".2f",
        cbar_label="Speedup Factor",
    )

    nodes_reduction = nodes_matrix / nodes_matrix_nc
    plot_heatmap(
        nodes_reduction,
        threads,
        beam_widths,
        "Nodes Expanded Ratio: Catapult / No-Catapult",
        "nodes_reduction.png",
        cmap="RdYlGn_r",
        fmt=".2f",
        cbar_label="Ratio (lower is better)",
    )

    # Relative improvement in QPS (percentage)
    qps_improvement = ((qps_matrix - qps_matrix_nc) / qps_matrix_nc) * 100
    plot_heatmap(
        qps_improvement,
        threads,
        beam_widths,
        "QPS Improvement: Catapult over No-Catapult (%)",
        "qps_improvement.png",
        cmap="Greens",
        fmt=".1f",
        cbar_label="Improvement (%)",
        custom_fmt="+{:.1f}%",
    )

    # Relative improvement in nodes expanded (percentage - negative is better, so invert)
    nodes_improvement = -((nodes_matrix - nodes_matrix_nc) / nodes_matrix_nc) * 100
    plot_heatmap(
        nodes_improvement,
        threads,
        beam_widths,
        "Nodes Expanded Reduction: Catapult vs No-Catapult (%)",
        "nodes_improvement.png",
        cmap="Greens",
        fmt=".1f",
        cbar_label="Reduction (% - positive is better)",
        custom_fmt="-{:.1f}%",
    )

    # Print summary statistics
    print()
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nQPS Speedup (Catapult vs No-Catapult):")
    print(f"  Average: {speedup_matrix.mean():.2f}x")
    print(f"  Maximum: {speedup_matrix.max():.2f}x")
    print(f"  Minimum: {speedup_matrix.min():.2f}x")

    print(f"\nNodes Expanded Reduction:")
    print(f"  Average ratio: {nodes_reduction.mean():.2f}")
    print(f"  Best (lowest): {nodes_reduction.min():.2f}")
    print(f"  Worst (highest): {nodes_reduction.max():.2f}")

    # Find best configurations
    best_qps_idx = np.unravel_index(qps_matrix.argmax(), qps_matrix.shape)
    best_speedup_idx = np.unravel_index(speedup_matrix.argmax(), speedup_matrix.shape)

    print(f"\nBest Catapult QPS Configuration:")
    print(
        f"  Threads={threads[best_qps_idx[0]]}, Beam Width={beam_widths[best_qps_idx[1]]}"
    )
    print(f"  QPS: {qps_matrix[best_qps_idx]:.0f}")

    print(f"\nBest Speedup Configuration:")
    print(
        f"  Threads={threads[best_speedup_idx[0]]}, Beam Width={beam_widths[best_speedup_idx[1]]}"
    )
    print(f"  Speedup: {speedup_matrix[best_speedup_idx]:.2f}x")
    print()
    print("All png files saved to current directory!")


if __name__ == "__main__":
    main()
