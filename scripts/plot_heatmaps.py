#!/usr/bin/env python3
"""
Plot heatmaps comparing catapulted vs non-catapulted execution metrics.

This script reads execution logs and creates two heatmaps:
1. Raw values from catapulted.json
2. Improvement percentage of catapulted over notcatapulted

Usage:
    python plot_heatmaps.py <metric_key> [--output output.png]

Example:
    python plot_heatmaps.py qps
    python plot_heatmaps.py elapsed_secs --output elapsed_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_json_file(filepath: Path) -> Dict:
    """Load and parse a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def extract_data(
    results: List[Dict], metric_key: str
) -> Dict[Tuple[int, int, int], float]:
    """
    Extract metric values indexed by (num_threads, beam_width, seed).

    Args:
        results: List of result dictionaries
        metric_key: The metric to extract (e.g., 'qps', 'elapsed_secs')

    Returns:
        Dictionary mapping (num_threads, beam_width, seed) -> metric_value
    """
    data = {}
    for result in results:
        key = (result["num_threads"], result["beam_width"], result["seed"])
        if metric_key not in result:
            print(f"Warning: metric '{metric_key}' not found in result: {result}")
            continue
        data[key] = result[metric_key]
    return data


def average_across_seeds(
    data: Dict[Tuple[int, int, int], float],
) -> Dict[Tuple[int, int], float]:
    """
    Average metric values across different seeds.

    Args:
        data: Dictionary mapping (num_threads, beam_width, seed) -> metric_value

    Returns:
        Dictionary mapping (num_threads, beam_width) -> averaged_value
    """
    # Group by (num_threads, beam_width)
    grouped = {}
    for (num_threads, beam_width, seed), value in data.items():
        key = (num_threads, beam_width)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(value)

    # Average across seeds
    averaged = {}
    for key, values in grouped.items():
        averaged[key] = np.mean(values)

    return averaged


def create_heatmap_matrix(
    data: Dict[Tuple[int, int], float],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Create a 2D matrix for heatmap plotting.

    Args:
        data: Dictionary mapping (num_threads, beam_width) -> value

    Returns:
        Tuple of (matrix, num_threads_labels, beam_width_labels)
    """
    # Get unique sorted values for each dimension
    num_threads_set = sorted(set(key[0] for key in data.keys()))
    beam_width_set = sorted(set(key[1] for key in data.keys()))

    # Create matrix - flipped so num_threads is on y-axis
    matrix = np.full((len(num_threads_set), len(beam_width_set)), np.nan)

    # Fill matrix - swapped indices so num_threads is rows (y-axis)
    for (num_threads, beam_width), value in data.items():
        i = num_threads_set.index(num_threads)
        j = beam_width_set.index(beam_width)
        matrix[i, j] = value

    return matrix, num_threads_set, beam_width_set


def calculate_improvement_pct(
    catapulted_data: Dict[Tuple[int, int], float],
    notcatapulted_data: Dict[Tuple[int, int], float],
    metric_key: str,
) -> Dict[Tuple[int, int], float]:
    """
    Calculate percentage improvement of catapulted over notcatapulted.

    For metrics where lower is better (e.g., elapsed_secs), improvement is:
        ((notcatapulted - catapulted) / notcatapulted) * 100

    For metrics where higher is better (e.g., qps), improvement is:
        ((catapulted - notcatapulted) / notcatapulted) * 100

    Args:
        catapulted_data: Averaged catapulted metrics
        notcatapulted_data: Averaged notcatapulted metrics
        metric_key: The metric being compared

    Returns:
        Dictionary mapping (num_threads, beam_width) -> improvement_percentage
    """
    # Metrics where lower values are better
    lower_is_better = {"elapsed_secs", "avg_dists_computed", "avg_nodes_visited"}

    improvement = {}
    for key in catapulted_data.keys():
        if key not in notcatapulted_data:
            print(f"Warning: key {key} not found in notcatapulted data")
            continue

        cat_val = catapulted_data[key]
        notcat_val = notcatapulted_data[key]

        if notcat_val == 0:
            print(f"Warning: notcatapulted value is 0 for key {key}, skipping")
            continue

        if metric_key in lower_is_better:
            # For metrics where lower is better
            improvement[key] = ((notcat_val - cat_val) / notcat_val) * 100
        else:
            # For metrics where higher is better
            improvement[key] = ((cat_val - notcat_val) / notcat_val) * 100

    return improvement


def plot_heatmaps(
    catapulted_matrix: np.ndarray,
    improvement_matrix: np.ndarray,
    num_threads_labels: List[int],
    beam_width_labels: List[int],
    metric_key: str,
    output_path: Path = None,
):
    """
    Plot two heatmaps side by side.

    Args:
        catapulted_matrix: Raw catapulted values
        improvement_matrix: Improvement percentage values (None if no comparison data)
        num_threads_labels: Labels for x-axis
        beam_width_labels: Labels for y-axis
        metric_key: The metric being plotted
        output_path: Path to save the figure (optional)
    """
    # Determine number of subplots based on whether we have comparison data
    num_plots = 1 if improvement_matrix is None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

    # Make axes iterable even if single plot
    if num_plots == 1:
        axes = [axes]

    # First heatmap: Raw catapulted values
    sns.heatmap(
        catapulted_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=beam_width_labels,
        yticklabels=num_threads_labels,
        ax=axes[0],
        cbar_kws={"label": metric_key},
    )
    axes[0].set_title(f"Catapulted: {metric_key}", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("beam_width", fontsize=12)
    axes[0].set_ylabel("num_threads", fontsize=12)

    # Second heatmap: Improvement percentage (only if comparison data exists)
    if improvement_matrix is not None:
        # Create custom annotations with +/- and % formatting
        annot_matrix = np.empty_like(improvement_matrix, dtype=object)
        for i in range(improvement_matrix.shape[0]):
            for j in range(improvement_matrix.shape[1]):
                val = improvement_matrix[i, j]
                if np.isnan(val):
                    annot_matrix[i, j] = ""
                elif val >= 0:
                    annot_matrix[i, j] = f"+{val:.2f}%"
                else:
                    annot_matrix[i, j] = f"{val:.2f}%"

        # Use a diverging colormap centered at 0
        vmax = np.nanmax(np.abs(improvement_matrix))
        sns.heatmap(
            improvement_matrix,
            annot=annot_matrix,
            fmt="",
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=beam_width_labels,
            yticklabels=num_threads_labels,
            ax=axes[1],
            cbar_kws={"label": "Improvement %"},
        )
        axes[1].set_title(
            f"Improvement: Catapulted vs NotCatapulted (%)",
            fontsize=14,
            fontweight="bold",
        )
        axes[1].set_xlabel("beam_width", fontsize=12)
        axes[1].set_ylabel("num_threads", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig(f"{metric_key}_heatmaps.png", dpi=300, bbox_inches="tight")
        print(f"Saved plot to {metric_key}_heatmaps.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot heatmaps comparing catapulted vs non-catapulted execution metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s qps
  %(prog)s elapsed_secs --output elapsed_comparison.png
  %(prog)s avg_dists_computed

Available metrics:
  - qps: Queries per second
  - elapsed_secs: Elapsed time in seconds
  - avg_dists_computed: Average distances computed
  - avg_nodes_visited: Average nodes visited
  - catapult_usage_pct: Catapult usage percentage
  - avg_catapults_added: Average catapults added
        """,
    )
    parser.add_argument(
        "metric_key",
        type=str,
        help="The metric to plot (e.g., qps, elapsed_secs, avg_dists_computed)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename for the plot (default: <metric_key>_heatmaps.png)",
    )
    parser.add_argument(
        "--catapulted",
        type=str,
        default="./execution-logs/catapulted.json",
        help="Path to catapulted.json file",
    )
    parser.add_argument(
        "--notcatapulted",
        type=str,
        default="./execution-logs/notcatapulted.json",
        help="Path to notcatapulted.json file",
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    catapulted_path = Path(args.catapulted)
    notcatapulted_path = Path(args.notcatapulted)
    output_path = Path(args.output) if args.output else None

    # Check if files exist
    if not catapulted_path.exists():
        print(f"Error: {catapulted_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load catapulted data
    catapulted_json = load_json_file(catapulted_path)
    catapulted_results = catapulted_json.get("results", [])

    if not catapulted_results:
        print("Error: No results found in catapulted.json", file=sys.stderr)
        sys.exit(1)

    # Load notcatapulted data if available
    notcatapulted_results = []
    has_comparison_data = False
    if notcatapulted_path.exists():
        notcatapulted_json = load_json_file(notcatapulted_path)
        notcatapulted_results = notcatapulted_json.get("results", [])
        if notcatapulted_results:
            # Check if the metric exists and is not null in notcatapulted data
            has_valid_metric = any(
                result.get(args.metric_key) is not None
                for result in notcatapulted_results
            )
            if has_valid_metric:
                has_comparison_data = True
            else:
                print(
                    f"Warning: Metric '{args.metric_key}' is null in all notcatapulted results, skipping comparison plot"
                )
        else:
            print(
                "Warning: No results found in notcatapulted.json, skipping comparison plot"
            )
    else:
        print(f"Warning: {notcatapulted_path} does not exist, skipping comparison plot")

    print(f"Processing metric: {args.metric_key}")
    print(f"Catapulted results: {len(catapulted_results)}")
    if has_comparison_data:
        print(f"NotCatapulted results: {len(notcatapulted_results)}")

    # Extract metric data
    catapulted_data = extract_data(catapulted_results, args.metric_key)

    # Average across seeds
    print("Averaging across seeds...")
    catapulted_avg = average_across_seeds(catapulted_data)

    # Calculate improvement only if we have comparison data
    improvement_matrix = None
    if has_comparison_data:
        notcatapulted_data = extract_data(notcatapulted_results, args.metric_key)
        notcatapulted_avg = average_across_seeds(notcatapulted_data)

        print("Calculating improvement percentages...")
        improvement = calculate_improvement_pct(
            catapulted_avg, notcatapulted_avg, args.metric_key
        )
        improvement_matrix, _, _ = create_heatmap_matrix(improvement)

    # Create matrices for heatmaps
    print("Creating heatmap matrices...")
    catapulted_matrix, num_threads_labels, beam_width_labels = create_heatmap_matrix(
        catapulted_avg
    )

    # Plot heatmaps
    print("Generating plots...")
    plot_heatmaps(
        catapulted_matrix,
        improvement_matrix,
        num_threads_labels,
        beam_width_labels,
        args.metric_key,
        output_path,
    )

    print("Done!")


if __name__ == "__main__":
    main()
