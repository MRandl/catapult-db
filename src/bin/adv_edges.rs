use catapult::{
    fs::Queries,
    numerics::AlignedBlock,
    search::{AdjacencyGraph, RunningMode, graph_algo::FlatSearch},
    sets::catapults::LruSet,
    statistics::Stats,
};
use clap::Parser;
use std::{path::PathBuf, str::FromStr};

const NUM_HASH: usize = 8;
const BUCKET_SIZE: usize = 40;
const LIMITATION: Option<usize> = Some(1_000_000);

/// Adversarial edge analysis: find graph edges that are never useful across a query set.
///
/// An edge (src→dst) is "adversarial" if its distance was computed at least once during
/// beam search but the dst node was never actually inserted into any candidate set —
/// meaning it never contributed to any result. Such edges slow down search without
/// providing benefit.
///
/// Outputs:
///   1. Total edges in the graph
///   2. Unique used edges across all queries
///   3. Adversarial edge count
///   4. Distribution stats on how often adversarial edges were considered
#[derive(Parser, Debug)]
#[command(name = "adv_edges")]
struct Args {
    /// Path to the queries file (numpy format)
    #[arg(short, long)]
    queries: String,

    /// Path to the graph metadata file
    #[arg(short, long)]
    graph: String,

    /// Path to the graph payload/data file
    #[arg(short, long)]
    payload: String,

    /// Beam width for search
    #[arg(long)]
    beam_width: usize,

    /// Seed for LSH initialisation
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

struct ConsiderationCountStats {
    min: u32,
    p25: u32,
    median: u32,
    p75: u32,
    p90: u32,
    p99: u32,
    max: u32,
    mean: f64,
    considered_once: usize,
}

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
    }

    let args = Args::parse();

    eprintln!("Loading queries...");
    let queries: Vec<Vec<AlignedBlock>> =
        Vec::<Vec<AlignedBlock>>::load_from_npy(&args.queries, LIMITATION);

    eprintln!("Loading graph...");
    let graph = AdjacencyGraph::<LruSet, FlatSearch>::load_flat_from_path(
        PathBuf::from_str(&args.graph).unwrap(),
        PathBuf::from_str(&args.payload).unwrap(),
        NUM_HASH,
        BUCKET_SIZE,
        args.seed,
        RunningMode::Vanilla,
    );

    let total_graph_edges: usize = graph.total_edge_count();
    eprintln!(
        "Graph loaded: {} nodes, {} directed edges",
        graph.len(),
        total_graph_edges
    );

    let num_queries = queries.len();
    eprintln!(
        "Running {} queries (single-threaded, tracking enabled)...",
        num_queries
    );

    let mut stats = Stats::new();
    stats.enable_adv_tracking();

    for (i, query) in queries.iter().enumerate() {
        if i % 10_000 == 0 {
            eprintln!("  query {}/{}", i, num_queries);
        }
        graph.beam_search(query, args.beam_width, args.beam_width, &mut stats);
    }

    eprintln!("Search complete. Analysing edges...");

    let tracking = stats
        .adv_tracking()
        .expect("tracking was enabled but is missing");

    let used_edge_count = tracking.used_edges.len();

    // Adversarial = considered at least once but never used.
    let mut adversarial_counts: Vec<u32> = tracking
        .edge_consider_counts
        .iter()
        .filter(|((src, dst), _)| !tracking.used_edges.contains(&(*src, *dst)))
        .map(|(_, &count)| count)
        .collect();

    let adversarial_edge_count = adversarial_counts.len();
    adversarial_counts.sort_unstable();

    let consideration_counts = if adversarial_counts.is_empty() {
        ConsiderationCountStats {
            min: 0,
            p25: 0,
            median: 0,
            p75: 0,
            p90: 0,
            p99: 0,
            max: 0,
            mean: 0.0,
            considered_once: 0,
        }
    } else {
        let sum: u64 = adversarial_counts.iter().map(|&c| c as u64).sum();
        let mean = sum as f64 / adversarial_counts.len() as f64;
        let considered_once = adversarial_counts.iter().filter(|&&c| c == 1).count();
        ConsiderationCountStats {
            min: adversarial_counts[0],
            p25: percentile(&adversarial_counts, 25.0),
            median: percentile(&adversarial_counts, 50.0),
            p75: percentile(&adversarial_counts, 75.0),
            p90: percentile(&adversarial_counts, 90.0),
            p99: percentile(&adversarial_counts, 99.0),
            max: *adversarial_counts.last().unwrap(),
            mean,
            considered_once,
        }
    };

    let adversarial_fraction = if total_graph_edges > 0 {
        adversarial_edge_count as f64 / total_graph_edges as f64
    } else {
        0.0
    };

    println!("beam_width:        {}", args.beam_width);
    println!("num_queries:       {}", num_queries);
    println!("total_graph_edges: {}", total_graph_edges);
    println!("used_edges:        {}", used_edge_count);
    println!(
        "adversarial_edges: {} ({:.2}%)",
        adversarial_edge_count,
        adversarial_fraction * 100.0
    );
    if adversarial_edge_count > 0 {
        println!(
            "consideration_counts: min={} p25={} median={} p75={} p90={} p99={} max={} mean={:.2}",
            consideration_counts.min,
            consideration_counts.p25,
            consideration_counts.median,
            consideration_counts.p75,
            consideration_counts.p90,
            consideration_counts.p99,
            consideration_counts.max,
            consideration_counts.mean,
        );
        println!(
            "considered_once:      {} ({:.2}%)",
            consideration_counts.considered_once,
            consideration_counts.considered_once as f64 / adversarial_edge_count as f64 * 100.0
        );
    }
}
