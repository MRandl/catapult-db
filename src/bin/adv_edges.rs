use catapult::{
    fs::Queries,
    numerics::AlignedBlock,
    search::{AdjacencyGraph, RunningMode, graph_algo::FlatSearch},
    sets::catapults::LruSet,
    statistics::{AdvEdgeTracking, Stats},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

const NUM_HASH: usize = 8;
const BUCKET_SIZE: usize = 40;
const LIMITATION: Option<usize> = None;
const BATCH_SIZE: usize = 4096;

/// Adversarial edge analysis: find graph edges that are never useful across a query set.
///
/// An edge (src→dst) is "adversarial" if its distance was computed at least once during
/// beam search but the dst node was never actually inserted into any candidate set —
/// meaning it never contributed to any result. Such edges slow down search without
/// providing benefit.
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

    /// Beam widths for search (comma-separated, e.g. "32,64,128")
    #[arg(long, value_delimiter = ',')]
    beam_width: Vec<usize>,

    /// Seeds for LSH initialisation (comma-separated, e.g. "42,123,456")
    #[arg(long, value_delimiter = ',', default_value = "42")]
    seeds: Vec<u64>,

    /// Running modes (comma-separated, e.g. "vanilla,catapult")
    #[arg(short, long, value_delimiter = ',', value_parser = clap::builder::PossibleValuesParser::new(["vanilla", "catapult", "lshapg"]))]
    modes: Vec<String>,

    /// Number of threads to use for parallel search
    #[arg(short, long)]
    threads: usize,

    /// Path to output JSON file
    #[arg(short, long)]
    output: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AdversarialEdge {
    src: usize,
    dst: usize,
    times_considered: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct RunResult {
    mode: String,
    seed: u64,
    beam_width: usize,
    num_queries: usize,
    total_graph_edges: usize,
    used_edges: usize,
    adversarial_edges: usize,
    adversarial_fraction: f64,
    /// Distribution of consideration counts for adversarial (never-inserted) edges
    adversarial_consideration_counts: ConsiderationCountStats,
    /// Distribution of consideration counts for used (at least once inserted) edges
    used_consideration_counts: ConsiderationCountStats,
    top_adversarial_edges: Vec<AdversarialEdge>,
    top_used_edges: Vec<AdversarialEdge>,
}

#[derive(Serialize, Deserialize, Debug)]
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

/// Builds the top-1000 edge list and consideration count stats from a raw (src, dst, count) set.
/// `edges` need not be pre-sorted; this function sorts them internally.
fn edge_report(mut edges: Vec<(usize, usize, u32)>) -> (Vec<AdversarialEdge>, ConsiderationCountStats) {
    edges.sort_unstable_by(|a, b| b.2.cmp(&a.2));

    let top: Vec<AdversarialEdge> = edges
        .iter()
        .take(1000)
        .map(|&(src, dst, times_considered)| AdversarialEdge { src, dst, times_considered })
        .collect();

    let mut counts: Vec<u32> = edges.iter().map(|&(_, _, c)| c).collect();
    counts.sort_unstable();
    let stats = counts_stats(&counts);

    (top, stats)
}

fn counts_stats(sorted: &[u32]) -> ConsiderationCountStats {
    if sorted.is_empty() {
        return ConsiderationCountStats {
            min: 0, p25: 0, median: 0, p75: 0, p90: 0, p99: 0, max: 0, mean: 0.0,
            considered_once: 0,
        };
    }
    let sum: u64 = sorted.iter().map(|&c| c as u64).sum();
    let considered_once = sorted.iter().filter(|&&c| c == 1).count();
    ConsiderationCountStats {
        min: sorted[0],
        p25: percentile(sorted, 25.0),
        median: percentile(sorted, 50.0),
        p75: percentile(sorted, 75.0),
        p90: percentile(sorted, 90.0),
        p99: percentile(sorted, 99.0),
        max: *sorted.last().unwrap(),
        mean: sum as f64 / sorted.len() as f64,
        considered_once,
    }
}

fn merge_tracking(trackings: Vec<AdvEdgeTracking>) -> AdvEdgeTracking {
    let mut merged = AdvEdgeTracking::new();
    for t in trackings {
        for (key, count) in t.edge_consider_counts {
            *merged.edge_consider_counts.entry(key).or_insert(0) += count;
        }
        merged.used_edges.extend(t.used_edges);
    }
    merged
}

fn run_analysis(
    graph: &Arc<AdjacencyGraph<LruSet, FlatSearch>>,
    queries: &Arc<Vec<Vec<AlignedBlock>>>,
    beam_width: usize,
    num_threads: usize,
    mode: &str,
    seed: u64,
) -> RunResult {
    let total_graph_edges = graph.total_edge_count();
    let num_queries = queries.len();
    let next_batch = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "  Running {} queries with mode={} seed={} beam_width={} threads={}...",
        num_queries, mode, seed, beam_width, num_threads
    );

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let graph = Arc::clone(graph);
            let queries = Arc::clone(queries);
            let next_batch = Arc::clone(&next_batch);

            thread::spawn(move || {
                let mut local_stats = Stats::new();
                local_stats.enable_adv_tracking();

                loop {
                    let batch_start = next_batch.fetch_add(BATCH_SIZE, Ordering::Relaxed);
                    if batch_start >= num_queries {
                        break;
                    }
                    let batch_end = (batch_start + BATCH_SIZE).min(num_queries);
                    for query in &queries[batch_start..batch_end] {
                        graph.beam_search(query, beam_width, beam_width, &mut local_stats);
                    }
                }

                local_stats
            })
        })
        .collect();

    let mut per_thread_tracking: Vec<AdvEdgeTracking> = Vec::with_capacity(num_threads);
    for handle in handles {
        let mut local_stats = handle.join().expect("Thread panicked");
        per_thread_tracking.push(local_stats.take_adv_tracking().expect("tracking missing"));
    }

    let tracking = merge_tracking(per_thread_tracking);
    let used_edge_count = tracking.used_edges.len();

    let adversarial_edges_raw: Vec<(usize, usize, u32)> = tracking
        .edge_consider_counts
        .iter()
        .filter(|((src, dst), _)| !tracking.used_edges.contains(&(*src, *dst)))
        .map(|(&(src, dst), &count)| (src, dst, count))
        .collect();

    let adversarial_edge_count = adversarial_edges_raw.len();
    let (top_adversarial_edges, adversarial_consideration_counts) =
        edge_report(adversarial_edges_raw);

    let used_edges_raw: Vec<(usize, usize, u32)> = tracking
        .edge_consider_counts
        .iter()
        .filter(|((src, dst), _)| tracking.used_edges.contains(&(*src, *dst)))
        .map(|(&(src, dst), &count)| (src, dst, count))
        .collect();
    let (top_used_edges, used_consideration_counts) = edge_report(used_edges_raw);

    let adversarial_fraction = if total_graph_edges > 0 {
        adversarial_edge_count as f64 / total_graph_edges as f64
    } else {
        0.0
    };

    eprintln!(
        "  Done: {}/{} adversarial edges ({:.2}%)",
        adversarial_edge_count,
        total_graph_edges,
        adversarial_fraction * 100.0
    );

    RunResult {
        mode: mode.to_string(),
        seed,
        beam_width,
        num_queries,
        total_graph_edges,
        used_edges: used_edge_count,
        adversarial_edges: adversarial_edge_count,
        adversarial_fraction,
        adversarial_consideration_counts,
        used_consideration_counts,
        top_adversarial_edges,
        top_used_edges,
    }
}

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
    }

    let args = Args::parse();

    eprintln!("Loading queries...");
    let queries = Arc::new(Vec::<Vec<AlignedBlock>>::load_from_npy(
        &args.queries,
        LIMITATION,
    ));
    eprintln!("Loaded {} queries.", queries.len());

    let total_jobs = args.modes.len() * args.seeds.len() * args.beam_width.len();
    eprintln!(
        "Running {} job(s): {} mode(s) x {} seed(s) x {} beam_width(s), {} thread(s) per job",
        total_jobs,
        args.modes.len(),
        args.seeds.len(),
        args.beam_width.len(),
        args.threads,
    );

    let mut results: Vec<RunResult> = Vec::with_capacity(total_jobs);

    for mode in &args.modes {
        for &seed in &args.seeds {
            eprintln!("\n--- mode={} seed={} ---", mode, seed);
            eprintln!("  Loading graph...");
            let graph = Arc::new(AdjacencyGraph::<LruSet, FlatSearch>::load_flat_from_path(
                PathBuf::from_str(&args.graph).unwrap(),
                PathBuf::from_str(&args.payload).unwrap(),
                NUM_HASH,
                BUCKET_SIZE,
                seed,
                RunningMode::from_string(mode),
            ));
            eprintln!(
                "  Graph: {} nodes, {} edges",
                graph.len(),
                graph.total_edge_count()
            );

            for &beam_width in &args.beam_width {
                graph.clear_all_catapults();
                results.push(run_analysis(
                    &graph,
                    &queries,
                    beam_width,
                    args.threads,
                    mode,
                    seed,
                ));
            }
        }
    }

    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write(&args.output, json).expect("Failed to write output file");
    eprintln!("\nResults written to: {}", args.output);
}
