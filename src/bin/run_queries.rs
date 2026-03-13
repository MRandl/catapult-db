use catapult::{
    fs::Queries,
    numerics::AlignedBlock,
    search::{AdjacencyGraph, RunningMode, graph_algo::FlatSearch},
    sets::catapults::LruSet,
    statistics::Stats,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    hint::black_box,
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};
use tracing::info_span;
use tracing_subscriber::prelude::*;

const NUM_HASH: usize = 8;
const BUCKET_SIZE: usize = 40;
const BATCH_SIZE: usize = 4096;
const LIMITATION: Option<usize> = Some(1_000_000);

#[derive(Serialize, Deserialize, Debug)]
struct SearchJobResult {
    seed: u64,
    num_threads: usize,
    beam_width: usize,
    num_queries: usize,
    catapults_enabled: bool,
    bucket_capacity: usize,
    num_hashes: usize,
    elapsed_secs: f64,
    qps: f64,
    checksum: Option<usize>,
    avg_dists_computed: f64,
    avg_nodes_visited: f64,
    searches_with_catapults: Option<usize>,
    catapult_usage_pct: Option<f64>,
    avg_catapults_added: Option<f64>,
    /// Per-query neighbors in query order, present only when --output-neighbors is set
    #[serde(skip_serializing_if = "Option::is_none")]
    neighbors: Option<Vec<Vec<usize>>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkResults {
    results: Vec<SearchJobResult>,
}

/// Vector search engine using adjacency graphs
#[derive(Parser, Debug)]
#[command(name = "catapult")]
#[command(about = "A vector search engine using adjacency graphs", long_about = None)]
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

    /// Whether catapults should be used or not
    #[arg(short, long, value_parser = clap::builder::PossibleValuesParser::new(["vanilla", "catapult", "lshapg"]))]
    mode: String,

    /// Number of threads to use for parallel search (comma-separated list, e.g., "1,2,4,8")
    #[arg(short, long, value_delimiter = ',')]
    threads: Vec<usize>,

    /// Width for beam search (comma-separated list, e.g., "10,20,40")
    #[arg(long, value_delimiter = ',')]
    beam_width: Vec<usize>,

    /// Seeds for random number generation (comma-separated list, e.g., "42,123,456")
    #[arg(long, value_delimiter = ',', default_value = "42")]
    seeds: Vec<u64>,

    /// Path to output JSON file
    #[arg(short, long)]
    output: String,

    /// Include per-query neighbor results (index + distance) in the output JSON
    #[arg(long, default_value_t = false)]
    output_neighbors: bool,
}

/// Runs beam search over all queries using a thread pool with work-stealing batches.
/// Returns results sorted by query index and aggregate stats.
fn parallel_beam_search(
    graph: &Arc<AdjacencyGraph<LruSet, FlatSearch>>,
    queries: &Arc<Vec<Vec<AlignedBlock>>>,
    num_threads: usize,
    beam_width: usize,
) -> (Vec<(usize, Vec<usize>)>, Stats) {
    let num_queries = queries.len();
    let next_batch = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let graph = Arc::clone(graph);
            let queries = Arc::clone(queries);
            let next_batch = Arc::clone(&next_batch);

            thread::spawn(move || {
                let mut local_results: Vec<(usize, Vec<usize>)> = Vec::new();
                let mut local_stats = Stats::new();

                loop {
                    let batch_start = next_batch.fetch_add(BATCH_SIZE, Ordering::Relaxed);
                    if batch_start >= num_queries {
                        break;
                    }
                    let batch_end = (batch_start + BATCH_SIZE).min(num_queries);

                    for (offset, query) in queries[batch_start..batch_end].iter().enumerate() {
                        let result = black_box(graph.beam_search(
                            query,
                            beam_width,
                            beam_width,
                            &mut local_stats,
                        ));
                        local_results.push((
                            batch_start + offset,
                            result.iter().map(|e| e.index.internal).collect(),
                        ));
                    }
                }

                (local_results, local_stats)
            })
        })
        .collect();

    let mut results: Vec<(usize, Vec<usize>)> = Vec::with_capacity(num_queries);
    let mut combined_stats = Stats::new();
    for handle in handles {
        let (local_results, local_stats) = handle.join().expect("Thread panicked");
        results.extend(local_results);
        combined_stats = combined_stats.merge(&local_stats);
    }

    // Restore query order (threads may complete batches out of order)
    results.sort_unstable_by_key(|(idx, _)| *idx);

    (results, combined_stats)
}

fn run_search_job(
    graph: Arc<AdjacencyGraph<LruSet, FlatSearch>>,
    queries: Arc<Vec<Vec<AlignedBlock>>>,
    num_threads: usize,
    beam_width: usize,
    catapults_enabled: bool,
    seed: u64,
    bucket_capacity: usize,
    num_hashes: usize,
    output_neighbors: bool,
) -> SearchJobResult {
    let num_queries = queries.len();
    eprintln!("\n==========");
    eprintln!(
        "Running with seed={}, threads={}, beam_width={}",
        seed, num_threads, beam_width
    );
    eprintln!("==========");

    let start_time = std::time::Instant::now();
    let (results, combined_stats) = parallel_beam_search(&graph, &queries, num_threads, beam_width);
    let elapsed = start_time.elapsed();

    let total_qps = num_queries as f64 / elapsed.as_secs_f64();
    let avg_dists_computed = combined_stats.get_computed_dists() as f64 / num_queries as f64;
    let avg_nodes_visited = combined_stats.get_nodes_visited() as f64 / num_queries as f64;
    let checksum = results.iter().map(|(_, res)| res[0]).reduce(|a, b| a + b);

    let (searches_with_catapults, catapult_usage_pct, avg_catapults_added) = if catapults_enabled {
        let n = combined_stats.get_searches_with_catapults();
        let usage_pct = if num_queries > 0 {
            (n as f64 / num_queries as f64) * 100.0
        } else {
            0.0
        };
        let avg = n as f64 / num_queries as f64;

        eprintln!(
            "Catapult stats: {}/{} searches used catapults ({:.2}%)",
            n, num_queries, usage_pct,
        );
        eprintln!(
            "  Avg per search: {:.2} dists computed, {:.2} nodes visited, {:.2} catapult used",
            avg_dists_computed, avg_nodes_visited, avg
        );

        (Some(n), Some(usage_pct), Some(avg))
    } else {
        eprintln!(
            "Avg per search: {:.2} dists computed, {:.2} visited nodes",
            avg_dists_computed, avg_nodes_visited
        );

        (None, None, None)
    };

    eprintln!("Checksum: {:?}", checksum);
    eprintln!(
        "Completed {} searches in {:.2}s ({:.2} QPS)",
        num_queries,
        elapsed.as_secs_f64(),
        total_qps
    );

    let neighbors = if output_neighbors {
        Some(results.into_iter().map(|(_, res)| res).collect())
    } else {
        None
    };

    SearchJobResult {
        seed,
        num_threads,
        beam_width,
        num_queries,
        catapults_enabled,
        bucket_capacity,
        num_hashes,
        elapsed_secs: elapsed.as_secs_f64(),
        qps: total_qps,
        checksum,
        avg_dists_computed,
        avg_nodes_visited,
        searches_with_catapults,
        catapult_usage_pct,
        avg_catapults_added,
        neighbors,
    }
}

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
        // parsing DiskANN files requires little endian.
    }

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    let (flame_layer, _flame_guard) =
        tracing_flame::FlameLayer::with_file("./tracing.folded").unwrap();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(flame_layer)
        .init();

    let args = Args::parse();

    // Load the queries
    eprintln!("Loading queries...");
    let queries: Vec<Vec<AlignedBlock>> = {
        let _span = info_span!("load_queries", path = %args.queries).entered();
        Vec::<Vec<AlignedBlock>>::load_from_npy(&args.queries, LIMITATION)
    };
    let queries = Arc::new(queries);

    eprintln!("\nStarting cartesian product sweep:");
    eprintln!("  Seeds: {:?}", args.seeds);
    eprintln!("  Threads: {:?}", args.threads);
    eprintln!("  Beam widths: {:?}", args.beam_width);
    eprintln!(
        "  Total jobs: {}",
        args.seeds.len() * args.threads.len() * args.beam_width.len()
    );

    let catapults_enabled = RunningMode::from_string(&args.mode) == RunningMode::Catapult;
    let mut all_results = Vec::new();

    // Run cartesian product of seeds, threads, and beam_width
    for &seed in &args.seeds {
        eprintln!(
            "\n--- Loading adjacency graph with seed={}, num_hash={}, bucket_cap={} ---",
            seed, NUM_HASH, BUCKET_SIZE
        );
        let full_graph = {
            let _span = info_span!("load_graph", seed, num_hash = NUM_HASH, bucket_cap = BUCKET_SIZE, mode = %args.mode).entered();
            Arc::new(AdjacencyGraph::<LruSet, FlatSearch>::load_flat_from_path(
                PathBuf::from_str(&args.graph).unwrap(),
                PathBuf::from_str(&args.payload).unwrap(),
                NUM_HASH,
                BUCKET_SIZE,
                seed,
                RunningMode::from_string(&args.mode),
            ))
        };
        let graph_size = full_graph.len();
        eprintln!("Adjacency graph loaded with {graph_size} nodes");

        for &num_threads in &args.threads {
            for &beam_width in &args.beam_width {
                // Clear all catapults before each job to ensure a clean slate
                full_graph.clear_all_catapults();

                let result = {
                    let _span = info_span!("search_job", seed, num_threads, beam_width).entered();
                    run_search_job(
                        Arc::clone(&full_graph),
                        Arc::clone(&queries),
                        num_threads,
                        beam_width,
                        catapults_enabled,
                        seed,
                        BUCKET_SIZE,
                        NUM_HASH,
                        args.output_neighbors,
                    )
                };
                all_results.push(result);
            }
        }
    }

    eprintln!("\n==========");
    eprintln!("All jobs completed!");
    eprintln!("==========");

    // Output results as JSON to file
    let benchmark_results = BenchmarkResults {
        results: all_results,
    };
    let json_output = serde_json::to_string_pretty(&benchmark_results).unwrap();
    std::fs::write(&args.output, json_output).expect("Failed to write output file");
    eprintln!("Results written to: {}", args.output);
}
