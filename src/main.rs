use catapult::{
    fs::Queries,
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{AdjacencyGraph, graph_algo::FlatSearch, hash_start::EngineStarterParams},
    sets::catapults::FifoSet,
    statistics::Stats,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    hint::black_box,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, atomic::AtomicUsize},
    thread,
};

/// JSON output structures
#[derive(Serialize, Deserialize, Debug)]
struct SearchJobResult {
    seed: u64,
    num_threads: usize,
    beam_width: usize,
    num_queries: usize,
    catapults_enabled: bool,
    elapsed_secs: f64,
    qps: f64,
    checksum: Option<usize>,
    avg_dists_computed: f64,
    avg_nodes_visited: f64,
    searches_with_catapults: Option<usize>,
    catapult_usage_pct: Option<f64>,
    avg_catapults_added: Option<f64>,
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

    /// Whethere catapults should be used or not
    #[arg(short, long)]
    catapults: bool,

    /// Number of threads to use for parallel search (comma-separated list, e.g., "1,2,4,8")
    #[arg(short, long, value_delimiter = ',')]
    threads: Vec<usize>,

    /// Number of neighbors to return for each query
    #[arg(long)]
    num_neighbors: usize,

    /// Width for beam search (comma-separated list, e.g., "10,20,40")
    #[arg(long, value_delimiter = ',')]
    beam_width: Vec<usize>,

    /// Seeds for random number generation (comma-separated list, e.g., "42,123,456")
    #[arg(long, value_delimiter = ',', default_value = "42")]
    seeds: Vec<u64>,

    /// Path to output JSON file
    #[arg(short, long)]
    output: String,
}

fn run_search_job(
    graph: Arc<AdjacencyGraph<FifoSet<30>, FlatSearch>>,
    queries: Arc<Vec<Vec<AlignedBlock>>>,
    num_threads: usize,
    beam_width: usize,
    num_neighbors: usize,
    catapults_enabled: bool,
    seed: u64,
) -> SearchJobResult {
    let num_queries = queries.len();
    eprintln!("\n==========");
    eprintln!(
        "Running with seed={}, threads={}, beam_width={}",
        seed, num_threads, beam_width
    );
    eprintln!("==========");

    let start_time = std::time::Instant::now();

    let batch_size = 4096;
    let next_batch = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..num_threads)
        .map(|_thread_id| {
            let graph = Arc::clone(&graph);
            let queries_clone = Arc::clone(&queries);
            let next_batch_clone = Arc::clone(&next_batch);

            thread::spawn(move || {
                let mut local_results = Vec::new();
                let mut local_stats = Stats::new();

                loop {
                    // Atomically grab the next batch of work
                    let batch_start = next_batch_clone
                        .fetch_add(batch_size, std::sync::atomic::Ordering::Relaxed);

                    if batch_start >= num_queries {
                        break;
                    }

                    let batch_end = std::cmp::min(batch_start + batch_size, num_queries);

                    // Process this batch
                    for query in &queries_clone[batch_start..batch_end] {
                        let _result = black_box(graph.beam_search(
                            query,
                            num_neighbors,
                            beam_width,
                            &mut local_stats,
                        ));
                        local_results.push(_result[0]);
                    }
                }

                (local_results, local_stats)
            })
        })
        .collect();

    let mut reses = Vec::with_capacity(num_queries);
    let mut combined_stats = Stats::new();
    for handle in handles {
        let (local_results, local_stats) = handle.join().expect("Thread panicked");
        reses.extend(local_results);
        combined_stats = combined_stats.merge(&local_stats)
    }

    let elapsed = start_time.elapsed();
    let total_qps = num_queries as f64 / elapsed.as_secs_f64();

    let avg_dists_computed = combined_stats.get_computed_dists() as f64 / num_queries as f64;
    let avg_nodes_visited = combined_stats.get_nodes_visited() as f64 / num_queries as f64;
    let checksum = reses
        .into_iter()
        .map(|e| e.index.internal)
        .reduce(|a, b| a + b);

    let result = if catapults_enabled {
        let catapult_usage_pct = if num_queries > 0 {
            (combined_stats.get_searches_with_catapults() as f64 / num_queries as f64) * 100.0
        } else {
            0.0
        };
        let avg_catapults_added =
            combined_stats.get_searches_with_catapults() as f64 / num_queries as f64;

        eprintln!(
            "Catapult stats: {}/{} searches used catapults ({:.2}%)",
            combined_stats.get_searches_with_catapults(),
            num_queries,
            catapult_usage_pct,
        );
        eprintln!(
            "  Avg per search: {:.2} dists computed, {:.2} nodes visited, {:.2} catapult used",
            avg_dists_computed, avg_nodes_visited, avg_catapults_added
        );

        SearchJobResult {
            seed,
            num_threads,
            beam_width,
            num_queries,
            catapults_enabled,
            elapsed_secs: elapsed.as_secs_f64(),
            qps: total_qps,
            checksum,
            avg_dists_computed,
            avg_nodes_visited,
            searches_with_catapults: Some(combined_stats.get_searches_with_catapults()),
            catapult_usage_pct: Some(catapult_usage_pct),
            avg_catapults_added: Some(avg_catapults_added),
        }
    } else {
        eprintln!(
            "Avg per search: {:.2} dists computed, {:.2} visited nodes",
            avg_dists_computed, avg_nodes_visited
        );

        SearchJobResult {
            seed,
            num_threads,
            beam_width,
            num_queries,
            catapults_enabled,
            elapsed_secs: elapsed.as_secs_f64(),
            qps: total_qps,
            checksum,
            avg_dists_computed,
            avg_nodes_visited,
            searches_with_catapults: None,
            catapult_usage_pct: None,
            avg_catapults_added: None,
        }
    };

    eprintln!("Checksum: {:?}", checksum);

    eprintln!(
        "Completed {} searches in {:.2}s ({:.2} QPS)",
        num_queries,
        elapsed.as_secs_f64(),
        total_qps
    );

    result
}

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
        // parsing DiskANN files requires little endian.
    }
    let args = Args::parse();

    // Load the queries
    eprintln!("Loading queries...");
    let queries: Vec<Vec<AlignedBlock>> = Vec::<Vec<AlignedBlock>>::load_from_npy(&args.queries);
    // .into_iter()
    // .take(150_000)
    // .collect();
    let queries = Arc::new(queries);

    eprintln!("\nStarting cartesian product sweep:");
    eprintln!("  Seeds: {:?}", args.seeds);
    eprintln!("  Threads: {:?}", args.threads);
    eprintln!("  Beam widths: {:?}", args.beam_width);
    eprintln!(
        "  Total jobs: {}",
        args.seeds.len() * args.threads.len() * args.beam_width.len()
    );

    let mut all_results = Vec::new();

    // Run cartesian product of seeds, threads, and beam_width
    for &seed in &args.seeds {
        eprintln!("\n--- Loading adjacency graph with seed={} ---", seed);
        let full_graph = Arc::new(
            AdjacencyGraph::<FifoSet<30>, FlatSearch>::load_flat_from_path(
                PathBuf::from_str(&args.graph).unwrap(),
                PathBuf::from_str(&args.payload).unwrap(),
                EngineStarterParams {
                    num_hash: 16,
                    plane_dim: queries[0].len() * SIMD_LANECOUNT,
                    starting_node: catapult::search::NodeId { internal: 0 },
                    seed,
                    enabled_catapults: args.catapults,
                },
            ),
        );
        let graph_size = full_graph.len();
        eprintln!("Adjacency graph loaded with {graph_size} nodes");

        for &num_threads in &args.threads {
            for &beam_width in &args.beam_width {
                // Clear all catapults before each job to ensure a clean slate
                full_graph.clear_all_catapults();

                let result = run_search_job(
                    Arc::clone(&full_graph),
                    Arc::clone(&queries),
                    num_threads,
                    beam_width,
                    args.num_neighbors,
                    args.catapults,
                    seed,
                );
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
