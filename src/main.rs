use catapult::{
    fs::Queries,
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{AdjacencyGraph, graph_algo::FlatSearch, hash_start::EngineStarterParams},
    sets::catapults::FifoSet,
    statistics::Stats,
};
use clap::Parser;
use std::{
    hint::black_box,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, atomic::AtomicUsize},
    thread,
};

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
}

fn run_search_job(
    graph: Arc<AdjacencyGraph<FifoSet<30>, FlatSearch>>,
    queries: Arc<Vec<Vec<AlignedBlock>>>,
    num_threads: usize,
    beam_width: usize,
    num_neighbors: usize,
    catapults_enabled: bool,
) {
    let num_queries = queries.len();
    println!("\n==========");
    println!(
        "Running with threads={}, beam_width={}",
        num_threads, beam_width
    );
    println!("==========");

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

    if catapults_enabled {
        let catapult_usage_pct = if num_queries > 0 {
            (combined_stats.get_searches_with_catapults() as f64 / num_queries as f64) * 100.0
        } else {
            0.0
        };
        let avg_dists_computed = combined_stats.get_computed_dists() as f64 / num_queries as f64;
        let avg_nodes_visited = combined_stats.get_nodes_visited() as f64 / num_queries as f64;
        let avg_catapults_added =
            combined_stats.get_searches_with_catapults() as f64 / num_queries as f64;
        println!(
            "Catapult stats: {}/{} searches used catapults ({:.2}%)",
            combined_stats.get_searches_with_catapults(),
            num_queries,
            catapult_usage_pct,
        );
        println!(
            "  Avg per search: {:.2} dists computed, {:.2} nodes visited, {:.2} catapult used",
            avg_dists_computed, avg_nodes_visited, avg_catapults_added
        );
    } else {
        let avg_dists_computed = combined_stats.get_computed_dists() as f64 / num_queries as f64;
        let avg_visited_nodes = combined_stats.get_nodes_visited() as f64 / num_queries as f64;
        println!(
            "Avg per search: {:.2} dists computed, {:.2} visited nodes",
            avg_dists_computed, avg_visited_nodes
        );
    }

    println!(
        "Checksum: {:?}",
        reses.into_iter().map(|e| e.index).reduce(|a, b| a + b)
    );

    let elapsed = start_time.elapsed();
    let total_qps = num_queries as f64 / elapsed.as_secs_f64();
    println!(
        "Completed {} searches in {:.2}s ({:.2} QPS)",
        num_queries,
        elapsed.as_secs_f64(),
        total_qps
    );
}

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
        // parsing DiskANN files requires little endian.
    }
    let args = Args::parse();

    // Load the adjacency graph from file
    println!("Loading adjacency graph...");
    let queries: Vec<Vec<AlignedBlock>> = Vec::<Vec<AlignedBlock>>::load_from_npy(&args.queries)
        .into_iter()
        .take(300_000)
        .collect();

    let full_graph = Arc::new(
        AdjacencyGraph::<FifoSet<30>, FlatSearch>::load_flat_from_path(
            PathBuf::from_str(&args.graph).unwrap(),
            PathBuf::from_str(&args.payload).unwrap(),
            EngineStarterParams {
                num_hash: 16,
                plane_dim: queries[0].len() * SIMD_LANECOUNT,
                starting_node: 0,
                seed: 42,
                enabled_catapults: args.catapults,
            },
        ),
    );
    let graph_size = full_graph.len();

    println!("Adjacency graph loaded with {graph_size} nodes");

    let queries = Arc::new(queries);

    println!("\nStarting cartesian product sweep:");
    println!("  Threads: {:?}", args.threads);
    println!("  Beam widths: {:?}", args.beam_width);
    println!(
        "  Total jobs: {}",
        args.threads.len() * args.beam_width.len()
    );

    // Run cartesian product of threads and beam_width
    for &num_threads in &args.threads {
        for &beam_width in &args.beam_width {
            // Clear all catapults before each job to ensure a clean slate
            full_graph.clear_all_catapults();

            run_search_job(
                Arc::clone(&full_graph),
                Arc::clone(&queries),
                num_threads,
                beam_width,
                args.num_neighbors,
                args.catapults,
            );
        }
    }

    println!("\n==========");
    println!("All jobs completed!");
    println!("==========");
}
