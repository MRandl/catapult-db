use catapult::{
    fs::Queries,
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{AdjacencyGraph, FlatCatapultChoice, FlatSearch, hash_start::EngineStarter},
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

fn main() {
    const {
        assert!(cfg!(target_endian = "little"));
        // parsing DiskANN files requires little endian.
    }
    let args = Args::parse();

    // Load the adjacency graph from file
    println!("Loading adjacency graph...");
    let adjacency = AdjacencyGraph::<FifoSet<30>, FlatSearch>::load_flat_from_path(
        PathBuf::from_str(&args.graph).unwrap(),
        PathBuf::from_str(&args.payload).unwrap(),
    );

    let queries: Vec<Vec<AlignedBlock>> = Vec::<Vec<AlignedBlock>>::load_from_npy(&args.queries)
        .into_iter()
        .take(300_000)
        .collect();

    let graph_size = adjacency.len();
    let num_hash = 16;
    let plane_dim = queries[0].len() * SIMD_LANECOUNT;
    let engine_seed = Some(42);
    let engine = EngineStarter::new(num_hash, plane_dim, graph_size, engine_seed);

    let full_graph = Arc::new(AdjacencyGraph::new_flat(
        adjacency,
        engine,
        if args.catapults {
            FlatCatapultChoice::CatapultsEnabled
        } else {
            FlatCatapultChoice::CatapultsDisabled
        },
    ));
    println!("Adjacency graph loaded with {graph_size} nodes");

    let num_queries = queries.len();
    let queries = Arc::new(queries);

    // Run cartesian product of all configurations
    println!(
        "\nRunning {} configurations (threads: {:?}, beam_width: {:?})",
        args.threads.len() * args.beam_width.len(),
        args.threads,
        args.beam_width
    );
    println!("{}", "=".repeat(80));

    for &num_threads in &args.threads {
        for &beam_width in &args.beam_width {
            println!(
                "\n--- Configuration: threads={}, beam_width={} ---",
                num_threads, beam_width
            );

            // Reset catapults before each experiment to ensure independence
            if args.catapults {
                full_graph.reset_catapults();
            }

            let start_time = std::time::Instant::now();
            let batch_size = 4096;
            let next_batch = Arc::new(AtomicUsize::new(0));

            let handles: Vec<_> = (0..num_threads)
                .map(|_thread_id| {
                    let graph = Arc::clone(&full_graph);
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
                                    args.num_neighbors,
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

                // Merge stats from this thread
                for _ in 0..local_stats.get_searches_with_catapults() {
                    combined_stats.bump_searches_with_catapults();
                }
                combined_stats.bump_catapults_used(local_stats.get_catapults_used());
                combined_stats
                    .bump_regular_neighbors_added(local_stats.get_regular_neighbors_added());
                for _ in 0..local_stats.get_nodes_expanded() {
                    combined_stats.bump_nodes_expanded();
                }
            }

            if args.catapults {
                let catapult_usage_pct = if num_queries > 0 {
                    (combined_stats.get_searches_with_catapults() as f64 / num_queries as f64)
                        * 100.0
                } else {
                    0.0
                };
                let avg_nodes_expanded =
                    combined_stats.get_nodes_expanded() as f64 / num_queries as f64;
                let avg_regular_added =
                    combined_stats.get_regular_neighbors_added() as f64 / num_queries as f64;
                let avg_catapults_added =
                    combined_stats.get_catapults_used() as f64 / num_queries as f64;
                println!(
                    "Catapult stats: {}/{} searches used catapults ({:.2}%), {} total catapult edges used",
                    combined_stats.get_searches_with_catapults(),
                    num_queries,
                    catapult_usage_pct,
                    combined_stats.get_catapults_used()
                );
                println!(
                    "  Avg per search: {:.2} nodes expanded, {:.2} regular neighbors added, {:.2} catapult neighbors added",
                    avg_nodes_expanded, avg_regular_added, avg_catapults_added
                );
            } else {
                let avg_nodes_expanded =
                    combined_stats.get_nodes_expanded() as f64 / num_queries as f64;
                let avg_regular_added =
                    combined_stats.get_regular_neighbors_added() as f64 / num_queries as f64;
                println!(
                    "Avg per search: {:.2} nodes expanded, {:.2} regular neighbors added",
                    avg_nodes_expanded, avg_regular_added
                );
            }

            println!(
                "Sanity check sum: {:?}",
                reses.into_iter().reduce(|a, b| a + b)
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
    }

    println!("\n{}", "=".repeat(80));
    println!("All configurations completed!");
}
