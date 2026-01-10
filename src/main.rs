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

    /// Number of threads to use for parallel search
    #[arg(short, long, default_value_t = 1)]
    threads: usize,

    /// Number of neighbors to return for each query
    #[arg(long)]
    num_neighbors: usize,

    /// Width for beam search
    #[arg(long)]
    beam_width: usize,
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
    let engine_seed = 42;
    let engine = EngineStarter::<FifoSet<30>>::new(
        num_hash,
        plane_dim,
        graph_size,
        engine_seed,
        args.catapults,
    );

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

    let num_threads = args.threads;
    let num_queries = queries.len();
    println!("Starting search of {num_queries} vectors using {num_threads} thread(s)...");
    let start_time = std::time::Instant::now();

    let queries = Arc::new(queries);
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
                            args.beam_width,
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

    if args.catapults {
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
        "{:?}",
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
