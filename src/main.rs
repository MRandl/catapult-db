use catapult::indexing::adjacency_graph::AdjacencyGraph;
use catapult::indexing::engine_starter::EngineStarter;
use catapult::indexing::eviction::FifoSet;
use clap::Parser;
use std::hint::black_box;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;

use catapult::fs::Queries;

const LOAD_LI_ENDIAN: bool = cfg!(target_endian = "little");

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
}

fn main() {
    let args = Args::parse();

    // Load the adjacency graph from file
    println!("Loading adjacency graph...");
    let adjacency = AdjacencyGraph::<FifoSet<30>>::load_from_path::<LOAD_LI_ENDIAN>(
        PathBuf::from_str(&args.graph).unwrap(),
        PathBuf::from_str(&args.payload).unwrap(),
    );

    let queries: Vec<Vec<f32>> = Vec::<Vec<f32>>::load_from_npy(&args.queries)
        .into_iter()
        .take(500_000)
        .collect();

    let graph_size = adjacency.len();
    let num_hash = 10;
    let plane_dim = queries[0].len();
    let engine_seed = Some(42);
    let engine = EngineStarter::new(num_hash, plane_dim, graph_size, engine_seed);

    let full_graph = Arc::new(AdjacencyGraph::new(adjacency, engine, args.catapults));
    println!("Adjacency graph loaded with {graph_size} nodes");

    let num_threads = args.threads;
    let num_queries = queries.len();
    println!("Starting search of {num_queries} vectors using {num_threads} thread(s)...",);
    let start_time = std::time::Instant::now();

    // Multi-threaded execution
    let queries = Arc::new(queries);
    let chunk_size = num_queries.div_ceil(num_threads);

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let graph = Arc::clone(&full_graph);
            let queries_clone = Arc::clone(&queries);
            let start = thread_id * chunk_size;
            let end = std::cmp::min(start + chunk_size, num_queries);

            thread::spawn(move || {
                let mut local_results = Vec::with_capacity(end - start);
                for query in &queries_clone[start..end] {
                    let _result = black_box(graph.beam_search(query, 2, 2));
                    local_results.push(_result[0].index);
                }
                local_results
            })
        })
        .collect();

    // Collect results from all threads
    let mut reses = Vec::with_capacity(num_queries);
    for handle in handles {
        let local_results = handle.join().expect("Thread panicked");
        reses.extend(local_results);
    }
    println!("{:?}", reses.into_iter().reduce(|a, b| a + b));

    let elapsed = start_time.elapsed();
    let total_qps = num_queries as f64 / elapsed.as_secs_f64();
    println!(
        "Completed {} searches in {:.2}s ({:.2} QPS)",
        num_queries,
        elapsed.as_secs_f64(),
        total_qps
    );
}
