use catapult::indexing::adjacency_graph::AdjacencyGraph;
use catapult::indexing::engine_starter::EngineStarter;
use catapult::indexing::eviction::FifoSet;
use clap::Parser;
use std::hint::black_box;
use std::path::PathBuf;
use std::str::FromStr;

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
}

fn main() {
    let args = Args::parse();

    // Load the adjacency graph from file
    println!("Loading adjacency graph...");
    let adjacency = AdjacencyGraph::<FifoSet<30>>::load_from_path::<LOAD_LI_ENDIAN>(
        PathBuf::from_str(&args.graph).unwrap(),
        PathBuf::from_str(&args.payload).unwrap(),
    );

    let queries = Vec::<Vec<f32>>::load_from_npy(&args.queries);

    let graph_size = adjacency.len();
    let num_hash = 10;
    let plane_dim = 768;
    let engine_seed = Some(42);
    let engine = EngineStarter::new(num_hash, plane_dim, graph_size, engine_seed);

    let full_graph = AdjacencyGraph::new(adjacency, engine, args.catapults);
    println!("Adjacency graph loaded with {} nodes", graph_size);

    // Search each random vector sequentially
    println!("Starting sequential search of {} vectors...", queries.len());
    let start_time = std::time::Instant::now();

    let mut reses = Vec::with_capacity(queries.len());

    for (i, query) in queries.iter().enumerate() {
        let _result = black_box(full_graph.beam_search(query, 2, 2));
        reses.push(_result[0].index);

        // Print progress every 100k queries
        if (i + 1) % 100_000 == 0 {
            let elapsed = start_time.elapsed();
            let qps = (i + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "Processed {}/{} queries ({:.2} QPS)",
                i + 1,
                queries.len(),
                qps
            );
        }
    }

    println!("{:?}", reses.into_iter().reduce(|a, b| a + b));

    let elapsed = start_time.elapsed();
    let total_qps = queries.len() as f64 / elapsed.as_secs_f64();
    println!(
        "Completed {} searches in {:.2}s ({:.2} QPS)",
        queries.len(),
        elapsed.as_secs_f64(),
        total_qps
    );
}
