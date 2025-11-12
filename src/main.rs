use catapult::indexing::adjacency_graph::AdjacencyGraph;
use catapult::indexing::engine_starter::EngineStarter;
use catapult::indexing::eviction::FifoSet;
use rand::Rng;
use std::hint::black_box;
use std::path::PathBuf;
use std::str::FromStr;

const LOAD_LI_ENDIAN: bool = cfg!(target_endian = "little");

fn main() {
    // Generate 1M 8-dimensional random vectors
    println!("Generating 1M 8-dimensional random vectors...");
    let mut rng = rand::rng();
    let queries: Vec<Vec<f32>> = (0..1_000_000)
        .map(|_| (0..8).map(|_| rng.random::<f32>()).collect())
        .collect();
    println!("Generated {} query vectors", queries.len());

    // Load the adjacency graph from file
    println!("Loading adjacency graph...");
    let adjacency = AdjacencyGraph::<FifoSet<30>>::load_from_path::<LOAD_LI_ENDIAN>(
        PathBuf::from_str("test_index/4vecs/ann").unwrap(),
        PathBuf::from_str("test_index/4vecs/ann.data").unwrap(),
    );

    let graph_size = adjacency.len();
    let num_hash = 10;
    let plane_dim = 8;
    let engine_seed = Some(42);
    let engine = EngineStarter::new(num_hash, plane_dim, graph_size, engine_seed);

    let full_graph = AdjacencyGraph::new(adjacency, engine);
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
            println!("q {:?} - {:?}", query, _result)
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
