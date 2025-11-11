use catapult::indexing::adjacency_graph::AdjacencyGraph;
use catapult::indexing::engine_starter::EngineStarter;
use catapult::indexing::eviction::FifoSet;
use std::path::PathBuf;
use std::str::FromStr;

const LOAD_LI_ENDIAN: bool = cfg!(target_endian = "little");

fn main() {
    assert!(LOAD_LI_ENDIAN); // check that the tests are being run on a little endian machine

    let adjacency = AdjacencyGraph::<FifoSet<30>>::load_from_path::<LOAD_LI_ENDIAN>(
        PathBuf::from_str("test_index/4vecs/ann").unwrap(),
        PathBuf::from_str("test_index/4vecs/ann.data").unwrap(),
    );

    for (index, node) in adjacency.iter().enumerate() {
        println!("{:?}: {:?}", index, node);
    }

    println!("================================");

    let adj_len = adjacency.len();
    let full_graph = AdjacencyGraph::new(adjacency, EngineStarter::new(10, 8, adj_len, Some(42)));

    println!("{:?}", full_graph.beam_search(&vec![2.5; 8], 2, 2));
}
