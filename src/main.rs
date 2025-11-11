use catapult::indexing::adjacency_graph::AdjacencyGraph;
use catapult::indexing::eviction::FifoSet;
use std::path::PathBuf;
use std::str::FromStr;

fn main() {
    let adjacency = AdjacencyGraph::<FifoSet<20>>::load_from_path(
        PathBuf::from_str("test_index/4vecs/ann").unwrap(),
    );

    for (index, node) in adjacency.iter().enumerate() {
        println!("{:?} {:?}", index, node);
    }
}
