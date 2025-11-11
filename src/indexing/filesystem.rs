use crate::indexing::{
    eviction::{FixedSet, neighbor_set::EvictionNeighborSet},
    node::{self, Node},
};

use super::adjacency_graph::AdjacencyGraph;
use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::PathBuf,
    sync::RwLock,
};

impl<T: EvictionNeighborSet> AdjacencyGraph<T> {
    fn next_u32<I>(iter: &mut I) -> Result<u32, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        let mut bytes = [0u8; 4];
        for b in &mut bytes {
            match iter.next() {
                Some(Ok(v)) => *b = v,
                Some(Err(e)) => return Err(e.to_string()),
                None => return Err("Unexpected end of stream".into()),
            }
        }
        Ok(u32::from_le_bytes(bytes))
    }

    fn next_u64<I>(iter: &mut I) -> Result<u64, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        let mut bytes = [0u8; 8];
        for b in &mut bytes {
            match iter.next() {
                Some(Ok(v)) => *b = v,
                Some(Err(e)) => return Err(e.to_string()),
                None => return Err("Unexpected end of stream".into()),
            }
        }
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn load_from_path(path: PathBuf) -> Vec<Node<T>> {
        let mut binfile = BufReader::new(File::open(path).expect("FNF")).bytes();

        let full_size = Self::next_u64(&mut binfile).expect("Misconfigured header");
        let max_degree = Self::next_u32(&mut binfile).expect("Misconfigured header");
        let entry_point = Self::next_u32(&mut binfile).expect("Misconfigured header");
        let num_frozen = Self::next_u64(&mut binfile).expect("Misconfigured header");

        println!(
            "size {} - degree {} - entry point {} - num frozen {}",
            full_size, max_degree, entry_point, num_frozen
        );

        let mut adjacency = Vec::new();

        while let Ok(pointsize) = Self::next_u32(&mut binfile) {
            let mut neighs = vec![];

            for _ in 0..pointsize {
                neighs.push(
                    Self::next_u32(&mut binfile)
                        .expect("Graph file declared more nodes than actually found")
                        as usize,
                );
            }

            adjacency.push(node::Node {
                neighbors: FixedSet::new(neighs),
                catapults: RwLock::new(T::new()),
                payload: Box::new([0.0]),
            });
        }

        adjacency
    }
}
