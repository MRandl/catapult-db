use crate::{
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{AdjacencyGraph, GraphSearchAlgorithm, Node},
    sets::catapults::{CatapultEvictingStructure, FixedSet},
};

use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::PathBuf,
    sync::RwLock,
};

impl<T: CatapultEvictingStructure, GraphSearchType: GraphSearchAlgorithm>
    AdjacencyGraph<T, GraphSearchType>
{
    fn next_bytes<I, const N: usize>(iter: &mut I) -> Result<[u8; N], String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        let mut bytes = [0u8; N];
        for b in &mut bytes {
            // .transpose() turns Option<Result<T, E>> into Result<Option<T>, E>
            *b = iter
                .next()
                .transpose()
                .map_err(|e| e.to_string())?
                .ok_or("Unexpected end of stream")?;
        }
        Ok(bytes)
    }

    // Now your specific functions are one-liners:

    fn next_u32<I>(iter: &mut I) -> Result<u32, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(u32::from_le_bytes)
    }

    fn next_u64<I>(iter: &mut I) -> Result<u64, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 8>(iter).map(u64::from_le_bytes)
    }

    fn next_f32<I>(iter: &mut I) -> Result<f32, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(f32::from_le_bytes)
    }

    fn next_payload<I>(iter: &mut I, size: usize) -> Result<Vec<AlignedBlock>, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        assert!(size.is_multiple_of(SIMD_LANECOUNT));

        let final_length = size / SIMD_LANECOUNT;
        let mut payload = Vec::with_capacity(final_length);
        for _ in 0..final_length {
            let mut block = [0.0; SIMD_LANECOUNT];
            for entry in block.iter_mut() {
                *entry = Self::next_f32(iter)?;
            }
            payload.push(AlignedBlock::new(block));
        }
        Ok(payload)
    }

    pub fn load_from_path<const LOAD_LI_ENDIAN: bool>(
        graph_path: PathBuf,
        payload_path: PathBuf,
    ) -> Vec<Node<T, GraphSearchType>> {
        let mut graph_file = BufReader::new(File::open(graph_path).expect("FNF")).bytes();
        let mut payload_file = BufReader::new(File::open(payload_path).expect("FNF")).bytes();

        let full_size = Self::next_u64(&mut graph_file).expect("Misconfigured header");
        let max_degree = Self::next_u32(&mut graph_file).expect("Misconfigured header");
        let entry_point = Self::next_u32(&mut graph_file).expect("Misconfigured header");
        let num_frozen = Self::next_u64(&mut graph_file).expect("Misconfigured header");

        let npoints = Self::next_u32(&mut payload_file).expect("Misconfigured header");
        let payload_dim = Self::next_u32(&mut payload_file).expect("Misconfigured header") as usize;

        println!(
            "size {full_size} - degree {max_degree} - entry point {entry_point} - num frozen {num_frozen} - npoints {npoints} - payload_dim {payload_dim}",
        );

        let mut adjacency = Vec::new();

        while let Ok(pointsize) = Self::next_u32(&mut graph_file) {
            let mut neighs = vec![];

            for _ in 0..pointsize {
                neighs.push(
                    Self::next_u32(&mut graph_file)
                        .expect("Graph file declared more nodes than actually found")
                        as usize,
                );
            }

            let associated_payload = Self::next_payload(&mut payload_file, payload_dim)
                .expect("Error while parsing payloads");

            adjacency.push(Node {
                neighbors: FixedSet::new(neighs),
                catapults: RwLock::new(T::new()),
                payload: associated_payload.into_boxed_slice(),
            });
        }

        // we should have read all of the file contents by now.
        assert!(graph_file.count() == 0);
        assert!(payload_file.count() == 0);

        adjacency
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        search::{AdjacencyGraph, FlatSearch},
        sets::catapults::{CatapultEvictingStructure, FifoSet},
    };

    #[test]
    fn loading_example_graph() {
        let graph_path = "test_index/ann";
        let payload_path = "test_index/ann_vectors.bin";
        const LITTLE_ENDIAN: bool = true;
        let graphed = AdjacencyGraph::<FifoSet<20>, FlatSearch>::load_from_path::<LITTLE_ENDIAN>(
            graph_path.into(),
            payload_path.into(),
        );

        assert!(graphed.len() == 4);
        for node in graphed.into_iter() {
            assert!(node.catapults.read().unwrap().to_vec().is_empty());
            assert!(node.neighbors.to_box(()).len() <= 2);
            for i in 0..node.payload.len() {
                assert!(node.payload[0] == node.payload[i]);
            }
        }
    }
}
