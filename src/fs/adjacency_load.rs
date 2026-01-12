use crate::{
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{
        AdjacencyGraph, Node,
        graph_algo::{FlatCatapultChoice, FlatSearch},
        hash_start::{EngineStarter, EngineStarterParams},
    },
    sets::{catapults::CatapultEvictingStructure, fixed::FlatFixedSet},
};

use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::PathBuf,
};

impl<T: CatapultEvictingStructure> AdjacencyGraph<T, FlatSearch> {
    fn next_bytes<I, const N: usize>(iter: &mut I) -> Option<[u8; N]>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        let mut bytes = [0u8; N];
        for b in &mut bytes {
            let next = iter.next();
            if let Some(Ok(byte)) = next {
                *b = byte
            } else {
                return None;
            }
        }
        Some(bytes)
    }

    fn next_u32<I>(iter: &mut I) -> Option<u32>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(u32::from_le_bytes)
    }

    fn next_u64<I>(iter: &mut I) -> Option<u64>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 8>(iter).map(u64::from_le_bytes)
    }

    fn next_f32<I>(iter: &mut I) -> Option<f32>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        Self::next_bytes::<I, 4>(iter).map(f32::from_le_bytes)
    }

    fn next_payload<I>(iter: &mut I, size: usize) -> Option<Vec<AlignedBlock>>
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
        Some(payload)
    }

    pub fn load_flat_from_path(
        graph_path: PathBuf,
        payload_path: PathBuf,
        mut engine_params: EngineStarterParams,
    ) -> Self {
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

        while let Some(pointsize) = Self::next_u32(&mut graph_file) {
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
                neighbors: FlatFixedSet::new(neighs),
                payload: associated_payload.into_boxed_slice(),
            });
        }

        // we should have read all of the file contents by now.
        assert!(graph_file.count() == 0);
        assert!(payload_file.count() == 0);

        engine_params.starting_node = entry_point as usize;
        AdjacencyGraph::new_flat(
            adjacency,
            EngineStarter::<T>::new(engine_params),
            FlatCatapultChoice::from_bool(engine_params.enabled_catapults),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        numerics::SIMD_LANECOUNT,
        search::{AdjacencyGraph, graph_algo::FlatSearch, hash_start::EngineStarterParams},
        sets::catapults::FifoSet,
    };

    #[test]
    fn loading_example_graph() {
        let graph_path = "test_index/ann";
        let payload_path = "test_index/ann_vectors.bin";
        let engine_params1 = EngineStarterParams::new(4, SIMD_LANECOUNT, 0, 42, true);
        let engine_params2 = EngineStarterParams::new(4, SIMD_LANECOUNT, 0, 42, false);

        let graphed1 = AdjacencyGraph::<FifoSet<20>, FlatSearch>::load_flat_from_path(
            graph_path.into(),
            payload_path.into(),
            engine_params1,
        );

        let graphed2 = AdjacencyGraph::<FifoSet<20>, FlatSearch>::load_flat_from_path(
            graph_path.into(),
            payload_path.into(),
            engine_params2,
        );

        assert!(graphed1.len() == 4);
        assert!(graphed2.len() == 4);
        assert!(graphed1.len() == graphed2.len());
    }
}
