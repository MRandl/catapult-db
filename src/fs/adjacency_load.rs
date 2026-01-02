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
    fn next_u32<I, const LOAD_LI_ENDIAN: bool>(iter: &mut I) -> Result<u32, String>
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

        if LOAD_LI_ENDIAN {
            Ok(u32::from_le_bytes(bytes))
        } else {
            Ok(u32::from_be_bytes(bytes))
            // For the one dude that wants to run catapultdb on a system that somehow runs on big endian.
            // I got your back, man. Also, go take a shower.
        }
    }

    fn next_u64<I, const LOAD_LI_ENDIAN: bool>(iter: &mut I) -> Result<u64, String>
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

        if LOAD_LI_ENDIAN {
            Ok(u64::from_le_bytes(bytes))
        } else {
            Ok(u64::from_be_bytes(bytes))
        }
    }

    fn next_f32<I, const LOAD_LI_ENDIAN: bool>(iter: &mut I) -> Result<f32, String>
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

        if LOAD_LI_ENDIAN {
            Ok(f32::from_le_bytes(bytes))
        } else {
            Ok(f32::from_be_bytes(bytes))
        }
    }

    fn next_payload<I, const LOAD_LI_ENDIAN: bool>(
        iter: &mut I,
        size: usize,
    ) -> Result<Vec<AlignedBlock>, String>
    where
        I: Iterator<Item = Result<u8, Error>>,
    {
        assert!(size.is_multiple_of(SIMD_LANECOUNT));

        let final_length = size / SIMD_LANECOUNT;
        let mut payload = Vec::with_capacity(final_length);
        for _ in 0..final_length {
            let mut block = [0.0; 8];
            for entry in block.iter_mut() {
                *entry = Self::next_f32::<_, LOAD_LI_ENDIAN>(iter)?;
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

        let full_size =
            Self::next_u64::<_, LOAD_LI_ENDIAN>(&mut graph_file).expect("Misconfigured header");
        let max_degree =
            Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut graph_file).expect("Misconfigured header");
        let entry_point =
            Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut graph_file).expect("Misconfigured header");
        let num_frozen =
            Self::next_u64::<_, LOAD_LI_ENDIAN>(&mut graph_file).expect("Misconfigured header");

        let npoints =
            Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut payload_file).expect("Misconfigured header");
        let payload_dim = Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut payload_file)
            .expect("Misconfigured header") as usize;

        println!(
            "size {full_size} - degree {max_degree} - entry point {entry_point} - num frozen {num_frozen} - npoints {npoints} - payload_dim {payload_dim}",
        );

        let mut adjacency = Vec::new();

        while let Ok(pointsize) = Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut graph_file) {
            let mut neighs = vec![];

            for _ in 0..pointsize {
                neighs.push(
                    Self::next_u32::<_, LOAD_LI_ENDIAN>(&mut graph_file)
                        .expect("Graph file declared more nodes than actually found")
                        as usize,
                );
            }

            let associated_payload =
                Self::next_payload::<_, LOAD_LI_ENDIAN>(&mut payload_file, payload_dim)
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
