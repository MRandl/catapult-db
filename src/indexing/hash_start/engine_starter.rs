use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use std::collections::HashMap;
use std::sync::RwLock;

use crate::indexing::SimilarityHasher;
use crate::numerics::AlignedBlock;

pub struct EngineStarter {
    hasher: SimilarityHasher,
    cached_starters: RwLock<HashMap<u64, Vec<usize>>>,
    max_len: usize,
}

impl EngineStarter {
    pub fn new(num_hash: usize, plane_dim: usize, graph_size: usize, seed: Option<u64>) -> Self {
        let hasher = if let Some(seed) = seed {
            SimilarityHasher::new_seeded(num_hash, plane_dim, seed)
        } else {
            SimilarityHasher::new(num_hash, plane_dim)
        };

        Self {
            hasher,
            cached_starters: RwLock::new(HashMap::new()),
            max_len: graph_size,
        }
    }

    pub fn select_starting_points(&self, query: &[AlignedBlock], k: usize) -> Vec<usize> {
        let signature = self.hasher.hash_int(query);
        if let Some(starters) = self.cached_starters.read().unwrap().get(&signature) {
            starters.clone()
        } else {
            let seed =
                (query[0].data[0].to_bits() as u64) << 32 | query[0].data[1].to_bits() as u64;

            let mut rng = StdRng::seed_from_u64(seed);
            let indices = sample(&mut rng, self.max_len, k).into_vec();
            self.cached_starters
                .write()
                .unwrap()
                .insert(signature, indices.clone());
            indices
        }
    }
}
