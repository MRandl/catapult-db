use rand::{RngCore, SeedableRng, rngs::StdRng, seq::index::sample};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use crate::indexing::SimilarityHasher;

pub struct EngineStarter {
    hasher: SimilarityHasher,
    cached_starters: RwLock<HashMap<u64, Vec<usize>>>,
    rand: Mutex<StdRng>,
    max_len: usize,
}

impl EngineStarter {
    pub fn new(num_hash: usize, plane_dim: usize, graph_size: usize, seed: Option<u64>) -> Self {
        let hasher = if let Some(seed) = seed {
            SimilarityHasher::new_seeded(num_hash, plane_dim, seed)
        } else {
            SimilarityHasher::new(num_hash, plane_dim)
        };

        let rng = StdRng::seed_from_u64(seed.unwrap_or(rand::rng().next_u64()));

        Self {
            hasher,
            cached_starters: RwLock::new(HashMap::new()),
            rand: Mutex::new(rng),
            max_len: graph_size,
        }
    }

    pub fn select_starting_points(&self, query: &[f32], k: usize) -> Vec<usize> {
        let signature = self.hasher.hash_int(query);
        if let Some(starters) = self.cached_starters.read().unwrap().get(&signature) {
            starters.clone()
        } else {
            let indices = sample(&mut self.rand.lock().unwrap(), self.max_len, k).into_vec();
            self.cached_starters
                .write()
                .unwrap()
                .insert(signature, indices.clone());
            indices
        }
    }
}
