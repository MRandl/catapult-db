use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use std::collections::HashMap;
use std::sync::RwLock;

use crate::{numerics::AlignedBlock, search::hash_start::hasher::SimilarityHasher};

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
            let seed = signature;
            // seeding the rand with the signature. This prevents a scheduling shenanigan where having two
            // threads trying to set this without seeding would make the starter points inconsistent across runs,
            // technically a race condition (benign).
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::numerics::SIMD_LANECOUNT;

    fn create_test_query(value: f32) -> Vec<AlignedBlock> {
        vec![AlignedBlock::new([value; SIMD_LANECOUNT])]
    }

    #[test]
    fn test_new_with_seed() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        assert_eq!(starter.max_len, 1000);
        assert_eq!(starter.cached_starters.read().unwrap().len(), 0);
    }

    #[test]
    fn test_new_without_seed() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 500, None);
        assert_eq!(starter.max_len, 500);
        assert_eq!(starter.cached_starters.read().unwrap().len(), 0);
    }

    #[test]
    fn test_select_starting_points_returns_correct_count() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, 10);

        assert_eq!(points.len(), 10);
    }

    #[test]
    fn test_select_starting_points_within_bounds() {
        let graph_size = 500;
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, graph_size, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, 50);

        // All indices should be within [0, graph_size)
        for &idx in &points {
            assert!(idx < graph_size, "Index {} out of bounds", idx);
        }
    }

    #[test]
    fn test_select_starting_points_unique_indices() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, 20);

        // All indices should be unique
        let mut sorted_points = points.clone();
        sorted_points.sort_unstable();
        sorted_points.dedup();
        assert_eq!(
            sorted_points.len(),
            points.len(),
            "Indices should be unique"
        );
    }

    #[test]
    fn test_caching_same_query() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query = create_test_query(1.0);

        let points1 = starter.select_starting_points(&query, 10);
        let points2 = starter.select_starting_points(&query, 10);

        // Same query should return same cached results
        assert_eq!(points1, points2);

        // Cache should have one entry
        assert_eq!(starter.cached_starters.read().unwrap().len(), 1);
    }

    #[test]
    fn test_different_queries_different_results() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query1 = create_test_query(1.0);
        let query2 = create_test_query(-1.0);

        let points1 = starter.select_starting_points(&query1, 10);
        let points2 = starter.select_starting_points(&query2, 10);

        // Different queries should likely return different results
        // (Not guaranteed but highly probable with good hashing)
        assert_ne!(points1, points2);

        // Cache should have two entries
        assert_eq!(starter.cached_starters.read().unwrap().len(), 2);
    }

    #[test]
    fn test_determinism_with_seed() {
        let starter1 = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let starter2 = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query = create_test_query(1.5);

        let points1 = starter1.select_starting_points(&query, 15);
        let points2 = starter2.select_starting_points(&query, 15);

        // Same seed and query should give same results
        assert_eq!(points1, points2);
    }

    #[test]
    fn test_different_seeds_different_results() {
        let starter1 = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let starter2 = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(99));
        let query = create_test_query(1.0);

        let points1 = starter1.select_starting_points(&query, 10);
        let points2 = starter2.select_starting_points(&query, 10);

        // Different seeds should give different results for same query
        assert_ne!(points1, points2);
    }

    #[test]
    fn test_select_single_starting_point() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 100, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, 1);

        assert_eq!(points.len(), 1);
        assert!(points[0] < 100);
    }

    #[test]
    fn test_select_all_points() {
        let graph_size = 50;
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, graph_size, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, graph_size);

        assert_eq!(points.len(), graph_size);

        // Should be all indices from 0 to graph_size-1
        let mut sorted_points = points.clone();
        sorted_points.sort_unstable();
        assert_eq!(sorted_points, (0..graph_size).collect::<Vec<_>>());
    }

    #[test]
    fn test_multiple_queries_build_cache() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));

        for i in 0..5 {
            let query = create_test_query(i as f32);
            starter.select_starting_points(&query, 10);
        }

        // Cache should have 5 entries (or fewer if there were hash collisions)
        let cache_size = starter.cached_starters.read().unwrap().len();
        assert!(cache_size <= 5);
        assert!(cache_size > 0);
    }

    #[test]
    fn test_multidimensional_query() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT * 4, 1000, Some(42));
        let query = vec![
            AlignedBlock::new([1.0; SIMD_LANECOUNT]),
            AlignedBlock::new([2.0; SIMD_LANECOUNT]),
            AlignedBlock::new([3.0; SIMD_LANECOUNT]),
            AlignedBlock::new([4.0; SIMD_LANECOUNT]),
        ];

        let points = starter.select_starting_points(&query, 20);

        assert_eq!(points.len(), 20);
        for &idx in &points {
            assert!(idx < 1000);
        }
    }

    #[test]
    fn test_consistency_across_multiple_calls() {
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, 1000, Some(42));
        let query = create_test_query(3.14);

        let points1 = starter.select_starting_points(&query, 15);
        let points2 = starter.select_starting_points(&query, 15);
        let points3 = starter.select_starting_points(&query, 15);

        assert_eq!(points1, points2);
        assert_eq!(points2, points3);
    }

    #[test]
    fn test_large_k_value() {
        let graph_size = 10000;
        let k = 500;
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, graph_size, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, k);

        assert_eq!(points.len(), k);

        // Verify all unique
        let mut sorted = points.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), k);
    }

    #[test]
    fn test_small_graph() {
        let graph_size = 10;
        let starter = EngineStarter::new(8, SIMD_LANECOUNT, graph_size, Some(42));
        let query = create_test_query(1.0);

        let points = starter.select_starting_points(&query, 5);

        assert_eq!(points.len(), 5);
        for &idx in &points {
            assert!(idx < graph_size);
        }
    }
}
