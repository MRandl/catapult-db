use std::sync::RwLock;

use crate::sets::catapults::CatapultEvictingStructure;
use crate::{numerics::AlignedBlock, search::hash_start::hasher::SimilarityHasher};

pub struct EngineStarter<T: CatapultEvictingStructure> {
    hasher: SimilarityHasher,
    starting_node: usize,
    catapults: Box<[RwLock<T>]>,
    enabled_catapults: bool,
}

pub struct StartingPoints {
    pub signature: usize,
    pub start_points: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct EngineStarterParams {
    pub num_hash: usize,
    pub plane_dim: usize,
    pub starting_node: usize,
    pub seed: u64,
    pub enabled_catapults: bool,
}

impl EngineStarterParams {
    pub fn new(
        num_hash: usize,
        plane_dim: usize,
        starting_node: usize,
        seed: u64,
        enabled_catapults: bool,
    ) -> Self {
        Self {
            num_hash,
            plane_dim,
            starting_node,
            seed,
            enabled_catapults,
        }
    }
}

impl<T> EngineStarter<T>
where
    T: CatapultEvictingStructure,
{
    pub fn new(params: EngineStarterParams) -> Self {
        let num_hash = params.num_hash;
        let plane_dim = params.plane_dim;
        let starting_node = params.starting_node;
        let seed = params.seed;
        let enabled_catapults = params.enabled_catapults;

        let hasher = SimilarityHasher::new_seeded(num_hash, plane_dim, seed);

        let amount_of_catapult_sets = 1 << num_hash;
        let mut catapult_vecs = Vec::with_capacity(amount_of_catapult_sets);
        for _ in 0..amount_of_catapult_sets {
            catapult_vecs.push(RwLock::new(T::new()));
        }

        Self {
            hasher,
            starting_node,
            catapults: catapult_vecs.into_boxed_slice(),
            enabled_catapults,
        }
    }

    pub fn select_starting_points(&self, query: &[AlignedBlock]) -> StartingPoints {
        let signature = self.hasher.hash_int(query);
        let catapults = if self.enabled_catapults {
            let mut catapults = self.catapults[signature].read().unwrap().to_vec();
            catapults.push(self.starting_node);
            catapults
        } else {
            vec![self.starting_node]
        };
        StartingPoints {
            signature,
            start_points: catapults,
        }
    }

    pub fn new_catapult(&self, signature: usize, new_cata: usize) {
        self.catapults[signature].write().unwrap().insert(new_cata);
    }

    pub fn clear_all_catapults(&self) {
        for catapult_set in self.catapults.iter() {
            catapult_set.write().unwrap().clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{numerics::SIMD_LANECOUNT, sets::catapults::FifoSet};

    type TestEngineStarter = EngineStarter<FifoSet<30>>;

    fn create_test_query(value: f32) -> Vec<AlignedBlock> {
        vec![AlignedBlock::new([value; SIMD_LANECOUNT])]
    }

    #[test]
    fn test_new_with_seed() {
        let params: EngineStarterParams = EngineStarterParams {
            num_hash: 8,
            plane_dim: SIMD_LANECOUNT,
            starting_node: 1000,
            seed: 42,
            enabled_catapults: true,
        };
        let starter = TestEngineStarter::new(params);
        assert_eq!(starter.starting_node, 1000);
        assert_eq!(starter.catapults.len(), 1 << 8); // 2^8 = 256
    }

    #[test]
    fn test_new_creates_correct_catapult_count() {
        let params: EngineStarterParams = EngineStarterParams {
            num_hash: 8,
            plane_dim: SIMD_LANECOUNT,
            starting_node: 1000,
            seed: 42,
            enabled_catapults: true,
        };
        let starter = TestEngineStarter::new(params);
        assert_eq!(starter.starting_node, 1000);
        assert_eq!(starter.catapults.len(), 1 << 8); // 2^8 = 256
    }

    #[test]
    fn test_select_starting_points_includes_starting_node() {
        let params: EngineStarterParams = EngineStarterParams {
            num_hash: 8,
            plane_dim: SIMD_LANECOUNT,
            starting_node: 1000,
            seed: 42,
            enabled_catapults: true,
        };
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result = starter.select_starting_points(&query);

        // Should always include the starting_node
        assert!(result.start_points.contains(&1000));
    }

    #[test]
    fn test_insert_and_retrieve_catapult() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result = starter.select_starting_points(&query);
        let signature = result.signature;

        // Insert a catapult for this signature
        starter.new_catapult(signature, 42);

        // Retrieve again and verify the catapult is included
        let result2 = starter.select_starting_points(&query);
        assert!(result2.start_points.contains(&42));
        assert!(result2.start_points.contains(&1000)); // starting_node still there

        starter.clear_all_catapults();
        let result3 = starter.select_starting_points(&query);
        assert!(!result3.start_points.contains(&42));
        assert!(result3.start_points.contains(&1000)); // starting_node still there
    }

    #[test]
    fn test_catapult_persists_across_queries() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result = starter.select_starting_points(&query);
        starter.new_catapult(result.signature, 99);

        // Query again multiple times
        for _ in 0..3 {
            let result_again = starter.select_starting_points(&query);
            assert!(result_again.start_points.contains(&99));
        }
    }

    #[test]
    fn test_same_query_returns_consistent_signature() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result1 = starter.select_starting_points(&query);
        let result2 = starter.select_starting_points(&query);

        // Same query should return same signature
        assert_eq!(result1.signature, result2.signature);
        assert_eq!(result1.start_points, result2.start_points);
    }

    #[test]
    fn test_different_queries_different_signatures() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query1 = create_test_query(1.0);
        let query2 = create_test_query(-1.0);

        let result1 = starter.select_starting_points(&query1);
        let result2 = starter.select_starting_points(&query2);

        // Different queries should likely return different signatures
        // (Not guaranteed but highly probable with good hashing)
        assert_ne!(result1.signature, result2.signature);
    }
    #[test]
    fn test_determinism_with_seed() {
        let params1 = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter1 = TestEngineStarter::new(params1);
        let params2 = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter2 = TestEngineStarter::new(params2);
        let query = create_test_query(1.5);

        let result1 = starter1.select_starting_points(&query);
        let result2 = starter2.select_starting_points(&query);

        // Same seed and query should give same signature
        assert_eq!(result1.signature, result2.signature);
    }

    #[test]
    fn test_different_seeds_different_signatures() {
        let params1 = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter1 = TestEngineStarter::new(params1);
        let params2 = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 99, true);
        let starter2 = TestEngineStarter::new(params2);
        let query = create_test_query(1.0);

        let result1 = starter1.select_starting_points(&query);
        let result2 = starter2.select_starting_points(&query);

        // Different seeds should give different signatures for same query
        assert_ne!(result1.signature, result2.signature);
    }

    #[test]
    fn test_multiple_catapults_same_signature() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result = starter.select_starting_points(&query);
        let signature = result.signature;

        // Insert multiple catapults for the same signature
        starter.new_catapult(signature, 100);
        starter.new_catapult(signature, 200);
        starter.new_catapult(signature, 300);

        let result2 = starter.select_starting_points(&query);

        // All catapults should be present
        assert!(result2.start_points.contains(&100));
        assert!(result2.start_points.contains(&200));
        assert!(result2.start_points.contains(&300));
        assert!(result2.start_points.contains(&1000));
    }

    #[test]
    fn test_different_signatures_independent_catapults() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query1 = create_test_query(1.0);
        let query2 = create_test_query(-1.0);

        let result1 = starter.select_starting_points(&query1);
        let result2 = starter.select_starting_points(&query2);

        // Insert catapults for different signatures
        starter.new_catapult(result1.signature, 111);
        starter.new_catapult(result2.signature, 222);

        let result1_again = starter.select_starting_points(&query1);
        let result2_again = starter.select_starting_points(&query2);

        // Each signature should only have its own catapults
        assert!(result1_again.start_points.contains(&111));
        assert!(!result1_again.start_points.contains(&222));

        assert!(result2_again.start_points.contains(&222));
        assert!(!result2_again.start_points.contains(&111));
    }

    #[test]
    fn test_signature_within_bounds() {
        let num_hash = 8;
        let params = EngineStarterParams::new(num_hash, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);

        for i in 0..10 {
            let query = create_test_query(i as f32);
            let result = starter.select_starting_points(&query);

            // Signature should be within bounds of catapult array
            assert!(result.signature < (1 << num_hash));
        }
    }

    #[test]
    fn test_multidimensional_query() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT * 4, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = vec![
            AlignedBlock::new([1.0; SIMD_LANECOUNT]),
            AlignedBlock::new([2.0; SIMD_LANECOUNT]),
            AlignedBlock::new([3.0; SIMD_LANECOUNT]),
            AlignedBlock::new([4.0; SIMD_LANECOUNT]),
        ];

        let result = starter.select_starting_points(&query);

        // Should return valid signature and include starting_node
        assert!(result.signature < (1 << 8));
        assert!(result.start_points.contains(&1000));
    }

    #[test]
    fn test_consistency_across_multiple_calls() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(3.0);

        let result1 = starter.select_starting_points(&query);
        let result2 = starter.select_starting_points(&query);
        let result3 = starter.select_starting_points(&query);

        assert_eq!(result1.signature, result2.signature);
        assert_eq!(result2.signature, result3.signature);
        assert_eq!(result1.start_points, result2.start_points);
        assert_eq!(result2.start_points, result3.start_points);
    }

    #[test]
    fn test_fifo_eviction_behavior() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 1000, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(1.0);

        let result = starter.select_starting_points(&query);
        let signature = result.signature;

        // Insert more than FifoSet capacity (30 items)
        for i in 0..35 {
            starter.new_catapult(signature, i);
        }

        let result2 = starter.select_starting_points(&query);

        // Should have at most 30 catapults + 1 starting_node
        assert!(result2.start_points.len() <= 31);

        // Starting node should always be present
        assert!(result2.start_points.contains(&1000));

        // Oldest entries should be evicted (0-4 should be gone)
        assert!(!result2.start_points.contains(&0));
        assert!(!result2.start_points.contains(&1));
        assert!(!result2.start_points.contains(&2));
        assert!(!result2.start_points.contains(&3));
        assert!(!result2.start_points.contains(&4));

        // Newest entries should be present (30-34)
        assert!(result2.start_points.contains(&30));
        assert!(result2.start_points.contains(&31));
        assert!(result2.start_points.contains(&32));
        assert!(result2.start_points.contains(&33));
        assert!(result2.start_points.contains(&34));
    }

    #[test]
    fn test_empty_catapult_only_returns_starting_node() {
        let params = EngineStarterParams::new(8, SIMD_LANECOUNT, 999, 42, true);
        let starter = TestEngineStarter::new(params);
        let query = create_test_query(5.5);

        let result = starter.select_starting_points(&query);

        // With no catapults inserted, should only return the starting_node
        assert_eq!(result.start_points.len(), 1);
        assert_eq!(result.start_points[0], 999);
    }
}
