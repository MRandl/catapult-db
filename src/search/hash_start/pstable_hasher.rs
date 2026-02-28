use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{StandardNormal, Uniform};

use crate::numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike};

#[allow(dead_code)]
pub struct PStableHashingBlock {
    a_vectors: Vec<Vec<AlignedBlock>>,
    bs: Vec<f32>,
    w: f32,
}

impl PStableHashingBlock {
    #[allow(dead_code)]
    pub fn new_seeded(num_hash: usize, stored_vectors_dim: usize, seed: u64, w: f32) -> Self {
        let rng1 = StdRng::seed_from_u64(seed);
        let rng2 = StdRng::seed_from_u64(u64::MAX ^ seed);

        assert!(
            stored_vectors_dim.is_multiple_of(SIMD_LANECOUNT),
            "dim must be multiple of SIMD_LANECOUNT"
        );

        let mut gaussian_iter = rng1.sample_iter(StandardNormal);
        let mut uniform_iter = rng2.sample_iter(Uniform::new(0.0, w).unwrap());

        let projections: Vec<Vec<AlignedBlock>> = (0..num_hash)
            .map(|_| {
                (0..stored_vectors_dim / SIMD_LANECOUNT)
                    .map(|_| {
                        let mut block = [0.0; SIMD_LANECOUNT];
                        for b in block.iter_mut() {
                            *b = gaussian_iter.next().unwrap();
                        }
                        AlignedBlock::new(block)
                    })
                    .collect()
            })
            .collect();

        let bs: Vec<f32> = (0..num_hash)
            .map(|_| uniform_iter.next().unwrap())
            .collect();

        Self {
            a_vectors: projections,
            bs,
            w,
        }
    }

    fn hash_one(&self, q: &[AlignedBlock], index: usize) -> f32 {
        ((q.dot(&self.a_vectors[index]) + self.bs[index]) / self.w).floor()
    }

    fn big_h(&self, q: &[AlignedBlock]) -> Vec<f32> {
        let mut accumulated_hashes = Vec::with_capacity(self.a_vectors.len());
        for i in 0..self.a_vectors.len() {
            accumulated_hashes.push(self.hash_one(q, i));
        }
        accumulated_hashes
    }

    fn irelu(candidate: f32) -> u128 {
        // WARN this is somewhat problematic. It is not described in the LSHAPG paper how they handle
        // negative values and (afaik, not 100% sure) their implementation gets to look at all incoming
        // vectors before actually generating any hash, and this is used to normalize the values and
        // guarantee in the process that they won't be negative. This seems to be incompatible with distribution
        // changes when inserting. I'm willing to get contradicted on this, it's all conditional on me
        // understanding the situation correctly, which is not a given.
        //
        // For now, I'll do an integer-style ReLU (negative maps to 0). Values too large will overflow
        // with a discontinuity in the bucket neighborhood, which kills neighbor discovery around
        // the non-continuous point, but that is not a correctness issue, only a performance one.
        //
        // We can also do constant-based offset to erase some of the problems caused by ReLU.
        // I'll investigate what's best for that baseline.
        if candidate.is_finite() {
            let shifted_candidate = candidate.floor() + 20.;
            if shifted_candidate >= 0.0 {
                shifted_candidate as u128
            } else {
                0u128
            }
        } else {
            0u128
        }
    }

    #[allow(dead_code)]
    pub fn hash(&self, q: &[AlignedBlock]) -> u128 {
        let hh = self.big_h(q);
        let dims_per_k = u128::BITS as usize / hh.len();

        let mut zint_res = 0u128;

        for i in (0..dims_per_k).rev() {
            let mask = 1u128 << i;
            for j in 0..hh.len() {
                zint_res <<= 1;
                if (Self::irelu(hh[j]) & mask) != 0 {
                    zint_res |= 1;
                }
            }
        }

        zint_res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_irelu_positive_values() {
        assert_eq!(PStableHashingBlock::irelu(5.7), 5);
        assert_eq!(PStableHashingBlock::irelu(0.0), 0);
        assert_eq!(PStableHashingBlock::irelu(123.99), 123);
        assert_eq!(PStableHashingBlock::irelu(1.0), 1);
    }

    #[test]
    fn test_irelu_negative_values() {
        // Negative values should map to 0 (ReLU behavior)
        assert_eq!(PStableHashingBlock::irelu(-1.0), 0);
        assert_eq!(PStableHashingBlock::irelu(-100.5), 0);
    }

    #[test]
    fn test_irelu_special_values() {
        // Non-finite values should map to 0
        assert_eq!(PStableHashingBlock::irelu(f32::NAN), 0);
        assert_eq!(PStableHashingBlock::irelu(f32::INFINITY), 0);
        assert_eq!(PStableHashingBlock::irelu(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn test_hash_deterministic() {
        // Same input should always produce same hash
        let hasher = PStableHashingBlock::new_seeded(2, 16, 42, 1.0);
        let q = vec![AlignedBlock::new([1.0; 16])];

        let hash1 = hasher.hash(&q);
        let hash2 = hasher.hash(&q);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_inputs_different_outputs() {
        // Different inputs should (usually) produce different hashes
        let hasher = PStableHashingBlock::new_seeded(2, 16, 42, 1.0);

        let q1 = vec![AlignedBlock::new([1.0; 16])];
        let q2 = vec![AlignedBlock::new([-2.0; 16])];

        let hash1 = hasher.hash(&q1);
        let hash2 = hasher.hash(&q2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_same_seed_same_hasher() {
        // Two hashers with the same seed should produce the same hash for the same input
        let hasher1 = PStableHashingBlock::new_seeded(4, 16, 99, 2.0);
        let hasher2 = PStableHashingBlock::new_seeded(4, 16, 99, 2.0);

        let q = vec![AlignedBlock::new([3.0; 16])];

        assert_eq!(hasher1.hash(&q), hasher2.hash(&q));
    }

    #[test]
    fn test_hash_different_seeds_different_hashers() {
        // Two hashers with different seeds should (usually) produce different hashes
        let hasher1 = PStableHashingBlock::new_seeded(4, 16, 1, 2.0);
        let hasher2 = PStableHashingBlock::new_seeded(4, 16, 2, 2.0);

        let q = vec![AlignedBlock::new([3.0; 16])];

        assert_ne!(hasher1.hash(&q), hasher2.hash(&q));
    }

    #[test]
    fn test_hash_multi_block_vectors() {
        // Vectors spanning multiple AlignedBlocks (dim=32)
        let hasher = PStableHashingBlock::new_seeded(2, 32, 7, 1.0);
        let q = vec![AlignedBlock::new([1.0; 16]), AlignedBlock::new([1.0; 16])];

        let hash1 = hasher.hash(&q);
        let hash2 = hasher.hash(&q);
        assert_eq!(hash1, hash2);
    }
}
