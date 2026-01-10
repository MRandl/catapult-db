use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike};

pub struct SimilarityHasher {
    stored_vectors_dim: usize,
    /// random hyperplane normals
    projections: Vec<Vec<AlignedBlock>>,
}

impl SimilarityHasher {
    /// Constructs a new hasher with `num_hash` hyperplanes in dimension `dim`,
    /// seeded from the given `seed`. This is deterministic.
    pub fn new_seeded(num_hash: usize, stored_vectors_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_rng(num_hash, stored_vectors_dim, &mut rng)
    }

    fn with_rng<R: Rng>(num_hash: usize, stored_vectors_dim: usize, rng: &mut R) -> Self {
        assert!(
            stored_vectors_dim.is_multiple_of(SIMD_LANECOUNT),
            "dim must be multiple of SIMD_LANECOUNT"
        );

        let mut gaussian_iter = rng.sample_iter(StandardNormal);
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

        SimilarityHasher {
            stored_vectors_dim,
            projections,
        }
    }

    /// Hashes `vector` to a `k`-length binary signature.
    #[allow(unused)]
    pub fn hash(&self, vector: &[AlignedBlock]) -> Vec<bool> {
        assert!(
            vector.len() == self.stored_vectors_dim / SIMD_LANECOUNT,
            "input vector has wrong dimension"
        );
        self.projections
            .iter()
            .map(|proj| {
                let dot: f32 = vector.dot(proj);
                dot >= 0.0
            })
            .collect()
    }

    pub fn hash_int(&self, vector: &[AlignedBlock]) -> usize {
        assert_eq!(
            vector.len(),
            self.stored_vectors_dim / SIMD_LANECOUNT,
            "input vector has wrong dimension"
        );
        assert!(self.projections.len() <= u64::BITS as usize); // less than 64 planes to fit signature in u64

        let mut projected = 0usize;
        for plane in self.projections.iter() {
            projected = projected << 1 | ((plane.dot(vector) >= 0.0) as usize);
        }

        projected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_seeded_determinism() {
        let h1 = SimilarityHasher::new_seeded(16, SIMD_LANECOUNT * 2, 12345);
        let h2 = SimilarityHasher::new_seeded(16, SIMD_LANECOUNT * 2, 12345);
        // Same seed → identical projections
        assert_eq!(h1.projections, h2.projections);
    }

    #[test]
    fn test_hash_seeded_matches_with_rng() {
        // compare new_seeded against manual with_rng
        let mut rng1 = StdRng::seed_from_u64(42);
        let manual = SimilarityHasher::with_rng(8, SIMD_LANECOUNT, &mut rng1);
        let seeded = SimilarityHasher::new_seeded(8, SIMD_LANECOUNT, 42);
        assert_eq!(manual.projections, seeded.projections);
    }

    #[test]
    fn test_hash_consistency_same_input() {
        let hasher = SimilarityHasher::new_seeded(16, SIMD_LANECOUNT, 123);
        let vec = vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])];
        let hash1 = hasher.hash(&vec);
        let hash2 = hasher.hash(&vec);
        assert_eq!(hash1, hash2, "Hash must be consistent for same input");
    }

    #[test]
    fn test_hash_difference_on_orthogonal_vectors() {
        let dim = SIMD_LANECOUNT;
        let hasher = SimilarityHasher::new_seeded(32, dim, 999);
        let mut v1 = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        let mut v2 = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        v1[0].data[0] = 1.0; // Unit vector along x
        v2[0].data[1] = 1.0; // Unit vector along y
        let h1 = hasher.hash(&v1);
        let h2 = hasher.hash(&v2);

        let hamming_distance: usize = h1.iter().zip(&h2).filter(|(a, b)| a != b).count();
        assert!(hamming_distance > 0);
    }

    #[test]
    fn test_hash_dot_sign_behavior() {
        // Manually define a simple hasher with known projection
        let hasher = SimilarityHasher {
            stored_vectors_dim: SIMD_LANECOUNT,
            projections: vec![
                vec![AlignedBlock::new([
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ])],
                vec![AlignedBlock::new([
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ])],
            ], // x-axis and y-axis projections
        };

        let input = vec![AlignedBlock::new([
            2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])];
        let result = hasher.hash(&input);

        // Expect: dot([2,-3], [1,0]) = 2 → true
        //         dot([2,-3], [0,1]) = -3 → false
        assert_eq!(result, vec![true, false]);
    }

    #[test]
    #[should_panic(expected = "dim must be multiple of SIMD_LANECOUNT")]
    fn test_panics_on_non_multiple_dimension() {
        // Line 29: dim not multiple of SIMD_LANECOUNT
        let _ = SimilarityHasher::new_seeded(8, SIMD_LANECOUNT + 1, 42);
    }

    #[test]
    #[should_panic(expected = "input vector has wrong dimension")]
    fn test_hash_panics_on_wrong_dimension() {
        // Line 58: vector length doesn't match
        let hasher = SimilarityHasher::new_seeded(8, SIMD_LANECOUNT * 2, 42);
        let wrong_vec = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])]; // Only 1 block instead of 2
        let _ = hasher.hash(&wrong_vec);
    }

    #[test]
    #[should_panic(expected = "input vector has wrong dimension")]
    fn test_hash_int_panics_on_wrong_dimension() {
        // Line 72: vector length doesn't match
        let hasher = SimilarityHasher::new_seeded(8, SIMD_LANECOUNT * 2, 42);
        let wrong_vec = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])]; // Only 1 block instead of 2
        let _ = hasher.hash_int(&wrong_vec);
    }
}
