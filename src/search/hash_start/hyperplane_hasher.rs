use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike};

/// A locality-sensitive hasher using random hyperplane projections.
///
/// Maps high-dimensional vectors to binary signatures by projecting them onto
/// random hyperplanes with Gaussian-distributed normals. Each bit in the signature
/// indicates which side of a hyperplane the vector lies on (positive or negative).
/// Similar vectors tend to produce similar signatures, enabling efficient bucketing.
pub struct SimilarityHasher {
    stored_vectors_dim: usize,
    /// Random hyperplane normal vectors, each as a sequence of aligned blocks.
    /// One hyperplane per hash bit.
    projections: Vec<Vec<AlignedBlock>>,
}

impl SimilarityHasher {
    /// Creates a new deterministic LSH hasher with random hyperplane projections.
    ///
    /// Generates `num_hash` random hyperplanes with Gaussian-distributed normal vectors
    /// for projecting vectors in the specified dimension. The same seed produces identical
    /// hyperplanes, ensuring reproducibility.
    ///
    /// # Arguments
    /// * `num_hash` - Number of hash bits / hyperplanes to generate
    /// * `stored_vectors_dim` - Dimension of input vectors in f32 elements (not blocks)
    /// * `seed` - Random seed for deterministic hyperplane generation
    ///
    /// # Returns
    /// A new `SimilarityHasher` instance
    ///
    /// # Panics
    /// Panics if `stored_vectors_dim` is not a multiple of `SIMD_LANECOUNT`
    pub fn new_seeded(num_hash: usize, stored_vectors_dim: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);

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

    /// Hashes a vector to a binary signature represented as a vector of booleans.
    ///
    /// Each boolean indicates whether the vector's projection onto the corresponding
    /// hyperplane is non-negative (true) or negative (false).
    ///
    /// # Arguments
    /// * `vector` - The input vector as aligned blocks
    ///
    /// # Returns
    /// A vector of booleans representing the binary hash signature
    ///
    /// # Panics
    /// Panics if the vector dimension doesn't match the hasher's configured dimension
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

    /// Hashes a vector to an integer signature by packing bits into a usize.
    ///
    /// Computes the same LSH signature as `hash()` but returns it as a packed integer
    /// where each bit represents a hyperplane projection sign. This is more efficient
    /// for use as an array index or hash table key.
    ///
    /// # Arguments
    /// * `vector` - The input vector as aligned blocks
    ///
    /// # Returns
    /// An integer representing the packed binary hash signature
    ///
    /// # Panics
    /// Panics if the vector dimension doesn't match the hasher's configured dimension,
    /// or if `num_hash` exceeds the number of bits in a u64
    pub fn hash_int(&self, vector: &[AlignedBlock]) -> usize {
        assert_eq!(
            vector.len(),
            self.stored_vectors_dim / SIMD_LANECOUNT,
            "input vector has wrong dimension"
        );
        assert!(self.projections.len() <= usize::BITS as usize); // less than 64 planes to fit signature in u64

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
