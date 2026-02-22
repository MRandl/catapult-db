use crate::numerics::{AlignedBlock, VectorLike};

#[allow(dead_code)]
pub struct PStableHashingBlock {
    a_vectors: Vec<Vec<AlignedBlock>>,
    bs: Vec<f32>,
    w: f32,
}

impl PStableHashingBlock {
    /// Creates a new PStableHashingBlock with the given random projection vectors and bias terms.
    ///
    /// # Arguments
    /// * `a_vectors` - Random projection vectors for LSH. Each vector should be the same length as input vectors.
    /// * `bs` - Bias terms for each hash function. Must have the same length as `a_vectors`.
    ///
    /// # Panics
    /// Panics if `a_vectors` and `bs` have different lengths.
    pub fn new(a_vectors: Vec<Vec<AlignedBlock>>, bs: Vec<f32>, w: f32) -> Self {
        assert_eq!(
            a_vectors.len(),
            bs.len(),
            "Number of projection vectors must match number of bias terms"
        );
        Self { a_vectors, bs, w }
    }

    fn hash_one(&self, q: &[AlignedBlock], index: usize) -> f32 {
        ((q.dot(&self.a_vectors[index]) + self.bs[index]) / self.w).floor()
    }

    fn big_h(&self, q: &[AlignedBlock]) -> Vec<f32> {
        let mut accumulated_hashes = Vec::with_capacity(self.a_vectors.len());
        for i in (0..self.a_vectors.len()) {
            accumulated_hashes.push(self.hash_one(q, i));
        }
        accumulated_hashes
    }

    fn irelu(candidate: f32) -> u128 {
        // WARN this is somewhat problematic. It is not described in the LSHAPG paper how they handle
        // negative values and (afaik, not 100% sure) their implementation gets to look at all incoming
        // vectors before actually generating any hash. This seems to be incompatible with distribution
        // changes when inserting, and that's another claim of the paper. Not sure what to think of the
        // whole situation, it's a bit awkward. Also, I'm willing to get contradicted: this is all
        // conditional on me understanding what their code is doing.
        //
        // For now, I'll do an integer-style ReLU (negative maps to 0). If that doesn't fly,
        // we can always ping the authors.
        //
        // We can also do constant-based offset to erase some of the problems caused by ReLU.
        // I'll investigate what's best for that baseline.
        if candidate.is_finite() && candidate >= 0.0 {
            candidate.floor() as u128
        } else {
            0u128
        }
    }

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
    fn test_new_constructor() {
        let a_vectors = vec![
            vec![AlignedBlock::new([1.0; 16])],
            vec![AlignedBlock::new([2.0; 16])],
        ];
        let bs = vec![0.5, 1.5];

        let hasher = PStableHashingBlock::new(a_vectors.clone(), bs.clone(), 1.0);
        assert_eq!(hasher.a_vectors.len(), 2);
        assert_eq!(hasher.bs.len(), 2);
        assert_eq!(hasher.bs, bs);
    }

    #[test]
    #[should_panic(expected = "Number of projection vectors must match number of bias terms")]
    fn test_new_constructor_length_mismatch() {
        let a_vectors = vec![vec![AlignedBlock::new([1.0; 16])]];
        let bs = vec![0.5, 1.5]; // Mismatched length
        PStableHashingBlock::new(a_vectors, bs, 1.0);
    }

    #[test]
    fn test_hash_one_basic() {
        // Create a simple hasher with one projection vector
        let a_vectors = vec![vec![AlignedBlock::new([1.0; 16])]];
        let bs = vec![0.5; 16];
        let hasher = PStableHashingBlock::new(a_vectors, bs, 1.0);

        // Query vector
        let q = vec![AlignedBlock::new([2.0; 16])];

        // hash_one should compute dot product + floor(bias)
        // dot([2.0; 16], [1.0; 16]) = 32.0
        // (32.0 + 0.5 / 1).floor = 32.0
        let result = hasher.hash_one(&q, 0);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_big_h_multiple_projections() {
        // Create hasher with multiple projection vectors
        let a_vectors = vec![
            vec![AlignedBlock::new([1.0; 16])],
            vec![AlignedBlock::new([2.0; 16])],
            vec![AlignedBlock::new([0.5; 16])],
        ];
        let bs = vec![0.0, 1.5, 2.9];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q = vec![AlignedBlock::new([1.0; 16])];

        let hashes = hasher.big_h(&q);
        assert_eq!(hashes.len(), 3);

        // Expected values:
        // hash[0] = dot([1.0], [1.0]) + floor(0.0) = 16.0 + 0.0 = 16.0
        // hash[1] = dot([1.0], [2.0]) + floor(1.5) = 32.0 + 1.0 = 33.0
        // hash[2] = dot([1.0], [0.5]) + floor(2.9) = 8.0 + 2.0 = 10.0
        assert_eq!(hashes[0], 16.0);
        assert_eq!(hashes[1], 33.0);
        assert_eq!(hashes[2], 10.0);
    }

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
    fn test_hash_zero_vector() {
        // With zero query vector, hash should depend only on bias terms
        let a_vectors = vec![
            vec![AlignedBlock::new([1.0; 16])],
            vec![AlignedBlock::new([2.0; 16])],
        ];
        let bs = vec![0.0, 0.0];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q = vec![AlignedBlock::new([0.0; 16])];
        let hash = hasher.hash(&q);

        // All hashes should be 0, so the final hash should be 0
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_hash_deterministic() {
        // Same input should always produce same hash
        let a_vectors = vec![
            vec![AlignedBlock::new([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ])],
            vec![AlignedBlock::new([
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ])],
        ];
        let bs = vec![1.5, 2.5];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q = vec![AlignedBlock::new([1.0; 16])];

        let hash1 = hasher.hash(&q);
        let hash2 = hasher.hash(&q);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_inputs_different_outputs() {
        // Different inputs should (usually) produce different hashes
        let a_vectors = vec![
            vec![AlignedBlock::new([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ])],
            vec![AlignedBlock::new([
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ])],
        ];
        let bs = vec![1.5, 2.5];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q1 = vec![AlignedBlock::new([1.0; 16])];
        let q2 = vec![AlignedBlock::new([2.0; 16])];

        let hash1 = hasher.hash(&q1);
        let hash2 = hasher.hash(&q2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_z_order_interleaving() {
        // Test that the Z-order interleaving works correctly
        // With 2 hash functions and u128 (128 bits), we have 64 bits per hash function
        let a_vectors = vec![
            vec![AlignedBlock::new([1.0; 16])],
            vec![AlignedBlock::new([0.0; 16])],
        ];
        let bs = vec![0.0, 0.0];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        // Query that produces hash values [16.0, 0.0]
        let q = vec![AlignedBlock::new([1.0; 16])];
        let hash = hasher.hash(&q);

        // With irelu(16.0) = 16 = 0b10000 and irelu(0.0) = 0
        // The bits should be interleaved in Z-order
        assert_ne!(hash, 0); // Should not be zero since first hash is non-zero
    }

    #[test]
    fn test_hash_with_multiple_blocks() {
        // Test with vectors that span multiple AlignedBlocks
        let a_vectors = vec![vec![
            AlignedBlock::new([1.0; 16]),
            AlignedBlock::new([2.0; 16]),
        ]];
        let bs = vec![0.0];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q = vec![AlignedBlock::new([1.0; 16]), AlignedBlock::new([1.0; 16])];

        // dot product = 16*1*1 + 16*1*2 = 16 + 32 = 48
        let hash_value = hasher.hash_one(&q, 0);
        assert_eq!(hash_value, 48.0);

        // The final hash should be non-zero
        let hash = hasher.hash(&q);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_hash_with_many_hash_functions() {
        // Test with maximum number of hash functions that fit in u128
        // u128 has 128 bits, so we can have up to 128 hash functions (1 bit each)
        let num_hashes = 64; // Use 64 for reasonable test
        let a_vectors = vec![vec![AlignedBlock::new([1.0; 16])]; num_hashes];
        let bs = vec![0.0; num_hashes];
        let hasher = PStableHashingBlock::new(a_vectors, bs);

        let q = vec![AlignedBlock::new([1.0; 16])];
        let hash = hasher.hash(&q);

        // Should compute without panicking
        assert_ne!(hash, 0);
    }
}
