/// Number of f32 elements per SIMD lane for parallel operations.
///
/// This value is set to 16 to match common SIMD architectures and ensure
/// efficient vectorized computations. The alignment requirement in `AlignedBlock`
/// must be kept in sync with this value (64 bytes = 16 f32s * 4 bytes).
pub const SIMD_LANECOUNT: usize = 16;

/// A 64-byte aligned block of 16 f32 values optimized for SIMD operations.
///
/// This structure ensures cache-line friendly memory layout and enables efficient
/// parallel distance computations using SIMD instructions. The 64-byte alignment
/// corresponds to `SIMD_LANECOUNT * size_of::<f32>()` for optimal performance.
#[repr(align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedBlock {
    /// Array of 16 f32 values representing a portion of a vector.
    pub data: [f32; SIMD_LANECOUNT],
}

impl AlignedBlock {
    /// Creates a new aligned block from an array of f32 values.
    ///
    /// # Arguments
    /// * `data` - Array of exactly `SIMD_LANECOUNT` (16) f32 values
    ///
    /// # Returns
    /// A new `AlignedBlock` containing the provided data
    pub fn new(data: [f32; SIMD_LANECOUNT]) -> Self {
        AlignedBlock { data }
    }

    /// Converts a flat vector of f32 values into SIMD-aligned blocks with zero-padding.
    ///
    /// This function chunks the input vector into blocks of `SIMD_LANECOUNT` elements.
    /// If the input length is not a multiple of `SIMD_LANECOUNT`, the final block is
    /// zero-padded to complete the block.
    ///
    /// # Arguments
    /// * `data` - Vector of f32 values to convert into aligned blocks
    ///
    /// # Returns
    /// A vector of `AlignedBlock` instances. If the input length is an exact multiple
    /// of `SIMD_LANECOUNT`, no padding is added. Otherwise, the final block is zero-padded.
    pub fn allocate_padded(data: Vec<f32>) -> Vec<AlignedBlock> {
        let mut returned = Vec::with_capacity(data.len().div_ceil(SIMD_LANECOUNT));

        let (chunked, remainder) = data.as_chunks::<SIMD_LANECOUNT>();
        for &chunk in chunked.iter() {
            returned.push(Self::new(chunk));
        }

        // Only add a remainder block if there are leftover elements
        if !remainder.is_empty() {
            let mut remainder_data = [0.0; SIMD_LANECOUNT];
            for (i, b) in remainder.iter().enumerate() {
                remainder_data[i] = *b;
            }
            returned.push(Self::new(remainder_data));
        }

        returned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignedblock_is_aligned_on_simd() {
        assert_eq!(
            align_of::<AlignedBlock>(),
            size_of::<f32>() * SIMD_LANECOUNT
        );
        // if this fails, we are not in a good shape. the general rule of thumb is that
        // repr(align(XX)) at the top of this file should be set to 4*lanecount (4 being the size of a f32 in bytes)
        // to force an SIMD read to be at a legal / performance-friendly location.
        // If you change the lanecount, please update the alignment accordingly.
        //
        // This is not set automatically because repr(align()) requires an integer *literal* and
        // does not accept const-time expressions. Probably for a good reason, I don't judge, but
        // then we need to have a sanity check as a test.
    }

    #[test]
    fn test_allocate_padded() {
        // Test with exact multiple of SIMD_LANECOUNT - should not add extra padding
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let blocks = AlignedBlock::allocate_padded(data.clone());
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].data,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );

        // Test with remainder - should add one zero-padded block
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let blocks = AlignedBlock::allocate_padded(data);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].data[0..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(blocks[0].data[5..], [0.0; 11]);

        // Test with multiple full blocks - should not add extra padding
        let data = vec![1.0; 32]; // Exactly 2 blocks
        let blocks = AlignedBlock::allocate_padded(data);
        assert_eq!(blocks.len(), 2);

        // Test with multiple blocks plus remainder
        let data = vec![1.0; 20]; // 1 full block + 4 elements
        let blocks = AlignedBlock::allocate_padded(data);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].data[0..4], [1.0; 4]);
        assert_eq!(blocks[1].data[4..], [0.0; 12]);
    }
}
