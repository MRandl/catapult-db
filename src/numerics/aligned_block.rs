pub const SIMD_LANECOUNT: usize = 16;

#[repr(align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedBlock {
    pub data: [f32; SIMD_LANECOUNT],
}

impl AlignedBlock {
    pub fn new(data: [f32; SIMD_LANECOUNT]) -> Self {
        AlignedBlock { data }
    }

    pub fn allocate_padded(data: Vec<f32>) -> Vec<AlignedBlock> {
        let mut returned = Vec::with_capacity(data.len().div_ceil(SIMD_LANECOUNT));

        let (chunked, remainder) = data.as_chunks::<SIMD_LANECOUNT>();
        for &chunk in chunked.iter() {
            returned.push(Self::new(chunk));
        }

        let mut remainder_data = [0.0; SIMD_LANECOUNT];
        for (i, b) in remainder.iter().enumerate() {
            remainder_data[i] = *b;
        }
        returned.push(Self::new(remainder_data));
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
        // Test with exact multiple of SIMD_LANECOUNT
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let blocks = AlignedBlock::allocate_padded(data.clone());
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0].data,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
        assert_eq!(blocks[1].data, [0.0; SIMD_LANECOUNT]);

        // Test with remainder
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let blocks = AlignedBlock::allocate_padded(data);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].data[0..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(blocks[0].data[5..], [0.0; 11]);
    }
}
