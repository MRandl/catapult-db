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
}
