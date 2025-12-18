use crate::numerics::SIMD_LANECOUNT;

#[repr(align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedBlock {
    pub data: [f32; SIMD_LANECOUNT],
}

impl AlignedBlock {
    pub fn new(data: [f32; SIMD_LANECOUNT]) -> Self {
        AlignedBlock { data }
    }

    pub fn data(&self) -> &[f32; SIMD_LANECOUNT] {
        &self.data
    }
}
