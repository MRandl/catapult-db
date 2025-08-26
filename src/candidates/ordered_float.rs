use std::cmp::Ordering;

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct TotalF32(pub f32);

impl PartialEq for TotalF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for TotalF32 {}

impl PartialOrd for TotalF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.total_cmp(&other.0))
    }
}

impl Ord for TotalF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Allow implicit promotion from f32 → TotalF32
impl From<f32> for TotalF32 {
    fn from(x: f32) -> Self {
        TotalF32(x)
    }
}
