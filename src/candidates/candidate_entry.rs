use crate::candidates::ordered_float::TotalF32;
use std::hash::Hash;

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub struct CandidateEntry {
    pub distance: TotalF32,
    pub index: usize,
}

impl PartialOrd for CandidateEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandidateEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}
