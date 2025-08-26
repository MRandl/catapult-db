use crate::candidates::ordered_float::TotalF32;

#[derive(PartialEq, Eq)]
pub struct CandidateEntry {
    pub distance: TotalF32,
    pub index: usize,
}

impl PartialOrd for CandidateEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for CandidateEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}
