use crate::sets::candidates::TotalF32;
use std::hash::Hash;

/// A candidate node in a graph search, storing its distance from the query point
/// and metadata about how it was discovered.
///
/// Candidates are ordered primarily by distance (ascending) for use in priority queues
/// during beam search algorithms. Two candidates with the same distance and index are
/// considered equal regardless of their catapult ancestry.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct CandidateEntry {
    /// Distance from the query point to this candidate node.
    pub distance: TotalF32,

    /// Index of the candidate node in the graph.
    pub index: usize,

    /// Whether this candidate was discovered via a catapult (long-range cached connection)
    /// rather than through standard neighbor traversal.
    pub has_catapult_ancestor: bool,
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
