use crate::indexing::eviction::neighbor_set::NeighborSet;

pub struct FixedSet {
    neighbors: Vec<usize>,
}

impl FixedSet {
    pub fn new(initial_values: Vec<usize>) -> Self {
        FixedSet {
            neighbors: initial_values,
        }
    }
}

impl NeighborSet for FixedSet {
    fn as_slice(&self) -> Vec<usize> {
        self.neighbors.clone()
    }
}
