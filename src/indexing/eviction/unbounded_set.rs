use crate::indexing::eviction::neighbors::{EvictionNeighborSet, NeighborSet};

pub struct UnboundedNeighborSet {
    neighbors: Vec<usize>,
}

impl UnboundedNeighborSet {
    pub fn from(value: Vec<usize>) -> Self {
        UnboundedNeighborSet { neighbors: value }
    }
}

impl NeighborSet for UnboundedNeighborSet {
    fn as_slice(&self) -> Vec<usize> {
        self.neighbors.clone()
    }
}

impl EvictionNeighborSet for UnboundedNeighborSet {
    fn insert(&mut self, neighbor: usize) {
        self.neighbors.push(neighbor);
    }
}
