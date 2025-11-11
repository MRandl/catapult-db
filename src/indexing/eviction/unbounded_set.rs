use crate::indexing::eviction::neighbor_set::EvictionNeighborSet;

pub struct UnboundedNeighborSet {
    neighbors: Vec<usize>,
}

impl UnboundedNeighborSet {
    pub fn from(value: Vec<usize>) -> Self {
        UnboundedNeighborSet { neighbors: value }
    }
}
impl EvictionNeighborSet for UnboundedNeighborSet {
    fn to_vec(&self) -> Vec<usize> {
        self.neighbors.clone()
    }

    fn insert(&mut self, neighbor: usize) {
        self.neighbors.push(neighbor);
    }

    fn new() -> Self {
        UnboundedNeighborSet { neighbors: vec![] }
    }
}
