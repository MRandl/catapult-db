use crate::sets::catapults::CatapultNeighborSet;

pub struct UnboundedNeighborSet {
    neighbors: Vec<usize>,
}

impl UnboundedNeighborSet {
    pub fn new() -> Self {
        UnboundedNeighborSet { neighbors: vec![] }
    }
}

impl Default for UnboundedNeighborSet {
    fn default() -> Self {
        Self::new()
    }
}

impl CatapultNeighborSet for UnboundedNeighborSet {
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
