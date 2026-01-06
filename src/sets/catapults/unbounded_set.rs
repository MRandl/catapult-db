use crate::sets::catapults::CatapultEvictingStructure;

#[derive(Debug)]
pub struct UnboundedNeighborSet {
    neighbors: Vec<usize>,
}

impl CatapultEvictingStructure for UnboundedNeighborSet {
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
