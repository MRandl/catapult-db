use crate::indexing::neighbor_set::NeighborSet;

pub struct Node {
    pub neighbors: NeighborSet,
    pub catapults: NeighborSet,
    pub payload: Box<[f32]>,
}
