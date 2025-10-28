use std::sync::RwLock;

use crate::indexing::neighbor_set::NeighborSet;

pub struct Node {
    pub neighbors: NeighborSet,
    pub catapults: RwLock<NeighborSet>,
    pub payload: Box<[f32]>,
}
