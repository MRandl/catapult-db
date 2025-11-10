use std::sync::RwLock;

use crate::indexing::eviction::{FixedSet, neighbors::EvictionNeighborSet};

pub struct Node<T: EvictionNeighborSet> {
    pub neighbors: FixedSet,
    pub catapults: RwLock<T>,
    pub payload: Box<[f32]>,
}
