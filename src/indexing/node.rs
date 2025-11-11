use std::{fmt::Debug, sync::RwLock};

use crate::indexing::eviction::{FixedSet, neighbor_set::EvictionNeighborSet};

#[derive(Debug)]
pub struct Node<T: EvictionNeighborSet> {
    pub neighbors: FixedSet,
    pub catapults: RwLock<T>,
    pub payload: Box<[f32]>,
}
