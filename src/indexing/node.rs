use std::{fmt::Debug, sync::RwLock};

use crate::indexing::eviction::{FixedSet, neighbor_set::CatapultNeighborSet};

#[derive(Debug)]
pub struct Node<T: CatapultNeighborSet> {
    pub neighbors: FixedSet,
    pub catapults: RwLock<T>,
    pub payload: Box<[f32]>,
}
