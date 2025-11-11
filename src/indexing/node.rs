use std::{fmt::Debug, sync::RwLock};

use crate::indexing::eviction::{FixedSet, neighbor_set::CatapultNeighborSet};

pub struct Node<T: CatapultNeighborSet> {
    pub neighbors: FixedSet,
    pub catapults: RwLock<T>,
    pub payload: Box<[f32]>,
}

impl<T: CatapultNeighborSet + Debug> Debug for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("neighbors", &self.neighbors)
            .field("catapults", &self.catapults.read().unwrap())
            .field("payload", &self.payload)
            .finish()
    }
}
