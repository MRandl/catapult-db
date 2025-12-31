use std::{fmt::Debug, sync::RwLock};

use crate::{
    indexing::{
        eviction::{FixedSet, catapult_neighbor_set::CatapultNeighborSet},
        graph_hierarchy::GraphSearchAlgo,
    },
    numerics::AlignedBlock,
};

pub struct Node<CatapultNeighbors, GraphSearchType>
where
    CatapultNeighbors: CatapultNeighborSet,
    GraphSearchType: GraphSearchAlgo,
{
    pub neighbors: FixedSet<GraphSearchType>,
    pub catapults: RwLock<CatapultNeighbors>,
    pub payload: Box<[AlignedBlock]>,
}

impl<T: CatapultNeighborSet + Debug, GraphSearchType: GraphSearchAlgo> Debug
    for Node<T, GraphSearchType>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("neighbors", &self.neighbors)
            .field("catapults", &self.catapults.read().unwrap())
            .field("payload", &self.payload)
            .finish()
    }
}
