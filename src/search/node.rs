use std::{fmt::Debug, sync::RwLock};

use crate::{
    numerics::AlignedBlock, search::graph_hierarchy::GraphSearchAlgorithm,
    sets::catapults::CatapultEvictingStructure, sets::catapults::FixedSet,
};

pub struct Node<CatapultNeighbors, GraphSearchType> {
    pub neighbors: FixedSet<GraphSearchType>,
    pub catapults: RwLock<CatapultNeighbors>,
    pub payload: Box<[AlignedBlock]>,
}

impl<T: CatapultEvictingStructure + Debug, GraphSearchType: GraphSearchAlgorithm> Debug
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
