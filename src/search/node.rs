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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::SIMD_LANECOUNT;
    use crate::search::graph_hierarchy::FlatSearch;
    use crate::sets::catapults::UnboundedNeighborSet;

    #[test]
    fn test_node_debug_format_basic() {
        let node = Node {
            payload: vec![AlignedBlock::new([1.5; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FixedSet::<FlatSearch>::new(vec![1, 2, 3]),
            catapults: RwLock::new(UnboundedNeighborSet::new()),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3] }}, catapults: UnboundedNeighborSet {{ neighbors: [] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [1.5; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_with_catapults() {
        let mut catapults = UnboundedNeighborSet::new();
        catapults.insert(10);
        catapults.insert(20);

        let node = Node {
            payload: vec![AlignedBlock::new([2.0; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FixedSet::<FlatSearch>::new(vec![5]),
            catapults: RwLock::new(catapults),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [5] }}, catapults: UnboundedNeighborSet {{ neighbors: [10, 20] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [2.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_empty_neighbors() {
        let node = Node {
            payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FixedSet::<FlatSearch>::new(vec![]),
            catapults: RwLock::new(UnboundedNeighborSet::new()),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [] }}, catapults: UnboundedNeighborSet {{ neighbors: [] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [0.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_multiple_payload_blocks() {
        let node = Node {
            payload: vec![
                AlignedBlock::new([1.0; SIMD_LANECOUNT]),
                AlignedBlock::new([2.0; SIMD_LANECOUNT]),
                AlignedBlock::new([3.0; SIMD_LANECOUNT]),
            ]
            .into_boxed_slice(),
            neighbors: FixedSet::<FlatSearch>::new(vec![1, 2, 3, 4, 5]),
            catapults: RwLock::new(UnboundedNeighborSet::new()),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3, 4, 5] }}, catapults: UnboundedNeighborSet {{ neighbors: [] }}, payload: [AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}] }}",
            [1.0; SIMD_LANECOUNT], [2.0; SIMD_LANECOUNT], [3.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }
}
