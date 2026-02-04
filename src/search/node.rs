use std::fmt::Debug;

use crate::{numerics::AlignedBlock, sets::fixed::FixedSet};

// The ID of a Node (usize), wrapped in its own struct to avoid accidental misuse.
pub struct NodeId {
    internal: usize,
}

/// A node in the proximity graph, containing its vector data and neighbor connections.
///
/// Each node stores both its connectivity information (neighbors) and the actual
/// vector embedding (payload) as SIMD-aligned blocks for efficient distance computation.
pub struct Node<FixedSetType: FixedSet + Debug> {
    /// The immutable set of neighbor node indices.
    pub neighbors: FixedSetType,

    /// The vector embedding for this node, stored as SIMD-aligned blocks of f32 values
    /// for efficient parallel distance computations.
    pub payload: Box<[AlignedBlock]>,
}

impl<FixedSetType: FixedSet + Debug> Debug for Node<FixedSetType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("neighbors", &self.neighbors)
            .field("payload", &self.payload)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::SIMD_LANECOUNT;
    use crate::sets::fixed::FlatFixedSet;

    #[test]
    fn test_node_debug_format_basic() {
        let node = Node {
            payload: vec![AlignedBlock::new([1.5; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FlatFixedSet::new(vec![1, 2, 3]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
            [1.5; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }

    #[test]
    fn test_node_debug_format_empty_neighbors() {
        let node = Node {
            payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
            neighbors: FlatFixedSet::new(vec![]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [] }}, payload: [AlignedBlock {{ data: {:?} }}] }}",
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
            neighbors: FlatFixedSet::new(vec![1, 2, 3, 4, 5]),
        };

        let debug_output = format!("{node:?}");
        let expected = format!(
            "Node {{ neighbors: FixedSet {{ neighbors: [1, 2, 3, 4, 5] }}, payload: [AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}, AlignedBlock {{ data: {:?} }}] }}",
            [1.0; SIMD_LANECOUNT], [2.0; SIMD_LANECOUNT], [3.0; SIMD_LANECOUNT]
        );

        assert_eq!(debug_output, expected);
    }
}
