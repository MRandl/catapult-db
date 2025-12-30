use itertools::Itertools;

use crate::{
    candidates::{CandidateEntry, CompressedBitset, SmallestK, VisitorSet},
    indexing::{
        engine_starter::EngineStarter, eviction::catapult_neighbor_set::CatapultNeighborSet,
        node::Node,
    },
    numerics::{AlignedBlock, VectorLike},
};

/// In-memory adjacency graph used for approximate nearest-neighbor (ANN) search.
///
/// # Invariants
/// - `adjacency[i]` represents node `i`.
/// - Each `Node.neighbors` entry is a valid index into `adjacency`.
/// - `Node.payload` implements `VectorLike` and supports `l2_squared(&[f32])`.
///
/// # Algorithms
/// - [`beam_search`] implements a *best-first beam search*: it keeps only the
///   `beam_width` closest candidates seen so far and repeatedly expands the best
///   not-yet-visited candidate until no such candidate remains.
pub struct AdjacencyGraph<T: CatapultNeighborSet> {
    adjacency: Vec<Node<T>>,
    starter: EngineStarter,
    catapults: bool,
}

impl<T: CatapultNeighborSet> AdjacencyGraph<T> {
    pub fn new(adj: Vec<Node<T>>, engine: EngineStarter, catapults: bool) -> Self {
        Self {
            adjacency: adj,
            starter: engine,
            catapults,
        }
    }

    /// Performs a best-first beam search from a single entry point.
    ///
    /// The method maintains a candidate set of size at most `beam_width`
    /// containing the closest seen nodes (by squared L2). At each step it
    /// expands the closest not-yet-visited node from that set, inserts all of
    /// its neighbors into the candidate set (subject to the beam capacity),
    /// marks the expanded node visited, and repeats until no unvisited
    /// candidates remain.
    ///
    /// # Parameters
    /// - `query`: Target vector.
    /// - `k`: Number of final nearest neighbors desired. **Note:** currently only
    ///   used for the assertion `beam_width >= k` (see “Issues” below).
    /// - `beam_width`: Maximum size of the maintained candidate set.
    ///
    /// # Returns
    /// The `k` indices (and their corresponding distances) that are closest
    /// to the given query according to the approximate beam-search algorithm
    ///
    /// # Panics
    /// - If `beam_width < k`.
    /// - If neighbor indices are out of bounds (violates the graph invariant).
    pub fn beam_search(
        &self,
        query: &[AlignedBlock],
        k: usize,
        beam_width: usize,
        level: Option<u32>,
    ) -> Vec<CandidateEntry> {
        assert!(beam_width >= k);

        let mut candidates: SmallestK = SmallestK::new(beam_width);
        let mut visited = CompressedBitset::new();

        // find a few starting points for the beam call, using LSH
        let starting_indices = self.starter.select_starting_points(query, beam_width);
        for starting_index in starting_indices {
            let starting_point = &self.adjacency[starting_index];
            let starting_score = starting_point.payload.l2_squared(query);

            candidates.insert(CandidateEntry {
                distance: starting_score.into(),
                index: starting_index,
            });
        }

        let initial_best_node = candidates
            .iter()
            .min()
            .map(|entry| entry.index)
            .expect("Corrupted LSH entries. Please provide a non-degenerate hashing engine, or just use its default constructor.");

        let mut best_index_in_candidates: Option<usize> = Some(initial_best_node);

        while let Some(best_candidate_index) = best_index_in_candidates {
            let best_candidate_node = &self.adjacency[best_candidate_index];

            let candidate_regular_neighbors = &best_candidate_node.neighbors;
            let candidate_catapults = &best_candidate_node.catapults.read().unwrap();

            for &neighbor in candidate_regular_neighbors
                .to_box(level)
                .iter()
                .chain(&candidate_catapults.to_vec())
                .unique()
            {
                let neighbor_node = &self.adjacency[neighbor];
                let neighbor_distance = neighbor_node.payload.l2_squared(query);
                candidates.insert(CandidateEntry {
                    distance: neighbor_distance.into(),
                    index: neighbor,
                });
            }

            visited.set(best_candidate_index);
            best_index_in_candidates = candidates
                .iter()
                .filter(|&elem| !visited.get(elem.index))
                .min()
                .map(|e| e.index)
        }

        let mut candidate_vec = candidates.into_iter().collect::<Vec<_>>();
        candidate_vec.sort();

        if self.catapults {
            self.adjacency[initial_best_node]
                .catapults
                .write()
                .unwrap()
                .insert(candidate_vec[0].index);
        }

        candidate_vec.into_iter().take(k).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::indexing::eviction::{FixedSet, unbounded_set::UnboundedNeighborSet};

    use super::*;
    use std::sync::RwLock;

    // A simple graph:
    // Nodes: 0 (pos 0), 1 (pos 10), 2 (pos 20), 3 (pos 30), 4 (pos 40)
    // Edges: 0 -> 1 -> 2 -> 3 -> 4
    // Query: 11.0
    fn setup_simple_graph() -> AdjacencyGraph<UnboundedNeighborSet> {
        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![1]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 10.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([10.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![2]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 20.0, Neighbors: [1, 3]
            Node {
                payload: vec![AlignedBlock::new([20.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![1, 3]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 30.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([30.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![4]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 40.0, Neighbors: []
            Node {
                payload: vec![AlignedBlock::new([40.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];
        // Start from node 0
        AdjacencyGraph::new(nodes, EngineStarter::new(4, 8, 4, Some(42)), true)
    }

    #[test]
    fn test_basic_search_path() {
        let graph = setup_simple_graph();
        let query = vec![AlignedBlock::new([11.0; 8])];
        let k = 2;
        let beam_width = 3;

        // Distances: 0(121), 1(1), 2(81), 3(361), 4(841)

        let results = graph.beam_search(&query, k, beam_width, None);

        // Final candidates (in order of distance): 1 (1), 2 (81), 0 (121)
        // Top K=2 results: [1, 2]
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 1);
        assert_eq!(results[0].distance, (8.0).into());
        assert_eq!(results[1].index, 2);
        assert_eq!(results[1].distance, (648.0).into());

        // Expanded: 0 (1 edge) + 1 (1 edge) + 2 (2 edges) + 0 (1 edge) = 5 edges
        assert_eq!(
            graph
                .adjacency
                .iter()
                .map(|n| {
                    n.catapults.read().unwrap().to_vec().len() + n.neighbors.to_box(None).len()
                })
                .sum::<usize>(),
            6
        );
        assert_eq!(
            graph
                .adjacency
                .iter()
                .map(|n| { n.neighbors.to_box(None).len() })
                .sum::<usize>(),
            5
        );
    }

    #[test]
    fn test_multiple_starting_points() {
        let nodes = vec![
            // 0: Pos 100.0, Dist 10000
            Node {
                payload: vec![AlignedBlock::new([100.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![2]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 0.0, Dist 1. (BEST of all)
            Node {
                payload: vec![AlignedBlock::new([0.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![0]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 5.0, Dist 36. (Worst starting point)
            Node {
                payload: vec![AlignedBlock::new([5.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![1]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];
        // Start points: 0, 2
        let graph = AdjacencyGraph::new(nodes, EngineStarter::new(4, 8, 3, Some(42)), true);
        let query = vec![AlignedBlock::new([1.0; 8])];
        let k = 2;
        let beam_width = 3;

        let results = graph.beam_search(&query, k, beam_width, None);
        println!(
            "Top K={} results: {:?}",
            k,
            results.iter().map(|e| e.index).collect::<Vec<_>>()
        );
        // Top K=2 results: [1, 2]
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 1);
        assert_eq!(results[0].distance, 8.0.into());
        assert_eq!(results[1].index, 2);
        assert_eq!(results[1].distance, 128.0.into());
    }

    #[test]
    fn test_complex_search_divergence() {
        // Query target: [0.0]
        let nodes: Vec<Node<UnboundedNeighborSet>> = vec![
            // 0: Pos 10.0, Dist 100. N: [1, 5]. (Start point)
            Node {
                payload: vec![AlignedBlock::new([10.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![1, 5]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 8.0, Dist 64. N: [2].
            Node {
                payload: vec![AlignedBlock::new([8.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![2]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 5.0, Dist 25. N: [3].
            Node {
                payload: vec![AlignedBlock::new([5.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![3]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 2.0, Dist 4. N: [4].
            Node {
                payload: vec![AlignedBlock::new([2.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![4]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 1.0, Dist 1. N: []. (The globally BEST node)
            Node {
                payload: vec![AlignedBlock::new([1.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 5: Pos 100.0, Dist 10000. N: []. (A distant dead end)
            Node {
                payload: vec![AlignedBlock::new([100.0; 8])].into_boxed_slice(),
                neighbors: FixedSet::new(vec![]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];
        let graph = AdjacencyGraph::new(nodes, EngineStarter::new(4, 8, 5, Some(42)), true);
        let query = vec![AlignedBlock::new([0.0; 8])];
        let k = 1;
        let beam_width = 2; // Tight beam width forces early pruning

        let results = graph.beam_search(&query, k, beam_width, None);

        // The globally best result (Node 4) must be found and returned.
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 4);
        assert_eq!(results[0].distance, 8.0.into());

        // Check catapult update: Initial node points to Node 4 (final best)
        assert!(
            graph
                .adjacency
                .iter()
                .map(|n| n.catapults.read().unwrap())
                .all(|cats| cats.to_vec().is_empty() || cats.to_vec() == vec![4])
        );
    }
}
