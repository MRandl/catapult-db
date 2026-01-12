use crate::{
    numerics::{AlignedBlock, VectorLike},
    search::{
        graph_algo::{FlatCatapultChoice, FlatSearch, GraphSearchAlgorithm},
        hash_start::EngineStarter,
        node::Node,
    },
    sets::{
        candidates::{CandidateEntry, SmallestKCandidates},
        catapults::CatapultEvictingStructure,
        fixed::{FixedSet, FlatFixedSet},
        visited::{CompressedBitset, VisitorSet},
    },
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
pub struct AdjacencyGraph<EvictPolicy, Algo>
where
    EvictPolicy: CatapultEvictingStructure,
    Algo: GraphSearchAlgorithm,
{
    adjacency: Vec<Node<Algo::FixedSetType>>,
    starter: Algo::StartingPointSelector<EvictPolicy>,
    catapults: Algo::CatapultChoice,
}

impl<EvictPolicy> AdjacencyGraph<EvictPolicy, FlatSearch>
where
    EvictPolicy: CatapultEvictingStructure,
{
    pub fn new_flat(
        adj: Vec<Node<FlatFixedSet>>,
        engine: EngineStarter<EvictPolicy>,
        catapults: FlatCatapultChoice,
    ) -> Self {
        Self {
            adjacency: adj,
            starter: engine,
            catapults,
        }
    }
}

impl<EvictPolicy, SearchAlgo> AdjacencyGraph<EvictPolicy, SearchAlgo>
where
    EvictPolicy: CatapultEvictingStructure,
    SearchAlgo: GraphSearchAlgorithm,
{
    fn distances_from_indices(
        &self,
        indices: &[usize],
        query: &[AlignedBlock],
        catapult_marker: bool,
    ) -> Vec<CandidateEntry> {
        indices
            .iter()
            .map(|&index| {
                let starting_point = &self.adjacency[index];
                let starting_score = starting_point.payload.l2_squared(query);

                CandidateEntry {
                    distance: starting_score.into(),
                    index,
                    has_catapult_ancestor: catapult_marker,
                }
            })
            .collect()
    }

    /// Performs a best-first beam search from an LSH-selected set of entry points.
    ///
    /// The method maintains a candidate set of size at most `beam_width`
    /// containing the closest seen nodes (by L2). At each step it
    /// expands the closest not-yet-visited node from that set, inserts all of
    /// its neighbors into the candidate set (subject to the beam capacity),
    /// marks the expanded node visited, and repeats until no unvisited
    /// candidates remain.
    ///
    ///
    /// # Parameters
    /// - `query`: Target vector.
    /// - `k`: Number of final nearest neighbors desired. **Note:** currently only
    ///   used for the assertion `beam_width >= k` (see "Issues" below).
    /// - `beam_width`: Maximum size of the maintained candidate set.
    /// - `level`: The HNSW layer to use, if any. Should be None for non-HNSW searches.
    ///
    /// # Returns
    /// The `k` indices (and their corresponding distances) that are closest
    /// to the given query according to the approximate beam-search algorithm
    ///
    /// # Panics
    /// - If `beam_width < k`.
    /// - If neighbor indices are out of bounds (violates the graph invariant).
    fn beam_search_raw(
        &self,
        query: &[AlignedBlock],
        starting_candidates: &[CandidateEntry],
        k: usize,
        beam_width: usize,
        level: <SearchAlgo::FixedSetType as FixedSet>::LevelContext,
        stats: &mut crate::statistics::Stats,
    ) -> Vec<CandidateEntry> {
        assert!(beam_width >= k);
        stats.bump_beam_calls();

        let mut candidates: SmallestKCandidates = SmallestKCandidates::new(beam_width);
        let mut visited = CompressedBitset::new();

        candidates.insert_batch(starting_candidates);

        // among the suggested entry points, one of them is 'the best'. Let's identify it.
        let initial_best_node = candidates.iter().min().copied().expect(
            "Corrupted starting point entries. Provide a non-empty starting_indices lists.",
        );

        let mut best_candidate: Option<CandidateEntry> = Some(initial_best_node);

        // while we have some node on which to expand (at first, the best LSH entry point),
        // we keep expanding it (i.e. looking at its neighbors for better guesses)
        while let Some(best_candidate_node) = best_candidate {
            let best_candidate_neighs = &self.adjacency[best_candidate_node.index].neighbors;
            // identify the neighbors and catapult landing points of our current best guess.
            // All of these guys become candidates for expansion. if we have too many candidates
            // (beam width parameter), the `candidates` data structure takes care of removing the
            // worst ones (and the duplicates).
            let neighbors = &best_candidate_neighs.to_level(level);
            let neighbor_distances = self.distances_from_indices(
                neighbors,
                query,
                best_candidate_node.has_catapult_ancestor,
            );
            stats.bump_computed_dists(neighbors.len());
            candidates.insert_batch(&neighbor_distances);

            // mark our current node as visited (not to be expanded again)
            visited.set(best_candidate_node.index);
            stats.bump_nodes_visited();

            // and find some other guy to expand, if possible. If not, we call it a day and return our best guesses.
            best_candidate = candidates
                .iter()
                .filter(|&elem| !visited.get(elem.index))
                .min()
                .copied()
        }

        // we have beam_width neighbors, we only need k so we need to rerank
        let mut candidate_vec = candidates.into_iter().collect::<Vec<_>>();
        candidate_vec.sort(); // note: implicitly relying on CandidateEntry ordering here

        // and return the best k, job done :)
        candidate_vec.into_iter().take(k).collect()
    }
}

impl<EvictPolicy> AdjacencyGraph<EvictPolicy, FlatSearch>
where
    EvictPolicy: CatapultEvictingStructure,
{
    pub fn beam_search(
        &self,
        query: &[AlignedBlock],
        k: usize,
        beam_width: usize,
        stats: &mut crate::statistics::Stats,
    ) -> Vec<CandidateEntry> {
        let hash_search = self.starter.select_starting_points(query);
        let signature = hash_search.signature;
        let entry_points = hash_search.start_points;

        let mut distances = self.distances_from_indices(&entry_points, query, true);
        let last_index = distances.len() - 1;
        distances[last_index].has_catapult_ancestor = false;

        let search_results = self.beam_search_raw(query, &distances, k, beam_width, (), stats);
        let best_result = search_results[0].index;

        if self.catapults.local_enabled() {
            self.starter.new_catapult(signature, best_result);
            if search_results.iter().any(|e| e.has_catapult_ancestor) {
                stats.bump_searches_with_catapults();
            }
        }
        search_results
    }

    pub fn clear_all_catapults(&self) {
        self.starter.clear_all_catapults();
    }

    #[allow(clippy::len_without_is_empty)] // checking the size of the graph is fine, but it is never ever empty
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        numerics::SIMD_LANECOUNT,
        search::{
            graph_algo::{FlatCatapultChoice, FlatSearch},
            hash_start::{EngineStarter, EngineStarterParams},
        },
        sets::{catapults::FifoSet, fixed::FlatFixedSet},
    };

    pub type TestEngineStarter = EngineStarter<FifoSet<30>>;

    use super::*;

    // A simple graph:
    // Nodes: 0 (pos 0), 1 (pos 10), 2 (pos 20), 3 (pos 30), 4 (pos 40)
    // Edges: 0 -> 1 -> 2 -> 3 -> 4
    // Query: 11.0
    fn setup_simple_graph(catapults_enabled: bool) -> AdjacencyGraph<FifoSet<30>, FlatSearch> {
        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
            // 1: Pos 10.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 2: Pos 20.0, Neighbors: [1, 3]
            Node {
                payload: vec![AlignedBlock::new([20.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1, 3]),
            },
            // 3: Pos 30.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([30.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
            },
            // 4: Pos 40.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([40.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
        ];
        let catapult_choice = if catapults_enabled {
            FlatCatapultChoice::CatapultsEnabled
        } else {
            FlatCatapultChoice::CatapultsDisabled
        };
        // Start from node 0
        let params = EngineStarterParams::new(4, SIMD_LANECOUNT, 0, 42, catapults_enabled);
        AdjacencyGraph::new_flat(nodes, EngineStarter::new(params), catapult_choice)
    }

    #[test]
    fn test_basic_search_path() {
        let graph = setup_simple_graph(true);
        let query = vec![AlignedBlock::new([11.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;

        // Distances: 0(121), 1(1), 2(81), 3(361), 4(841)
        let mut stats = crate::statistics::Stats::new();

        let results1 = graph.beam_search(&query, k, beam_width, &mut stats);
        let results2 = graph.beam_search(&query, k, beam_width, &mut stats);

        assert_eq!(stats.get_searches_with_catapults(), 1);

        assert_eq!(
            results1.iter().map(|e| e.index).collect::<Vec<_>>(),
            results2.iter().map(|e| e.index).collect::<Vec<_>>()
        );

        // Final candidates (in order of distance): 1 (1), 2 (81), 0 (121)
        // Top K=2 results: [1, 2]
        assert_eq!(results1.len(), k);
        assert_eq!(results1[0].index, 1);
        assert_eq!(results1[1].index, 2);

        // Verify the graph structure - count total edges
        let total_edges: usize = graph
            .adjacency
            .iter()
            .map(|n| n.neighbors.to_level(()).len())
            .sum();
        assert_eq!(total_edges, 6);
    }

    #[test]
    fn test_first_query_same_results_with_and_without_catapults() {
        // Test that the first query returns identical results regardless of catapult setting
        // This is expected because catapults only affect subsequent queries after being established
        let graph_with_catapults = setup_simple_graph(true);
        let graph_without_catapults = setup_simple_graph(false);

        let query = vec![AlignedBlock::new([11.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;

        let mut stats_with = crate::statistics::Stats::new();
        let mut stats_without = crate::statistics::Stats::new();

        let results_with = {
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with);
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with); // only this one has access to the catapult
            graph_with_catapults.clear_all_catapults();
            graph_with_catapults.beam_search(&query, k, beam_width, &mut stats_with)
        };
        let results_without =
            graph_without_catapults.beam_search(&query, k, beam_width, &mut stats_without);

        // Both should return the same indices
        assert_eq!(
            results_with.iter().map(|e| e.index).collect::<Vec<_>>(),
            results_without.iter().map(|e| e.index).collect::<Vec<_>>(),
            "First query should return same results with and without catapults"
        );

        // Both should return the same distances
        assert_eq!(
            results_with.iter().map(|e| e.distance).collect::<Vec<_>>(),
            results_without
                .iter()
                .map(|e| e.distance)
                .collect::<Vec<_>>(),
            "First query should return same distances with and without catapults"
        );

        // Graph without catapults should never use catapults
        assert_eq!(stats_without.get_searches_with_catapults(), 0);
        assert_eq!(stats_with.get_searches_with_catapults(), 1);
    }

    #[test]
    fn test_multiple_starting_points() {
        let nodes = vec![
            // 0: Pos 100.0, Dist 10000
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 1: Pos 0.0, Dist 1. (BEST of all)
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![0]),
            },
            // 2: Pos 5.0, Dist 36. (Worst starting point)
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
            },
        ];
        // Start points: 0, 2
        let params = EngineStarterParams::new(4, SIMD_LANECOUNT, 0, 42, true);
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(params),
            FlatCatapultChoice::CatapultsEnabled,
        );
        let query = vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);
        // Top K=2 results: [1, 2]
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 1);
        assert_eq!(results[1].index, 2);
    }

    #[test]
    fn test_complex_search_divergence() {
        // Query target: [0.0]
        let nodes: Vec<Node<FlatFixedSet>> = vec![
            // 0: Pos 10.0, Dist 100. N: [1, 5]. (Start point)
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1, 5]),
            },
            // 1: Pos 8.0, Dist 64. N: [2].
            Node {
                payload: vec![AlignedBlock::new([8.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2]),
            },
            // 2: Pos 5.0, Dist 25. N: [3].
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![3]),
            },
            // 3: Pos 2.0, Dist 4. N: [4].
            Node {
                payload: vec![AlignedBlock::new([2.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
            },
            // 4: Pos 1.0, Dist 1. N: []. (The globally BEST node)
            Node {
                payload: vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![]),
            },
            // 5: Pos 100.0, Dist 10000. N: []. (A distant dead end)
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![]),
            },
        ];
        let params = EngineStarterParams::new(4, SIMD_LANECOUNT, 1, 42, true);
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(params),
            FlatCatapultChoice::CatapultsEnabled,
        );
        let query = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        let k = 1;
        let beam_width = 2; // Tight beam width forces early pruning
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);

        // The globally best result (Node 4) must be found and returned.
        assert_eq!(results.len(), k);
        assert_eq!(results[0].index, 4);
    }
}
