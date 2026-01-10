use crate::{
    numerics::{AlignedBlock, VectorLike},
    search::{
        graph_hierarchy::{FlatCatapultChoice, FlatSearch, GraphSearchAlgorithm},
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
    _catapults: Algo::CatapultChoice,
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
            _catapults: catapults,
        }
    }
}

// impl<EvictPolicy> AdjacencyGraph<EvictPolicy, HNSWSearch>
// where
//     EvictPolicy: CatapultEvictingStructure,
// {
//     pub fn new_hnsw(
//         adj: Vec<Node<HierarchicalFixedSet>>,
//         engine: HNSWEngineStarter,
//         catapults: HNSWCatapultChoice,
//     ) -> Self {
//         Self {
//             adjacency: adj,
//             starter: engine,
//             catapults,
//         }
//     }
// }

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

        if FlatSearch::local_catapults_enabled(self._catapults) {
            self.starter.new_catapult(signature, best_result);
            if search_results.iter().any(|e| e.has_catapult_ancestor) {
                stats.bump_searches_with_catapults();
            }
        }
        search_results
    }
}

/*impl<EvictPolicy> AdjacencyGraph<EvictPolicy, HNSWSearch>
where
    EvictPolicy: CatapultEvictingStructure,
{
    pub fn beam_search(
        &self,
        query: &[AlignedBlock],
        k: usize,
        beam_width: usize,
        stats: &mut crate::statistics::Stats,
    ) -> Vec<usize> {
        let starting_points = self.starter.select_starting_points(query, k);
        // if there is an interesting catapult in the starting set, we immediately start
        // at the bottom level (only in catapult finalizing mode, see graph_hierarchy.rs)
        let found_neighbors = if self.catapults == HNSWCatapultChoice::FinalizingCatapults
            && let Some(catapulted_zero_neighbors) =
                self.try_catapulting_beam(query, k, beam_width, &starting_points, stats)
        {
            // we found a somewhat promising catapult to level 0, so we search level 0 directly
            catapulted_zero_neighbors
        } else {
            let mut curr_points = starting_points.clone();
            for curr_level in (0_usize..=self.starter.max_level).rev() {
                // note: curr_points maintains a size of k (<= beam_width) even though two consecutive
                // beam search calls work with beam_width elements, which may be suboptimal.
                // todo check the original HNSW paper for how they tackle this. I'm probably paranoid.
                curr_points =
                    self.beam_search_raw(query, &curr_points, k, beam_width, curr_level, stats)
            }
            curr_points
        };

        black_box(1);
        //todo : catapults insertions

        found_neighbors
    }

    fn try_catapulting_beam(
        &self,
        query: &[AlignedBlock],
        k: usize,
        beam_width: usize,
        curr_points: &[usize],
        stats: &mut crate::statistics::Stats,
    ) -> Option<Vec<usize>> {
        let mut all_neighbors = Vec::new();
        for starting_point in curr_points.iter() {
            let starting_neighs = self.adjacency[*starting_point]
                .neighbors
                .to_level(self.starter.max_level);
            all_neighbors.extend_from_slice(&starting_neighs);
        }
        all_neighbors.sort_unstable();
        all_neighbors.dedup();

        let mut all_catapults = Vec::new();
        for starting_point in curr_points.iter() {
            let starting_catas = []; // todo fix
            all_catapults.extend_from_slice(&starting_catas);
        }
        all_catapults.sort_unstable();
        all_catapults.dedup();

        let best_neighbor = self
            .distances_from_indices(&all_neighbors, query)
            .into_iter()
            .min()
            .expect("Starting nodes are disconnected from the graph. That's pretty awkward.");
        //   ^^^^^^ crash if we end up in the situation where the graph is disconnected.
        // pretty sure the graph builders (DiskANN / FAISS) make sure this NEVER EVER happens, but
        // let's be sure and check here.

        let best_catapult = self
            .distances_from_indices(&all_catapults, query)
            .into_iter()
            .min();

        // if we have a catapult that is interesting, trigger it and start at level zero.
        if let Some(best_catapult) = best_catapult
            && best_catapult < best_neighbor
        {
            let level_zero_starter = [best_catapult.index]; // one-man starter. do your best, little man
            Some(self.beam_search_raw(query, &level_zero_starter, k, beam_width, 0, stats))
        } else {
            None
        }
    }
}*/

#[cfg(test)]
mod tests {
    use crate::{
        numerics::SIMD_LANECOUNT,
        search::{
            graph_hierarchy::{FlatCatapultChoice, FlatSearch},
            hash_start::EngineStarter,
        },
        sets::{catapults::FifoSet, fixed::FlatFixedSet},
    };

    pub type TestEngineStarter = EngineStarter<FifoSet<30>>;

    use super::*;

    // A simple graph:
    // Nodes: 0 (pos 0), 1 (pos 10), 2 (pos 20), 3 (pos 30), 4 (pos 40)
    // Edges: 0 -> 1 -> 2 -> 3 -> 4
    // Query: 11.0
    fn setup_simple_graph() -> AdjacencyGraph<FifoSet<30>, FlatSearch> {
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
        // Start from node 0
        AdjacencyGraph::new_flat(
            nodes,
            EngineStarter::new(4, SIMD_LANECOUNT, 0, 42, true),
            FlatCatapultChoice::CatapultsEnabled,
        )
    }

    #[test]
    fn test_basic_search_path() {
        let graph = setup_simple_graph();
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
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(4, SIMD_LANECOUNT, 0, 42, true),
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
        let graph = AdjacencyGraph::new_flat(
            nodes,
            TestEngineStarter::new(4, SIMD_LANECOUNT, 1, 42, true),
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

    /*
    // HNSW Tests - Commented out

    fn setup_hnsw_graph() -> AdjacencyGraph<UnboundedNeighborSet, HNSWSearch> {
        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 10.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![2]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 20.0, Neighbors: [1, 3]
            Node {
                payload: vec![AlignedBlock::new([20.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1, 3]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 30.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([30.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![4]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 40.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([40.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];

        AdjacencyGraph::new_hnsw(
            nodes,
            HNSWEngineStarter::new(EngineStarter::new(4, SIMD_LANECOUNT, 5, Some(42)), 0),
            HNSWCatapultChoice::CatapultsDisabled,
        )
    }

    #[test]
    fn test_hnsw_basic_search_path() {
        let graph = setup_hnsw_graph();
        let query = vec![AlignedBlock::new([11.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;
        let mut stats = crate::statistics::Stats::new();

        // Query at 11.0, distances: 0(121), 1(1), 2(81), 3(361), 4(841)
        let results = graph.beam_search(&query, k, beam_width, &mut stats);

        // HNSW should return exactly k results
        assert_eq!(results.len(), k);

        // All returned indices should be valid (within graph bounds)
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn test_hnsw_multiple_starting_points() {
        let nodes = vec![
            // 0: Pos 100.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![2]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 0.0, Neighbors: [0] (BEST overall)
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![0]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 5.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];

        let graph = AdjacencyGraph::new_hnsw(
            nodes,
            HNSWEngineStarter::new(EngineStarter::new(4, SIMD_LANECOUNT, 3, Some(42)), 0),
            HNSWCatapultChoice::CatapultsDisabled,
        );

        let query = vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);

        // Should find nodes 1 and 2 as the nearest
        assert_eq!(results.len(), k);
        assert_eq!(results[0], 1);
        assert_eq!(results[1], 2);
    }

    #[test]
    fn test_hnsw_complex_search_divergence() {
        // HNSW graph with a distant dead end
        let nodes: Vec<Node<UnboundedNeighborSet, HierarchicalFixedSet>> = vec![
            // 0: Pos 10.0, Neighbors: [1, 5]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1, 5]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 8.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([8.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![2]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 5.0, Neighbors: [3]
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![3]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 2.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([2.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![4]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 1.0, Neighbors: [] (globally BEST)
            Node {
                payload: vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 5: Pos 100.0, Neighbors: [] (distant dead end)
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];

        let graph = AdjacencyGraph::new_hnsw(
            nodes,
            HNSWEngineStarter::new(EngineStarter::new(4, SIMD_LANECOUNT, 6, Some(42)), 0),
            HNSWCatapultChoice::SameLevelCatapults,
        );

        let query = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        let k = 1;
        let beam_width = 2;
        let mut stats = crate::statistics::Stats::new();

        let results = graph.beam_search(&query, k, beam_width, &mut stats);

        // Should find node 4 as the best result
        assert_eq!(results.len(), k);
        assert_eq!(results[0], 4);
    }

    #[test]
    fn test_hnsw_finalizing_catapuls() {
        // HNSW graph with a distant dead end
        let nodes: Vec<Node<UnboundedNeighborSet, HierarchicalFixedSet>> = vec![
            // 0: Pos 10.0, Neighbors: [1, 5]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![1, 5]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 8.0, Neighbors: [2]
            Node {
                payload: vec![AlignedBlock::new([8.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![2]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 5.0, Neighbors: [3]
            Node {
                payload: vec![AlignedBlock::new([5.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![3]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 2.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([2.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![4]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 1.0, Neighbors: [5] (globally BEST)
            Node {
                payload: vec![AlignedBlock::new([1.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![5]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 5: Pos 100.0, Neighbors: [0]
            Node {
                payload: vec![AlignedBlock::new([100.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: HierarchicalFixedSet::new(vec![vec![0]]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];

        let graph = AdjacencyGraph::new_hnsw(
            nodes,
            HNSWEngineStarter::new(EngineStarter::new(4, SIMD_LANECOUNT, 6, Some(42)), 0),
            HNSWCatapultChoice::FinalizingCatapults,
        );

        let query = vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])];
        let k = 1;
        let beam_width = 2;
        let mut stats = crate::statistics::Stats::new();

        let results1 = graph.beam_search(&query, k, beam_width, &mut stats);
        let results2 = graph.beam_search(&query, k, beam_width, &mut stats);

        // Should find node 4 as the best result
        assert_eq!(results1.len(), k);
        assert_eq!(results1[0], 4);
        assert_eq!(results1, results2);
    }

    #[test]
    fn test_catapult_usage_coverage() {
        // This test ensures that used_catapult_this_search is set to true.
        // We need a scenario where:
        // 1. First search establishes a catapult from starting node to a distant best node
        // 2. Second search uses that catapult to skip intermediate nodes
        //
        // Graph structure:
        //    0 (start) --> 1 --> 2 --> 3
        //                   \         /
        //                    4 <-----
        //
        // Query at pos 35: First search from 0 traverses 0->1->2->3, finds 3 is best (dist 0)
        // This creates catapult: 0 -> 3
        // Second search from 0 has catapult 0->3 which gets added to candidates, triggering the flag

        let nodes = vec![
            // 0: Pos 0.0, Neighbors: [1]
            Node {
                payload: vec![AlignedBlock::new([0.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![1]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 1: Pos 10.0, Neighbors: [2, 4]
            Node {
                payload: vec![AlignedBlock::new([10.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![2, 4]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 2: Pos 20.0, Neighbors: [3]
            Node {
                payload: vec![AlignedBlock::new([20.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![3]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 3: Pos 35.0, Neighbors: [4]
            Node {
                payload: vec![AlignedBlock::new([35.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![4]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
            // 4: Pos 50.0, Neighbors: []
            Node {
                payload: vec![AlignedBlock::new([50.0; SIMD_LANECOUNT])].into_boxed_slice(),
                neighbors: FlatFixedSet::new(vec![]),
                catapults: RwLock::new(UnboundedNeighborSet::new()),
            },
        ];

        let graph = AdjacencyGraph::new_flat(
            nodes,
            EngineStarter::new(4, SIMD_LANECOUNT, 5, Some(42)),
            FlatCatapultChoice::CatapultsEnabled,
        );

        // Manually add a catapult from node 0 to node 3 (simulating a previous search result)
        graph.adjacency[0].catapults.write().unwrap().insert(3);

        let query = vec![AlignedBlock::new([35.0; SIMD_LANECOUNT])];
        let k = 2;
        let beam_width = 3;

        // Use beam_search_raw directly with starting point [0]
        // When node 0 is expanded, its catapult to node 3 should be used
        let mut stats = crate::statistics::Stats::new();
        let starting_points = vec![0];

        let results =
            graph.beam_search_raw(&query, &starting_points, k, beam_width, (), &mut stats);

        // Should find node 3 as the best result
        assert_eq!(results[0], 3);

        // Should have used the catapult from node 0 to node 3
        assert_eq!(
            stats.get_searches_with_catapults(),
            1,
            "Expected catapult to be used during search"
        );
        assert!(
            stats.get_catapults_used() > 0,
            "Expected at least one catapult edge to be added to candidates"
        );
    }
    */
}
