use crate::{
    candidates::{BitSet, CandidateEntry, SmallestK},
    indexing::{engine_starter::EngineStarter, node::Node},
    numerics::VectorLike,
    statistics::Stats,
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
pub struct AdjacencyGraph {
    adjacency: Vec<Node>,
    starter: EngineStarter,
    stats: Stats,
}

impl AdjacencyGraph {
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
    /// The maintained `SmallestK<CandidateEntry>` of size ≤ `beam_width`. See
    /// “Issues” for the discrepancy between `k` and the returned size.
    ///
    /// # Panics
    /// - If `beam_width < k`.
    /// - If neighbor indices are out of bounds (violates the graph invariant).
    pub fn beam_search(
        &mut self,
        query: &[f32],
        k: usize,
        beam_width: usize,
    ) -> Vec<CandidateEntry> {
        assert!(beam_width >= k);
        self.stats.bump_beam_calls();

        let mut candidates: SmallestK<CandidateEntry> = SmallestK::new(beam_width);
        let mut visited = BitSet::new(self.adjacency.len());

        let starting_indices = self.starter.select_starting_points(query, k);
        for starting_index in starting_indices {
            let starting_point = &self.adjacency[starting_index];
            let starting_score = starting_point.payload.l2_squared(query);

            candidates.insert(CandidateEntry {
                distance: starting_score.into(),
                index: starting_index,
            });
        }

        let mut best_index_in_candidates: Option<usize> =
            candidates.iter().min().map(|entry| entry.index);

        while let Some(best_candidate_index) = best_index_in_candidates {
            let best_candidate_node = &self.adjacency[best_candidate_index];
            let best_candidate_neighbors = &best_candidate_node.neighbors;
            self.stats.add_node_count(best_candidate_neighbors.len());

            for &neighbor in best_candidate_neighbors {
                let neighbor_node = &self.adjacency[neighbor];
                let neighbor_distance = neighbor_node.payload.l2_squared(query);
                candidates.insert(CandidateEntry {
                    distance: neighbor_distance.into(),
                    index: neighbor,
                });
            }

            visited.set_bit(best_candidate_index);
            best_index_in_candidates = candidates
                .iter()
                .filter(|&elem| !visited.get_bit(elem.index))
                .min()
                .map(|e| e.index)
        }

        let mut candidate_vec = candidates.into_iter().collect::<Vec<_>>();
        candidate_vec.sort();
        candidate_vec.into_iter().take(k).collect()
    }
}
