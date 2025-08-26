use crate::{
    candidates::{BitSet, CandidateEntry, SmallestK},
    indexing::node::Node,
    numerics::f32slice::VectorLike,
};

struct AdjacencyGraph {
    adjacency: Vec<Node>,
}

impl AdjacencyGraph {
    pub fn pick_starting_point(&self, _query: &[f32]) -> usize {
        // will involve lsh at some point
        assert!(self.adjacency.len() > 0);
        0
    }

    pub fn beam_search(&self, query: &[f32], k: usize, beam_width: usize) -> SmallestK<CandidateEntry> {
        assert!(beam_width >= k);

        let mut candidates: SmallestK<CandidateEntry> = SmallestK::new(beam_width);
        let mut visited = BitSet::new(self.adjacency.len());

        let starting_index = self.pick_starting_point(&query);
        let starting_point = &self.adjacency[starting_index];
        let starting_score = starting_point.payload.l2_squared(query);

        candidates.insert(CandidateEntry {
            distance: starting_score.into(),
            index: starting_index,
        });

        let mut best_index_in_candidates: Option<usize> = Some(starting_index);

        while let Some(best_index) = best_index_in_candidates {
            let best_candidate_node = &self.adjacency[best_index];
            let best_neighbors = &best_candidate_node.neighbors;
            for &neighbor in best_neighbors {
                let neighbor_node = &self.adjacency[neighbor];
                let neighbor_distance = neighbor_node.payload.l2_squared(query);
                candidates.insert(CandidateEntry {
                    distance: neighbor_distance.into(),
                    index: neighbor,
                });
            }
            visited.set_bit(best_index);
            best_index_in_candidates = candidates
                .iter()
                .filter(|&elem| !visited.get_bit(elem.index))
                .min()
                .map(|e| e.index)
        }

        candidates
    }
}
