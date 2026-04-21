use hashbrown::HashMap;

/// Per-(src, dst) edge tracking data collected during beam search.
///
/// Enabled only when adversarial edge analysis is requested; not tracked in normal runs.
/// **Not merged** by `Stats::merge` — merge manually via union/sum after joining threads.
pub struct AdvEdgeTracking {
    /// How many times each directed graph edge (src, dst) had its distance computed,
    /// i.e. src was expanded and dst is in src's neighbor list.
    pub edge_consider_counts: HashMap<(usize, usize), u32>,
    /// How many times each edge (src, dst) was "used": both src and dst visited in the same search.
    /// Accumulated across all queries via summation.
    pub used_edge_counts: HashMap<(usize, usize), u32>,
}

impl AdvEdgeTracking {
    pub fn new() -> Self {
        Self {
            edge_consider_counts: HashMap::new(),
            used_edge_counts: HashMap::new(),
        }
    }
}

impl Default for AdvEdgeTracking {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::statistics::AdvEdgeTracking;

    #[test]
    pub fn adversarial_hash() {
        let mut adv = AdvEdgeTracking::default();
        adv.edge_consider_counts.insert((1, 1), 1234);

        assert_eq!(adv.edge_consider_counts.get(&(1, 1)), Some(&1234));
        assert_eq!(adv.edge_consider_counts.get(&(2, 2)), None);
    }
}
