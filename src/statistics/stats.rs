pub struct Stats {
    beam_calls: usize,
    nodes_visited: usize,
    dists_computed: usize,
    searches_with_catapults: usize,
    work_avoided: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            beam_calls: 0,
            nodes_visited: 0,
            dists_computed: 0,
            searches_with_catapults: 0,
            work_avoided: 0,
        }
    }

    /// Record into the statistics object that a new beam search call has been performed
    pub fn bump_beam_calls(&mut self) {
        self.beam_calls += 1
    }

    pub fn get_beam_calls(&self) -> usize {
        self.beam_calls
    }

    // record that the amonut of nodes explored by the beam search
    pub fn bump_nodes_visited(&mut self) {
        self.nodes_visited += 1;
    }

    pub fn get_nodes_visited(&self) -> usize {
        self.nodes_visited
    }

    pub fn bump_computed_dists(&mut self, amt: usize) {
        self.dists_computed += amt;
    }

    pub fn get_computed_dists(&self) -> usize {
        self.dists_computed
    }

    /// Record that a search effectively used at least one catapult
    pub fn bump_searches_with_catapults(&mut self) {
        self.searches_with_catapults += 1;
    }

    pub fn get_searches_with_catapults(&self) -> usize {
        self.searches_with_catapults
    }

    pub fn bump_work_avoided(&mut self, amt: usize) {
        self.work_avoided += amt;
    }

    pub fn get_work_avoided(&self) -> usize {
        self.work_avoided
    }

    pub fn merge(&self, othr: &Self) -> Self {
        Self {
            beam_calls: self.beam_calls + othr.beam_calls,
            nodes_visited: self.nodes_visited + othr.nodes_visited,
            dists_computed: self.dists_computed + othr.dists_computed,
            searches_with_catapults: self.searches_with_catapults + othr.searches_with_catapults,
            work_avoided: self.work_avoided + othr.work_avoided,
        }
    }
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_initialized_to_zero() {
        let stats = Stats::default();
        assert_eq!(stats.get_beam_calls(), 0);
        assert_eq!(stats.get_computed_dists(), 0);
        assert_eq!(stats.get_nodes_visited(), 0);
        assert_eq!(stats.get_searches_with_catapults(), 0);
    }

    #[test]
    fn test_bump_beam_calls() {
        let mut stats = Stats::new();
        assert_eq!(stats.get_beam_calls(), 0);

        stats.bump_beam_calls();
        assert_eq!(stats.get_beam_calls(), 1);

        stats.bump_beam_calls();
        assert_eq!(stats.get_beam_calls(), 2);

        stats.bump_beam_calls();
        assert_eq!(stats.get_beam_calls(), 3);
    }

    #[test]
    fn test_bump_nodes_visited() {
        let mut stats = Stats::new();
        assert_eq!(stats.get_nodes_visited(), 0);

        stats.bump_nodes_visited();
        assert_eq!(stats.get_nodes_visited(), 1);

        stats.bump_nodes_visited();
        assert_eq!(stats.get_nodes_visited(), 2);

        stats.bump_nodes_visited();
        assert_eq!(stats.get_nodes_visited(), 3);
    }

    #[test]
    fn test_bump_computed_dists() {
        let mut stats = Stats::new();
        assert_eq!(stats.get_computed_dists(), 0);

        stats.bump_computed_dists(5);
        assert_eq!(stats.get_computed_dists(), 5);

        stats.bump_computed_dists(10);
        assert_eq!(stats.get_computed_dists(), 15);

        stats.bump_computed_dists(0);
        assert_eq!(stats.get_computed_dists(), 15);

        stats.bump_computed_dists(100);
        assert_eq!(stats.get_computed_dists(), 115);
    }

    #[test]
    fn test_bump_searches_with_catapults() {
        let mut stats = Stats::new();
        assert_eq!(stats.get_searches_with_catapults(), 0);

        stats.bump_searches_with_catapults();
        assert_eq!(stats.get_searches_with_catapults(), 1);

        stats.bump_searches_with_catapults();
        assert_eq!(stats.get_searches_with_catapults(), 2);

        stats.bump_searches_with_catapults();
        assert_eq!(stats.get_searches_with_catapults(), 3);
    }

    #[test]
    fn test_merge() {
        let mut stats1 = Stats::new();
        stats1.bump_beam_calls();
        stats1.bump_beam_calls();
        stats1.bump_nodes_visited();
        stats1.bump_computed_dists(10);
        stats1.bump_searches_with_catapults();

        let mut stats2 = Stats::new();
        stats2.bump_beam_calls();
        stats2.bump_nodes_visited();
        stats2.bump_nodes_visited();
        stats2.bump_nodes_visited();
        stats2.bump_computed_dists(25);
        stats2.bump_searches_with_catapults();
        stats2.bump_searches_with_catapults();

        let merged = stats1.merge(&stats2);

        assert_eq!(merged.get_beam_calls(), 3);
        assert_eq!(merged.get_nodes_visited(), 4);
        assert_eq!(merged.get_computed_dists(), 35);
        assert_eq!(merged.get_searches_with_catapults(), 3);
    }
}
