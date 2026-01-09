pub struct Stats {
    beam_calls: usize,
    nodes_queried: usize,
    searches_with_catapults: usize,
    catapults_used: usize,
    regular_neighbors_added: usize,
    nodes_expanded: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            beam_calls: 0,
            nodes_queried: 0,
            searches_with_catapults: 0,
            catapults_used: 0,
            regular_neighbors_added: 0,
            nodes_expanded: 0,
        }
    }

    /// Record into the statistics object that a new beam search call has been performed
    pub fn bump_beam_calls(&mut self) {
        self.beam_calls += 1
    }

    /// Record into the statistics object that a bunch of new edges were taken into consideration
    /// in the graph exploration phase
    pub fn bump_edges(&mut self, node_amount: usize) {
        self.nodes_queried += node_amount
    }

    pub fn get_beam_calls(&self) -> usize {
        self.beam_calls
    }

    pub fn get_nodes_queried(&self) -> usize {
        self.nodes_queried
    }

    /// Record that a search used at least one catapult
    pub fn bump_searches_with_catapults(&mut self) {
        self.searches_with_catapults += 1;
    }

    /// Record the number of catapult edges used in a single expansion
    pub fn bump_catapults_used(&mut self, count: usize) {
        self.catapults_used += count;
    }

    pub fn get_searches_with_catapults(&self) -> usize {
        self.searches_with_catapults
    }

    pub fn get_catapults_used(&self) -> usize {
        self.catapults_used
    }

    /// Record the number of regular neighbor edges added
    pub fn bump_regular_neighbors_added(&mut self, count: usize) {
        self.regular_neighbors_added += count;
    }

    pub fn get_regular_neighbors_added(&self) -> usize {
        self.regular_neighbors_added
    }

    /// Record that a node was expanded during beam search
    pub fn bump_nodes_expanded(&mut self) {
        self.nodes_expanded += 1;
    }

    pub fn get_nodes_expanded(&self) -> usize {
        self.nodes_expanded
    }

    /// Write all contents of the stats object to stdout. Will dump to the path at some point
    /// when debugging is over
    pub fn dump(&self, path: String) {
        //TODO
        println!("Dumping to sysout, supposedly at path {path}");
        println!(
            "beam calls : {}, nodes queried : {}, searches with catapults: {}, catapults used: {}, regular neighbors added: {}, nodes expanded: {}",
            self.beam_calls,
            self.nodes_queried,
            self.searches_with_catapults,
            self.catapults_used,
            self.regular_neighbors_added,
            self.nodes_expanded
        );
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
    fn test_new_stats_initialized_to_zero() {
        let stats = Stats::new();
        assert_eq!(stats.get_beam_calls(), 0);
        assert_eq!(stats.get_nodes_queried(), 0);
    }

    #[test]
    fn test_default_stats_initialized_to_zero() {
        let stats = Stats::default();
        assert_eq!(stats.get_beam_calls(), 0);
        assert_eq!(stats.get_nodes_queried(), 0);
    }

    #[test]
    fn test_bump_beam_calls_increments_by_one() {
        let mut stats = Stats::new();
        stats.bump_beam_calls();
        assert_eq!(stats.get_beam_calls(), 1);
        assert_eq!(stats.get_nodes_queried(), 0);
    }

    #[test]
    fn test_bump_beam_calls_multiple_times() {
        let mut stats = Stats::new();
        stats.bump_beam_calls();
        stats.bump_beam_calls();
        stats.bump_beam_calls();
        assert_eq!(stats.get_beam_calls(), 3);
        assert_eq!(stats.get_nodes_queried(), 0);
    }

    #[test]
    fn test_bump_edges_adds_node_amount() {
        let mut stats = Stats::new();
        stats.bump_edges(5);
        assert_eq!(stats.get_beam_calls(), 0);
        assert_eq!(stats.get_nodes_queried(), 5);
    }

    #[test]
    fn test_bump_edges_accumulates() {
        let mut stats = Stats::new();
        stats.bump_edges(5);
        stats.bump_edges(10);
        stats.bump_edges(3);
        assert_eq!(stats.get_nodes_queried(), 18);
    }

    #[test]
    fn test_bump_edges_with_zero() {
        let mut stats = Stats::new();
        stats.bump_edges(0);
        assert_eq!(stats.get_nodes_queried(), 0);
    }

    #[test]
    fn test_combined_operations() {
        let mut stats = Stats::new();
        stats.bump_beam_calls();
        stats.bump_edges(5);
        stats.bump_beam_calls();
        stats.bump_edges(10);

        assert_eq!(stats.get_beam_calls(), 2);
        assert_eq!(stats.get_nodes_queried(), 15);
    }

    #[test]
    fn test_dump_does_not_panic() {
        let mut stats = Stats::new();
        stats.bump_beam_calls();
        stats.bump_edges(42);
        // Just verify dump doesn't panic
        stats.dump("/tmp/test_stats.txt".to_string());
    }

    #[test]
    fn test_large_values() {
        let mut stats = Stats::new();
        for _ in 0..1000 {
            stats.bump_beam_calls();
        }
        stats.bump_edges(1_000_000);

        assert_eq!(stats.get_beam_calls(), 1000);
        assert_eq!(stats.get_nodes_queried(), 1_000_000);
    }
}
