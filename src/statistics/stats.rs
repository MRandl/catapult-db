pub struct Stats {
    beam_calls: usize,
    nodes_queried: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            beam_calls: 0,
            nodes_queried: 0,
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

    /// Write all contents of the stats object to stdout. Will dump to the path at some point
    /// when debugging is over
    pub fn dump(&self, path: String) {
        //TODO
        println!("Dumping to sysout, supposedly at path {path}");
        println!(
            "beam calls : {}, nodes queried : {}",
            self.beam_calls, self.nodes_queried
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
