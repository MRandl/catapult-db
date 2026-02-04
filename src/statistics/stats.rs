/// Performance statistics for tracking beam search operations.
///
/// Collects metrics about search efficiency including the number of searches performed,
/// nodes explored, distances computed, and how often catapults provided acceleration.
/// Statistics can be merged across threads for parallel workloads.
pub struct Stats {
    /// Total number of beam search calls performed
    beam_calls: usize,

    /// Total number of nodes expanded during searches
    nodes_visited: usize,

    /// Total number of distance computations performed
    dists_computed: usize,

    /// Number of searches that benefited from at least one catapult starting point
    searches_with_catapults: usize,
}

impl Stats {
    /// Creates a new statistics tracker with all counters initialized to zero.
    ///
    /// # Returns
    /// A new `Stats` instance with zeroed counters
    pub fn new() -> Self {
        Stats {
            beam_calls: 0,
            nodes_visited: 0,
            dists_computed: 0,
            searches_with_catapults: 0,
        }
    }

    /// Increments the beam search call counter by one.
    ///
    /// Should be called once per beam search invocation.
    pub fn bump_beam_calls(&mut self) {
        self.beam_calls += 1
    }

    /// Returns the total number of beam searches performed.
    ///
    /// # Returns
    /// The current beam call count
    pub fn get_beam_calls(&self) -> usize {
        self.beam_calls
    }

    /// Increments the visited nodes counter by one.
    ///
    /// Should be called each time a node is expanded during beam search.
    pub fn bump_nodes_visited(&mut self) {
        self.nodes_visited += 1;
    }

    /// Returns the total number of nodes visited across all searches.
    ///
    /// # Returns
    /// The current nodes visited count
    pub fn get_nodes_visited(&self) -> usize {
        self.nodes_visited
    }

    /// Increments the distance computation counter.
    ///
    /// # Arguments
    /// * `amt` - The number of distance computations to add
    pub fn bump_computed_dists(&mut self, amt: usize) {
        self.dists_computed += amt;
    }

    /// Returns the total number of distance computations performed.
    ///
    /// # Returns
    /// The current distance computation count
    pub fn get_computed_dists(&self) -> usize {
        self.dists_computed
    }

    /// Increments the counter for searches that used catapults.
    ///
    /// Should be called once per search that benefited from at least one catapult
    /// starting point (i.e., a cached result from a similar previous query).
    pub fn bump_searches_with_catapults(&mut self) {
        self.searches_with_catapults += 1;
    }

    /// Returns the number of searches that benefited from catapults.
    ///
    /// # Returns
    /// The current catapult-accelerated search count
    pub fn get_searches_with_catapults(&self) -> usize {
        self.searches_with_catapults
    }

    /// Merges two statistics objects by summing their counters.
    ///
    /// This is useful for aggregating statistics from multiple threads or batches.
    ///
    /// # Arguments
    /// * `othr` - The other statistics object to merge with
    ///
    /// # Returns
    /// A new `Stats` instance with summed counters
    pub fn merge(&self, othr: &Self) -> Self {
        Self {
            beam_calls: self.beam_calls + othr.beam_calls,
            nodes_visited: self.nodes_visited + othr.nodes_visited,
            dists_computed: self.dists_computed + othr.dists_computed,
            searches_with_catapults: self.searches_with_catapults + othr.searches_with_catapults,
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
