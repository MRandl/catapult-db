pub struct Stats {
    beam_calls: usize,
    nodes_queried: usize,
}

impl Stats {
    /// Record into the statistics object that a new beam search call has been performed
    pub fn bump_beam_calls(&mut self) {
        self.beam_calls += 1
    }

    /// Record into the statistics object that a bunch of new edges were taken into consideration
    /// in the graph exploration phase
    pub fn add_node_count(&mut self, node_amount: usize) {
        self.nodes_queried += node_amount
    }

    /// Write all contents of the stats object to stdout. Will dump to the path at some point
    /// when debugging is over
    pub fn dump(&self, path: String) {
        println!("Dumping to sysout, supposedly at path {}", path);
        println!(
            "beam calls : {}, nodes queried : {}",
            self.beam_calls, self.nodes_queried
        );
    }
}
