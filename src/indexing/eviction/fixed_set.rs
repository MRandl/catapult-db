pub struct FixedSet {
    neighbors: Vec<usize>,
}

impl FixedSet {
    pub fn new(initial_values: Vec<usize>) -> Self {
        FixedSet {
            neighbors: initial_values,
        }
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.neighbors.clone()
    }
}
