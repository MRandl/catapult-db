#[derive(Debug)]
pub struct FixedSet {
    neighbors: Box<[usize]>,
}

impl FixedSet {
    pub fn new(initial_values: Vec<usize>) -> Self {
        FixedSet {
            neighbors: initial_values.into_boxed_slice(),
        }
    }

    pub fn to_vec(&self, _at_level: Option<u32>) -> Box<[usize]> {
        self.neighbors.clone()
    }
}
