use crate::candidates::{VisitorSet, set::integer_map::IntegerSet};

impl VisitorSet for IntegerSet {
    fn get(&self, i: usize) -> bool {
        self.contains(&i)
    }

    fn set(&mut self, i: usize) {
        self.insert(i);
    }
}
