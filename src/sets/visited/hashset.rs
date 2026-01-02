use crate::sets::visited::{IntegerSet, VisitorSet};

impl VisitorSet for IntegerSet {
    fn get(&self, i: usize) -> bool {
        self.contains(&i)
    }

    fn set(&mut self, i: usize) {
        self.insert(i);
    }
}
