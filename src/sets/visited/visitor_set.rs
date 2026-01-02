pub trait VisitorSet {
    fn get(&self, i: usize) -> bool;
    fn set(&mut self, i: usize);
}
