pub trait EvictionNeighborSet {
    fn insert(&mut self, neighbor: usize);
    fn new() -> Self;
    fn to_vec(&self) -> Vec<usize>;
}
