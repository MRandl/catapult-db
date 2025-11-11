pub trait NeighborSet {
    fn as_slice(&self) -> Vec<usize>;
}

pub trait EvictionNeighborSet: NeighborSet {
    fn insert(&mut self, neighbor: usize);
    fn new() -> Self;
}
