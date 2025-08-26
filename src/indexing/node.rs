use crate::indexing::neighbor_set::NeighborSet;

pub struct Node<P> {
    neighbors : NeighborSet,
    payload : P
}