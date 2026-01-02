use std::{fmt::Debug, marker::PhantomData};

use crate::search::GraphSearchAlgorithm;

pub struct FixedSet<GraphSearchType> {
    neighbors: Box<[usize]>,
    _phantom: PhantomData<GraphSearchType>,
}

impl<T> FixedSet<T>
where
    T: GraphSearchAlgorithm,
{
    pub fn new(initial_values: Vec<usize>) -> Self {
        FixedSet {
            neighbors: initial_values.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn to_box(&self, _at_level: T::LevelContext) -> Box<[usize]> {
        self.neighbors.clone()
    }
}

impl<T> Debug for FixedSet<T>
where
    T: GraphSearchAlgorithm,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedSet")
            .field("neighbors", &self.neighbors)
            .finish()
    }
}
