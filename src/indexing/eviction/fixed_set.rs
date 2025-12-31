use std::{fmt::Debug, marker::PhantomData};

use crate::indexing::graph_hierarchy::GraphSearchAlgo;

pub struct FixedSet<GraphSearchType>
where
    GraphSearchType: GraphSearchAlgo,
{
    neighbors: Box<[usize]>,
    _phantom: PhantomData<GraphSearchType>,
}

impl<T> FixedSet<T>
where
    T: GraphSearchAlgo,
{
    pub fn new(initial_values: Vec<usize>) -> Self {
        FixedSet {
            neighbors: initial_values.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn to_box(&self, _at_level: Option<u32>) -> Box<[usize]> {
        self.neighbors.clone()
    }
}

impl<T> Debug for FixedSet<T>
where
    T: GraphSearchAlgo,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedSet")
            .field("neighbors", &self.neighbors)
            .field("_phantom", &self._phantom)
            .finish()
    }
}
