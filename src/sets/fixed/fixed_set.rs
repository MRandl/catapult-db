use std::fmt::Debug;

pub trait FixedSet: Debug {
    type LevelContext: Copy + Clone;
    fn to_level(&self, _at_level: Self::LevelContext) -> Box<[usize]>;
}

pub struct FlatFixedSet {
    neighbors: Box<[usize]>,
}

impl FlatFixedSet {
    pub fn new(initial_values: Vec<usize>) -> Self {
        FlatFixedSet {
            neighbors: initial_values.into_boxed_slice(),
        }
    }
}

impl FixedSet for FlatFixedSet {
    type LevelContext = ();
    fn to_level(&self, _: ()) -> Box<[usize]> {
        self.neighbors.clone()
    }
}

#[derive(Debug)]
pub struct HierarchicalFixedSet {
    neighbors: Box<[Box<[usize]>]>,
}

impl HierarchicalFixedSet {
    pub fn new(initial_values: Vec<Vec<usize>>) -> Self {
        HierarchicalFixedSet {
            neighbors: initial_values
                .into_iter()
                .map(|v| v.into_boxed_slice())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl FixedSet for HierarchicalFixedSet {
    type LevelContext = usize;
    fn to_level(&self, level: usize) -> Box<[usize]> {
        self.neighbors.get(level).unwrap().clone()
    }
}

impl Debug for FlatFixedSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedSet")
            .field("neighbors", &self.neighbors)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_fixed_set() {
        let values = vec![1, 2, 3, 4, 5];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values.clone());

        assert_eq!(fixed_set.neighbors.len(), 5);
        assert_eq!(&*fixed_set.neighbors, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_new_with_empty_vec() {
        let values: Vec<usize> = vec![];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(fixed_set.neighbors.len(), 0);
    }

    #[test]
    fn test_new_with_single_element() {
        let values = vec![42];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(fixed_set.neighbors.len(), 1);
        assert_eq!(fixed_set.neighbors[0], 42);
    }

    #[test]
    fn test_to_box_returns_cloned_neighbors_flat_search() {
        let values = vec![10, 20, 30];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result = fixed_set.to_level(());

        assert_eq!(result.len(), 3);
        assert_eq!(&*result, &[10, 20, 30]);
    }

    #[test]
    fn test_to_box_returns_cloned_neighbors_hnsw_search() {
        let values = vec![100, 200, 300, 400];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result = fixed_set.to_level(()); // level 5

        assert_eq!(result.len(), 4);
        assert_eq!(&*result, &[100, 200, 300, 400]);
    }

    #[test]
    fn test_to_box_with_different_levels_returns_same_data() {
        let values = vec![1, 2, 3];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result_level_0 = fixed_set.to_level(());
        let result_level_5 = fixed_set.to_level(());

        assert_eq!(&*result_level_0, &*result_level_5);
        assert_eq!(&*result_level_0, &[1, 2, 3]);
    }

    #[test]
    fn test_to_box_independence() {
        let values = vec![1, 2, 3];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result1 = fixed_set.to_level(());
        let result2 = fixed_set.to_level(());

        // Both results should have the same values
        assert_eq!(&*result1, &*result2);

        // But they should be independent clones
        assert_ne!(result1.as_ptr(), result2.as_ptr());
    }

    #[test]
    fn test_debug_formatting() {
        let values = vec![7, 8, 9];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let debug_string = format!("{fixed_set:?}");

        assert!(debug_string.contains("FixedSet"));
        assert!(debug_string.contains("neighbors"));
        assert!(debug_string.contains("7"));
        assert!(debug_string.contains("8"));
        assert!(debug_string.contains("9"));
    }

    #[test]
    fn test_large_dataset() {
        let values: Vec<usize> = (0..10000).collect();
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values.clone());

        assert_eq!(fixed_set.neighbors.len(), 10000);
        assert_eq!(fixed_set.neighbors[0], 0);
        assert_eq!(fixed_set.neighbors[9999], 9999);
    }

    #[test]
    fn test_with_duplicate_values() {
        let values = vec![5, 5, 5, 10, 10];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(fixed_set.neighbors.len(), 5);
        assert_eq!(&*fixed_set.neighbors, &[5, 5, 5, 10, 10]);
    }

    #[test]
    fn test_with_max_usize_value() {
        let values = vec![usize::MAX, usize::MAX - 1, 0];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(fixed_set.neighbors.len(), 3);
        assert_eq!(fixed_set.neighbors[0], usize::MAX);
        assert_eq!(fixed_set.neighbors[1], usize::MAX - 1);
        assert_eq!(fixed_set.neighbors[2], 0);
    }

    #[test]
    fn test_different_graph_search_types() {
        let values = vec![1, 2, 3];

        let flat_set: FlatFixedSet = FlatFixedSet::new(values.clone());
        let hnsw_set: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(&*flat_set.neighbors, &*hnsw_set.neighbors);
    }
}
