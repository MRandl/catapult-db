use std::fmt::Debug;

/// A trait for immutable neighbor sets in flat proximity graphs.
///
/// This abstraction provides a uniform interface for accessing neighbor relationships
/// in graph nodes.
pub trait FixedSet: Debug {
    /// Returns a cloned copy of the neighbor indices.
    ///
    /// # Returns
    /// A boxed slice containing the neighbor node indices
    fn to_slice(&self) -> Box<[usize]>;
}

/// An immutable set of neighbor indices for a flat proximity graph node.
///
/// Stores a single fixed list of neighbor indices that does not vary by level.
/// This is suitable for single-layer proximity graphs like DiskANN-style structures.
pub struct FlatFixedSet {
    neighbors: Box<[usize]>,
}

impl FlatFixedSet {
    /// Creates a new flat fixed set from a vector of neighbor indices.
    ///
    /// The provided indices are converted to an immutable boxed slice.
    ///
    /// # Arguments
    /// * `initial_values` - Vector of node indices representing the neighbors
    ///
    /// # Returns
    /// A new `FlatFixedSet` containing the provided neighbor indices
    pub fn new(initial_values: Vec<usize>) -> Self {
        FlatFixedSet {
            neighbors: initial_values.into_boxed_slice(),
        }
    }
}

impl FixedSet for FlatFixedSet {
    fn to_slice(&self) -> Box<[usize]> {
        self.neighbors.clone()
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
    fn test_neighbors_returns_cloned_data() {
        let values = vec![10, 20, 30];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result = fixed_set.to_slice();

        assert_eq!(result.len(), 3);
        assert_eq!(&*result, &[10, 20, 30]);
    }

    #[test]
    fn test_neighbors_returns_all_values() {
        let values = vec![100, 200, 300, 400];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result = fixed_set.to_slice();

        assert_eq!(result.len(), 4);
        assert_eq!(&*result, &[100, 200, 300, 400]);
    }

    #[test]
    fn test_neighbors_consistent_across_calls() {
        let values = vec![1, 2, 3];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result1 = fixed_set.to_slice();
        let result2 = fixed_set.to_slice();

        assert_eq!(&*result1, &*result2);
        assert_eq!(&*result1, &[1, 2, 3]);
    }

    #[test]
    fn test_neighbors_independence() {
        let values = vec![1, 2, 3];
        let fixed_set: FlatFixedSet = FlatFixedSet::new(values);

        let result1 = fixed_set.to_slice();
        let result2 = fixed_set.to_slice();

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
    fn test_multiple_instances_with_same_data() {
        let values = vec![1, 2, 3];

        let set1: FlatFixedSet = FlatFixedSet::new(values.clone());
        let set2: FlatFixedSet = FlatFixedSet::new(values);

        assert_eq!(&*set1.neighbors, &*set2.neighbors);
    }
}
