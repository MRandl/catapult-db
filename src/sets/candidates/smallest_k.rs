use std::vec::IntoIter;

use crate::sets::candidates::CandidateEntry;

/// A bounded priority queue that maintains the k smallest unique candidate entries.
///
/// This structure keeps track of the k smallest `CandidateEntry` elements seen so far,
/// automatically deduplicating by (distance, index) and evicting larger elements when
/// capacity is exceeded. Elements are maintained in sorted order for efficient access
/// to the best candidates.
///
/// # Insertion Semantics
/// - Duplicate entries (same distance and index) are ignored
/// - If not full, new unique entries are inserted in sorted order
/// - If full and the new entry is smaller than the current maximum, the maximum is evicted
/// - If full and the new entry is larger than or equal to the maximum, it is ignored
///
/// # Time Complexity
/// - `insert_batch`: O(n log k) where n is batch size and k is capacity
/// - Deduplication check: O(k) worst case when many entries share the same distance
pub struct SmallestKCandidates {
    sorted_members: Vec<CandidateEntry>,
    capacity: usize,
}

impl SmallestKCandidates {
    /// Creates a new empty `SmallestKCandidates` with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of unique candidates to retain, must be greater than 0
    ///
    /// # Returns
    /// A new empty `SmallestKCandidates` instance
    ///
    /// # Panics
    /// Panics if `capacity == 0`
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        SmallestKCandidates {
            sorted_members: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts a batch of candidate entries, maintaining the k smallest unique elements.
    ///
    /// For each item in the batch, attempts to insert it while maintaining sorted order
    /// and the capacity constraint. Duplicates (same distance and index) are ignored.
    ///
    /// # Arguments
    /// * `items` - Slice of candidate entries to insert
    ///
    /// # Returns
    /// The number of items actually added (excluding duplicates and rejected entries)
    pub fn insert_batch(&mut self, items: &[CandidateEntry]) -> usize {
        let mut added_count = 0;

        for item in items {
            // 1. find the insertion point (O(log K))
            let idx = self.sorted_members.partition_point(|m| m < item);

            // 2. duplicate Check - when distances are equal, we need to check all entries with the same distance
            // Check if this index already exists anywhere in the array with the same distance
            let mut is_duplicate = false;
            let mut check_idx = idx;
            while check_idx < self.sorted_members.len()
                && self.sorted_members[check_idx].distance == item.distance
            {
                if self.sorted_members[check_idx].index == item.index {
                    is_duplicate = true;
                    break;
                }
                check_idx += 1;
            }
            if is_duplicate {
                continue;
            }

            // 3. size Management
            if self.sorted_members.len() < self.capacity {
                // Not full yet: maintain sort order by inserting at idx
                self.sorted_members.insert(idx, *item);
                added_count += 1;
            } else if idx < self.capacity {
                // Full, but new item is smaller than our current max (last element)
                // Remove the largest element and insert the new one
                self.sorted_members.pop();
                self.sorted_members.insert(idx, *item);
                added_count += 1;
            }
            // If idx == self.capacity, item is >= all current members; ignore it.
        }

        added_count
    }

    /// Returns an iterator over the candidate entries in sorted order (smallest to largest).
    ///
    /// # Returns
    /// An iterator that yields references to `CandidateEntry` in ascending distance order
    pub fn iter(&self) -> std::slice::Iter<'_, CandidateEntry> {
        self.sorted_members.iter()
    }
}

impl IntoIterator for SmallestKCandidates {
    type Item = CandidateEntry;
    type IntoIter = IntoIter<CandidateEntry>;

    fn into_iter(self) -> IntoIter<CandidateEntry> {
        self.sorted_members.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use rand_distr::num_traits::ToPrimitive;

    use crate::search::NodeId;

    use super::*;

    fn contents_sorted(sk: &SmallestKCandidates) -> Vec<CandidateEntry> {
        // Access via clone; sorts ascending.
        let mut ret = sk.sorted_members.to_vec();
        ret.sort();
        ret
    }

    #[test]
    fn keeps_k_smallest_basic() {
        let mut sk = SmallestKCandidates::new(3);
        for x in 1..=10 {
            sk.insert_batch(&[CandidateEntry {
                distance: x.to_f32().unwrap().into(),
                index: NodeId { internal: x },
                has_catapult_ancestor: false,
            }]);
        }
        assert_eq!(sk.sorted_members.len(), 3);
        assert_eq!(
            contents_sorted(&sk)
                .iter()
                .map(|c| c.index.internal)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn keeps_k_smallest_batch() {
        // Capacity of 3
        let mut sk = SmallestKCandidates::new(3);

        // Batch 1: Unsorted, includes duplicates, and fits within capacity
        let batch_1 = vec![
            CandidateEntry {
                distance: 10.0.into(),
                index: NodeId { internal: 10 },
                has_catapult_ancestor: false,
            },
            CandidateEntry {
                distance: 5.0.into(),
                index: NodeId { internal: 5 },
                has_catapult_ancestor: false,
            },
            CandidateEntry {
                distance: 10.0.into(),
                index: NodeId { internal: 10 },
                has_catapult_ancestor: false,
            }, // Duplicate inside batch
        ];
        sk.insert_batch(&batch_1);

        // State should be [5, 10]
        assert_eq!(sk.sorted_members.len(), 2);
        assert_eq!(sk.sorted_members[0].index.internal, 5);

        // Batch 2: Larger than capacity, contains smaller values,
        // and contains a duplicate of an existing member (5)
        let batch_2 = vec![
            CandidateEntry {
                distance: 2.0.into(),
                index: NodeId { internal: 2 },
                has_catapult_ancestor: false,
            }, // New smallest
            CandidateEntry {
                distance: 5.0.into(),
                index: NodeId { internal: 5 },
                has_catapult_ancestor: false,
            }, // Duplicate of existing
            CandidateEntry {
                distance: 7.0.into(),
                index: NodeId { internal: 7 },
                has_catapult_ancestor: false,
            }, // New middle
            CandidateEntry {
                distance: 1.0.into(),
                index: NodeId { internal: 1 },
                has_catapult_ancestor: false,
            }, // New absolute smallest
        ];
        sk.insert_batch(&batch_2);

        // Final state should be the 3 smallest unique indices: [1, 2, 5]
        // 7 and 10 should have been displaced/ignored.
        assert_eq!(sk.sorted_members.len(), 3);

        let results: Vec<usize> = sk.sorted_members.iter().map(|c| c.index.internal).collect();
        assert_eq!(results, vec![1, 2, 5]);
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _ = SmallestKCandidates::new(0);
    }

    // Helper to create a CandidateEntry quickly
    fn entry(dist: f32, idx: usize) -> CandidateEntry {
        CandidateEntry {
            distance: dist.into(),
            index: NodeId { internal: idx },
            has_catapult_ancestor: false,
        }
    }

    #[test]
    fn test_reverse_insertion() {
        // Inserting largest to smallest should result in the k smallest
        let mut sk = SmallestKCandidates::new(3);
        for i in (1..=10).rev() {
            sk.insert_batch(&[entry(i as f32, i)]);
        }
        assert_eq!(sk.sorted_members.len(), 3);
        let mut results: Vec<_> = sk.into_iter().map(|c| c.index.internal).collect();
        results.sort();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_duplicate_prevention() {
        let mut sk = SmallestKCandidates::new(5);
        // Insert the same item multiple times
        sk.insert_batch(&[
            entry(1.0, 200),
            entry(1.0, 100),
            entry(1.0, 100),
            entry(1.0, 200),
        ]);

        assert_eq!(sk.sorted_members.len(), 2);
    }

    #[test]
    fn test_threshold_eviction() {
        let mut sk = SmallestKCandidates::new(2);

        // Fill it: [10.0, 20.0]
        sk.insert_batch(&[entry(10.0, 1), entry(20.0, 2), entry(30.0, 3)]);

        assert_eq!(sk.sorted_members.len(), 2);

        // This should replace 20.0 (15.0 < 20.0)
        sk.insert_batch(&[entry(15.0, 4)]);

        let mut final_scores: Vec<f32> = sk.iter().map(|c| c.distance.0).collect();
        final_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(final_scores, vec![10.0, 15.0]);
    }

    #[test]
    fn test_capacity_one() {
        let mut sk = SmallestKCandidates::new(1);
        sk.insert_batch(&[entry(50.0, 1), entry(10.0, 2), entry(100.0, 3)]); // Ignore

        assert_eq!(sk.sorted_members.len(), 1);
        assert_eq!(sk.iter().next().unwrap().index.internal, 2);
    }

    #[test]
    fn test_randomized_consistency() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);

        let k = 10;
        let mut sk = SmallestKCandidates::new(k);
        let mut all_entries = Vec::new();

        for i in 0..100 {
            let dist = rng.random_range(0.0..1000.0);
            let e = entry(dist, i);
            sk.insert_batch(std::slice::from_ref(&e));
            all_entries.push(e);
        }

        // The "Truth": Sort all entries and take the first K unique indices
        all_entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Simple unique by index for comparison
        let mut expected: Vec<CandidateEntry> = Vec::new();
        for e in all_entries {
            if !expected.iter().any(|x| x.index == e.index) {
                expected.push(e);
            }
            if expected.len() == k {
                break;
            }
        }

        let mut actual: Vec<_> = sk.into_iter().collect();
        actual.sort();

        assert_eq!(actual.len(), k);
        for i in 0..k {
            assert_eq!(actual[i].index, expected[i].index);
        }
    }
}
