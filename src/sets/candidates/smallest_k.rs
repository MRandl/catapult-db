use std::vec::IntoIter;

use crate::sets::candidates::CandidateEntry;

/// A bounded structure that keeps the *k unique smallest* CandidateEntry elements seen so far,
/// implemented using array scans.
///
/// # Semantics
/// - Inserting an element that is already stored in this data structure has no effect.
/// - Inserting an element that is not already stored can go three ways:
///   + If the data structure is not full yet (i.e. with *k* unique elements), it stores the new element.
///   + Otherwise, if the newly inserted element is larger than any currently stored element, it is ignored.
///   + Otherwise, the maximum is dropped and the new element takes its place
/// ```
pub struct SmallestKCandidates {
    sorted_members: Vec<CandidateEntry>,
    capacity: usize,
}

impl SmallestKCandidates {
    /// Creates a new `SmallestKCandidates` that retains at most `capacity` elements.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    ///
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        SmallestKCandidates {
            sorted_members: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts a batch of items, maintaining the k smallest unique elements.
    /// Returns the number of elements that were actually added (not duplicates or rejected).
    pub fn insert_batch(&mut self, items: &[CandidateEntry]) -> usize {
        let mut added_count = 0;

        for item in items {
            // 1. find the insertion point (O(log K))
            let idx = self.sorted_members.partition_point(|m| m < item);

            // 2. duplicate Check (O(1) after binary search)
            if idx < self.sorted_members.len() && self.sorted_members[idx].index == item.index {
                continue;
            }

            // 3. size Management
            if self.sorted_members.len() < self.capacity {
                // Not full yet: maintain sort order by inserting at idx
                self.sorted_members.insert(idx, item.clone());
                added_count += 1;
            } else if idx < self.capacity {
                // Full, but new item is smaller than our current max (last element)
                // Remove the largest element and insert the new one
                self.sorted_members.pop();
                self.sorted_members.insert(idx, item.clone());
                added_count += 1;
            }
            // If idx == self.capacity, item is >= all current members; ignore it.
        }

        added_count
    }

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
                index: x,
            }]);
        }
        assert_eq!(sk.sorted_members.len(), 3);
        assert_eq!(
            contents_sorted(&sk)
                .iter()
                .map(|c| c.index)
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
                index: 10,
            },
            CandidateEntry {
                distance: 5.0.into(),
                index: 5,
            },
            CandidateEntry {
                distance: 10.0.into(),
                index: 10,
            }, // Duplicate inside batch
        ];
        sk.insert_batch(&batch_1);

        // State should be [5, 10]
        assert_eq!(sk.sorted_members.len(), 2);
        assert_eq!(sk.sorted_members[0].index, 5);

        // Batch 2: Larger than capacity, contains smaller values,
        // and contains a duplicate of an existing member (5)
        let batch_2 = vec![
            CandidateEntry {
                distance: 2.0.into(),
                index: 2,
            }, // New smallest
            CandidateEntry {
                distance: 5.0.into(),
                index: 5,
            }, // Duplicate of existing
            CandidateEntry {
                distance: 7.0.into(),
                index: 7,
            }, // New middle
            CandidateEntry {
                distance: 1.0.into(),
                index: 1,
            }, // New absolute smallest
        ];
        sk.insert_batch(&batch_2);

        // Final state should be the 3 smallest unique indices: [1, 2, 5]
        // 7 and 10 should have been displaced/ignored.
        assert_eq!(sk.sorted_members.len(), 3);

        let results: Vec<usize> = sk.sorted_members.iter().map(|c| c.index).collect();
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
            index: idx,
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
        let mut results: Vec<_> = sk.into_iter().map(|c| c.index).collect();
        results.sort();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_duplicate_prevention() {
        let mut sk = SmallestKCandidates::new(5);
        // Insert the same item multiple times
        sk.insert_batch(&[entry(1.0, 100), entry(1.0, 100), entry(2.0, 200)]);

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
        assert_eq!(sk.iter().next().unwrap().index, 2);
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
