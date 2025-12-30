use std::vec::IntoIter;

use crate::candidates::CandidateEntry;

/// A bounded structure that keeps the *k smallest* elements seen so far,
/// implemented using `std::collections::HashSet`.
///
/// # Semantics
/// - Let `k = capacity`. While not full, all inserted elements are retained.
/// - Once full, on inserting `item`:
///   - If `item` is **strictly smaller** than the current maximum among kept items
///     (the *threshold*), the threshold is evicted and `item` is inserted.
///   - Otherwise, `item` is ignored (dropped).
/// - Ties: when full, inserting an element *equal* to the threshold is dropped.
///
/// `peek()` exposes the *current maximum among the retained elements* (the threshold).
///
/// # Complexity
/// Memory is `O(k)`. Insertion is `O(k)` due to finding the max element, plus `O(1)`
/// average case for HashSet operations. `peek()` is `O(k)` as it scans for the maximum.
/// ```
pub struct SmallestK {
    members: Vec<CandidateEntry>,
    capacity: usize,
}

impl SmallestK {
    /// Creates a new `SmallestK` that retains at most `capacity` elements.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    ///
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        SmallestK {
            members: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts `item`, possibly evicting the current threshold if `item` is strictly smaller.
    ///
    /// Equal items to the threshold are dropped when full.
    pub fn insert(&mut self, item: CandidateEntry) {
        debug_assert!(self.members.len() <= self.capacity);
        // should be impossible to break this, by construction.

        let mut max_index = 0;

        // Single pass: find max element index AND check for equality
        for (i, member) in self.members.iter().enumerate() {
            if member.index == item.index {
                return;
            }
            if member > &self.members[max_index] {
                max_index = i;
            }
        }

        if self.members.len() < self.capacity {
            self.members.push(item);
        } else {
            // If full, only swap if the new item is strictly smaller than our found max
            if item < self.members[max_index] {
                self.members[max_index] = item;
            }
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CandidateEntry> {
        self.members.iter()
    }
}

impl IntoIterator for SmallestK {
    type Item = CandidateEntry;
    type IntoIter = IntoIter<CandidateEntry>;

    fn into_iter(self) -> IntoIter<CandidateEntry> {
        self.members.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use rand_distr::num_traits::ToPrimitive;

    use super::*;

    fn contents_sorted(sk: &SmallestK) -> Vec<CandidateEntry> {
        // Access via clone; sorts ascending.
        let mut ret = sk.members.to_vec();
        ret.sort();
        ret
    }

    #[test]
    fn keeps_k_smallest_basic() {
        let mut sk = SmallestK::new(3);
        for x in 1..=10 {
            sk.insert(CandidateEntry {
                distance: x.to_f32().unwrap().into(),
                index: x,
            });
        }
        assert_eq!(sk.members.len(), 3);
        assert_eq!(
            contents_sorted(&sk)
                .iter()
                .map(|c| c.index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _ = SmallestK::new(0);
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
        let mut sk = SmallestK::new(3);
        for i in (1..=10).rev() {
            sk.insert(entry(i as f32, i));
        }
        assert_eq!(sk.members.len(), 3);
        let mut results: Vec<_> = sk.into_iter().map(|c| c.index).collect();
        results.sort();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_duplicate_prevention() {
        let mut sk = SmallestK::new(5);
        // Insert the same item multiple times
        sk.insert(entry(1.0, 100));
        sk.insert(entry(1.0, 100));
        sk.insert(entry(2.0, 200));

        assert_eq!(
            sk.members.len(),
            2,
            "Should have ignored identical duplicate"
        );
    }

    #[test]
    fn test_threshold_eviction() {
        let mut sk = SmallestK::new(2);

        // Fill it: [10.0, 20.0]
        sk.insert(entry(10.0, 1));
        sk.insert(entry(20.0, 2));

        // This should be ignored (30.0 > 20.0)
        sk.insert(entry(30.0, 3));
        assert_eq!(sk.members.len(), 2);

        // This should replace 20.0 (15.0 < 20.0)
        sk.insert(entry(15.0, 4));

        let mut final_scores: Vec<f32> = sk.iter().map(|c| c.distance.0).collect();
        final_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(final_scores, vec![10.0, 15.0]);
    }

    #[test]
    fn test_capacity_one() {
        let mut sk = SmallestK::new(1);
        sk.insert(entry(50.0, 1));
        sk.insert(entry(10.0, 2)); // Replace
        sk.insert(entry(100.0, 3)); // Ignore

        assert_eq!(sk.members.len(), 1);
        assert_eq!(sk.iter().next().unwrap().index, 2);
    }

    #[test]
    fn test_randomized_consistency() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);

        let k = 10;
        let mut sk = SmallestK::new(k);
        let mut all_entries = Vec::new();

        for i in 0..100 {
            let dist = rng.random_range(0.0..1000.0);
            let e = entry(dist, i);
            sk.insert(e.clone());
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
