use std::collections::{
    HashSet,
    hash_set::{IntoIter, Iter},
};

use std::hash::Hash;

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
///
/// # Example
/// ```
/// use catapult::candidates::SmallestK;
///
/// let mut sk = SmallestK::new(3);
/// for x in [10, 1, 7, 3, 9, 2] {
///     sk.insert(x);
/// }
/// // Keeps the three smallest: {1, 2, 3}
/// assert_eq!(sk.len(), 3);
/// assert_eq!(sk.peek().copied(), Some(3)); // threshold (max among kept)
/// ```
pub struct SmallestK<T> {
    members: HashSet<T>, //could be a btreeset. maybe evaluate the tradeoff at some point.
    capacity: usize,
}

impl<T: Ord + Eq + Hash + Clone> SmallestK<T> {
    /// Creates a new `SmallestK` that retains at most `capacity` elements.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    ///
    /// # Example
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let sk = SmallestK::<i32>::new(4);
    /// assert_eq!(sk.capacity(), 4);
    /// assert!(sk.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        SmallestK {
            members: HashSet::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the number of elements currently retained (`<= capacity`).
    ///
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let mut sk = SmallestK::new(2);
    /// sk.insert(5);
    /// assert_eq!(sk.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns the configured maximum capacity (k).
    ///
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let sk = SmallestK::<i32>::new(8);
    /// assert_eq!(sk.capacity(), 8);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if no elements are currently retained.
    ///
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let sk = SmallestK::<i32>::new(3);
    /// assert!(sk.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Returns whether the structure has reached full capacity.
    pub fn is_full(&self) -> bool {
        self.members.len() == self.capacity
    }

    /// Returns a reference to the **largest** element among the retained ones,
    /// i.e., the current *threshold*.
    ///
    /// When full, only new elements strictly smaller than this value will be admitted.
    ///
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let mut sk = SmallestK::new(3);
    /// for x in [5, 2, 7, 1, 4] { sk.insert(x); }
    /// // Kept: {1,2,4}; threshold is 4
    /// assert_eq!(sk.peek().copied(), Some(4));
    /// ```
    pub fn peek(&self) -> Option<&T> {
        self.members.iter().max()
    }

    /// Inserts `item`, possibly evicting the current threshold if `item` is strictly smaller.
    ///
    /// Equal items to the threshold are dropped when full.
    ///
    /// ```
    /// use catapult::candidates::SmallestK;
    ///
    /// let mut sk = SmallestK::new(2);
    /// sk.insert(10);
    /// sk.insert(3);
    /// sk.insert(5); // dropped (not smaller than threshold 10? threshold is max among kept -> 10)
    /// assert_eq!(sk.len(), 2);
    /// ```
    pub fn insert(&mut self, item: T) {
        debug_assert!(self.members.len() <= self.capacity);
        // should be impossible to break this, by construction.

        if self.members.len() < self.capacity {
            self.members.insert(item);
        } else {
            let bound = self
                .peek()
                .expect("set is full yet has no elements")
                .clone();

            if item < bound {
                self.members.remove(&bound);
                self.members.insert(item);
            }
        }
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.members.iter()
    }
}

impl<T> IntoIterator for SmallestK<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.members.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contents_sorted<T: Ord + Clone>(sk: &SmallestK<T>) -> Vec<T> {
        // Access via clone; sorts ascending.
        let mut ret = sk.members.iter().cloned().collect::<Vec<_>>();
        ret.sort();
        ret
    }

    #[test]
    fn keeps_k_smallest_basic() {
        let mut sk = SmallestK::new(3);
        for x in 1..=10 {
            sk.insert(x);
        }
        assert_eq!(sk.len(), 3);
        assert_eq!(contents_sorted(&sk), vec![1, 2, 3]);
        assert_eq!(sk.peek(), Some(&3)); // threshold
    }

    #[test]
    fn threshold_behavior_and_replacement() {
        let mut sk = SmallestK::new(3);
        for x in [10, 9, 8] {
            sk.insert(x);
        }
        assert_eq!(contents_sorted(&sk), vec![8, 9, 10]);
        assert_eq!(sk.peek(), Some(&10));

        sk.insert(7);
        assert_eq!(contents_sorted(&sk), vec![7, 8, 9]);
        assert_eq!(sk.peek(), Some(&9));

        sk.insert(6);
        assert_eq!(contents_sorted(&sk), vec![6, 7, 8]);
        assert_eq!(sk.peek(), Some(&8));
    }

    #[test]
    fn rejects_larger_when_full() {
        let mut sk = SmallestK::new(3);
        for x in [1, 2, 3] {
            sk.insert(x);
        }
        assert_eq!(contents_sorted(&sk), vec![1, 2, 3]);
        sk.insert(10);
        sk.insert(9);
        assert_eq!(contents_sorted(&sk), vec![1, 2, 3]); // unchanged
        assert_eq!(sk.peek(), Some(&3));
    }

    #[test]
    fn growing_until_full_then_enforcing_policy() {
        let mut sk = SmallestK::new(4);
        for x in [7, 3, 9] {
            sk.insert(x);
        }
        assert_eq!(sk.len(), 3);
        assert_eq!(contents_sorted(&sk), vec![3, 7, 9]);
        sk.insert(1); // now full
        assert_eq!(sk.len(), 4);
        assert_eq!(contents_sorted(&sk), vec![1, 3, 7, 9]);

        // Now apply policy: only < threshold (currently 9) gets in
        sk.insert(8);
        assert_eq!(contents_sorted(&sk), vec![1, 3, 7, 8]);
        sk.insert(100); // rejected
        assert_eq!(contents_sorted(&sk), vec![1, 3, 7, 8]);
    }

    #[test]
    fn deterministic_property_matches_sorted_take_k() {
        let cap = 50usize;
        let mut sk = SmallestK::new(cap);
        let mut all = Vec::with_capacity(10_000);

        // Generate a non-trivial sequence without randomness:
        // e.g. x_n = (n * 37) mod 10_000  (full period modulo arithmetic progression).
        let modulus = 10_000i32;
        let step = 37i32;
        let mut x = 0i32;
        for _ in 0..10_000 {
            all.push(x);
            sk.insert(x);
            x = (x + step) % modulus;
        }

        // Reference: global sort then take k smallest
        all.sort_unstable();
        let expected: Vec<_> = all.into_iter().take(cap).collect();
        assert_eq!(contents_sorted(&sk), expected);
        assert_eq!(sk.len(), cap);
        assert_eq!(sk.peek().copied(), expected.last().copied()); // threshold equals k-th smallest
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _ = SmallestK::<i32>::new(0);
    }
}
