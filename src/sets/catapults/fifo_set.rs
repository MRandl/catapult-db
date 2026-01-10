use std::collections::VecDeque;

use crate::sets::catapults::CatapultEvictingStructure;

pub struct FifoSet<const CAPACITY: usize> {
    queue: VecDeque<usize>,
}

impl<const CAPACITY: usize> FifoSet<CAPACITY> {
    pub fn new() -> Self {
        assert!(CAPACITY > 0);
        FifoSet {
            queue: VecDeque::with_capacity(CAPACITY),
        }
    }
}

impl<const CAPACITY: usize> Default for FifoSet<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> CatapultEvictingStructure for FifoSet<CAPACITY> {
    fn to_vec(&self) -> Vec<usize> {
        self.queue.iter().copied().collect()
    }

    fn insert(&mut self, key: usize) {
        // Remove any existing occurrence of the key to maintain set behavior
        if let Some(pos) = self.queue.iter().position(|&x| x == key) {
            self.queue.remove(pos);
        }

        // If at capacity, evict the oldest element
        if self.queue.len() == CAPACITY {
            self.queue.pop_front();
        }

        // Insert the new element at the back
        self.queue.push_back(key);
    }

    fn new() -> Self {
        FifoSet {
            queue: VecDeque::new(),
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
    }
}

impl<const CAPACITY: usize> std::fmt::Debug for FifoSet<CAPACITY> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FifoSet")
            .field("capacity", &CAPACITY)
            .field("queue", &self.queue)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_fifo_set() {
        let fifo = FifoSet::<5>::new();
        assert_eq!(fifo.queue.len(), 0);
    }

    #[test]
    fn insert_single_element() {
        let mut fifo = FifoSet::<3>::new();
        fifo.insert(10);
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0], 10);
    }

    #[test]
    fn insert_up_to_capacity() {
        let mut fifo = FifoSet::<3>::new();
        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 1);
        assert_eq!(fifo.queue[1], 2);
        assert_eq!(fifo.queue[2], 3);
    }

    #[test]
    fn insert_beyond_capacity_evicts_oldest() {
        let mut fifo = FifoSet::<3>::new();
        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);
        fifo.insert(4); // Should evict 1

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 2);
        assert_eq!(fifo.queue[1], 3);
        assert_eq!(fifo.queue[2], 4);
    }

    #[test]
    fn fifo_order_maintained_over_multiple_evictions() {
        let mut fifo = FifoSet::<3>::new();

        // Fill to capacity
        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);

        // Insert more, triggering multiple evictions
        fifo.insert(4); // Evicts 1, queue: [2, 3, 4]
        fifo.insert(5); // Evicts 2, queue: [3, 4, 5]
        fifo.insert(6); // Evicts 3, queue: [4, 5, 6]

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 4);
        assert_eq!(fifo.queue[1], 5);
        assert_eq!(fifo.queue[2], 6);
    }

    #[test]
    fn capacity_one_always_contains_last_inserted() {
        let mut fifo = FifoSet::<1>::new();

        fifo.insert(10);
        assert_eq!(fifo.queue[0], 10);

        fifo.insert(20);
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0], 20);

        fifo.insert(30);
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0], 30);
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _fifo = FifoSet::<0>::new();
    }

    #[test]
    fn large_capacity_behaves_correctly() {
        let mut fifo = FifoSet::<1000>::new();

        // Insert 500 elements
        for i in 0..500 {
            fifo.insert(i);
        }

        assert_eq!(fifo.queue.len(), 500);
        assert_eq!(fifo.queue[0], 0);
        assert_eq!(fifo.queue[499], 499);
    }

    #[test]
    fn duplicate_values_are_deduplicated_keeping_last() {
        let mut fifo = FifoSet::<5>::new();

        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);
        fifo.insert(1); // Should remove old 1 and add at end

        // Should have [2, 3, 1] - only one occurrence of 1 at the end
        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 2);
        assert_eq!(fifo.queue[1], 3);
        assert_eq!(fifo.queue[2], 1);
    }

    #[test]
    fn insert_sequence_with_eviction_maintains_order() {
        let mut fifo = FifoSet::<4>::new();

        // Insert sequence: 10, 20, 30, 40, 50, 60
        for val in [10, 20, 30, 40, 50, 60] {
            fifo.insert(val);
        }

        // Should contain last 4: [30, 40, 50, 60]
        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0], 30);
        assert_eq!(fifo.queue[1], 40);
        assert_eq!(fifo.queue[2], 50);
        assert_eq!(fifo.queue[3], 60);
    }

    #[test]
    fn alternating_insert_pattern() {
        let mut fifo = FifoSet::<2>::new();

        fifo.insert(1);
        fifo.insert(2);
        assert_eq!(fifo.queue.len(), 2);

        fifo.insert(3); // Evicts 1
        assert_eq!(fifo.queue[0], 2);
        assert_eq!(fifo.queue[1], 3);

        fifo.insert(4); // Evicts 2
        assert_eq!(fifo.queue[0], 3);
        assert_eq!(fifo.queue[1], 4);
    }

    #[test]
    fn stress_test_many_insertions() {
        let mut fifo = FifoSet::<100>::new();

        // Insert 10,000 elements
        for i in 0..10_000 {
            fifo.insert(i);
        }

        // Should contain last 100: [9900..10000)
        assert_eq!(fifo.queue.len(), 100);
        assert_eq!(fifo.queue[0], 9900);
        assert_eq!(fifo.queue[99], 9999);
    }

    #[test]
    fn test_debug() {
        let mut fifo = FifoSet::<3>::new();
        fifo.insert(1);
        fifo.insert(2);

        let debug_str = format!("{fifo:?}");
        assert_eq!(debug_str, "FifoSet { capacity: 3, queue: [1, 2] }");
    }

    #[test]
    fn test_default() {
        let fifo: FifoSet<5> = FifoSet::default();
        assert_eq!(fifo.queue.len(), 0);
        assert_eq!(fifo.queue.capacity(), 5);
    }

    #[test]
    fn duplicate_insertion_maintains_set_property() {
        let mut fifo = FifoSet::<3>::new();

        fifo.insert(1);
        fifo.insert(1);

        // Should only have one element
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0], 1);
    }

    #[test]
    fn duplicate_with_full_capacity_maintains_set_behavior() {
        let mut fifo = FifoSet::<3>::new();

        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);
        // Queue: [1, 2, 3]

        fifo.insert(2); // Should remove 2 from middle, add at end
        // Queue: [1, 3, 2]

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 1);
        assert_eq!(fifo.queue[1], 3);
        assert_eq!(fifo.queue[2], 2);
    }

    #[test]
    fn repeated_duplicates_maintain_fifo_order() {
        let mut fifo = FifoSet::<4>::new();

        fifo.insert(1);
        fifo.insert(2);
        fifo.insert(3);
        fifo.insert(4);
        // Queue: [1, 2, 3, 4]

        fifo.insert(2); // Remove 2 from position 1, add at end
        // Queue: [1, 3, 4, 2]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0], 1);
        assert_eq!(fifo.queue[1], 3);
        assert_eq!(fifo.queue[2], 4);
        assert_eq!(fifo.queue[3], 2);

        fifo.insert(3); // Remove 3 from position 1, add at end
        // Queue: [1, 4, 2, 3]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0], 1);
        assert_eq!(fifo.queue[1], 4);
        assert_eq!(fifo.queue[2], 2);
        assert_eq!(fifo.queue[3], 3);

        fifo.insert(5); // New element, evict 1
        // Queue: [4, 2, 3, 5]

        assert_eq!(fifo.queue.len(), 4);
        assert_eq!(fifo.queue[0], 4);
        assert_eq!(fifo.queue[1], 2);
        assert_eq!(fifo.queue[2], 3);
        assert_eq!(fifo.queue[3], 5);
    }

    #[test]
    fn set_behavior_no_duplicates_in_final_state() {
        let mut fifo = FifoSet::<5>::new();

        fifo.insert(10);
        fifo.insert(20);
        fifo.insert(10);
        fifo.insert(30);
        fifo.insert(20);

        // Should have [10, 30, 20] - no duplicates
        assert_eq!(fifo.queue.len(), 3);

        // Verify no duplicates
        let vec = fifo.to_vec();
        let mut sorted = vec.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(vec.len(), sorted.len(), "Should have no duplicates");
    }
}
