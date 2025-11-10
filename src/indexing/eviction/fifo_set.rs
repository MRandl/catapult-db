use std::collections::VecDeque;

use crate::indexing::eviction::neighbors::{EvictionNeighborSet, NeighborSet};

pub struct FifoSet {
    cap: usize,
    queue: VecDeque<usize>,
}

impl FifoSet {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0);
        FifoSet {
            cap,
            queue: VecDeque::with_capacity(cap),
        }
    }
}

impl NeighborSet for FifoSet {
    fn as_slice(&self) -> Vec<usize> {
        self.queue.iter().copied().collect::<Vec<_>>()
    }
}

impl EvictionNeighborSet for FifoSet {
    fn insert(&mut self, key: usize) {
        if self.queue.len() == self.cap {
            self.queue.pop_front();
        }
        self.queue.push_back(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_fifo_set() {
        let fifo = FifoSet::new(5);
        assert_eq!(fifo.cap, 5);
        assert_eq!(fifo.queue.len(), 0);
    }

    #[test]
    fn insert_single_element() {
        let mut fifo = FifoSet::new(3);
        fifo.insert(10);
        assert_eq!(fifo.queue.len(), 1);
        assert_eq!(fifo.queue[0], 10);
    }

    #[test]
    fn insert_up_to_capacity() {
        let mut fifo = FifoSet::new(3);
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
        let mut fifo = FifoSet::new(3);
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
        let mut fifo = FifoSet::new(3);

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
        let mut fifo = FifoSet::new(1);

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
        let _fifo = FifoSet::new(0);
    }

    #[test]
    fn large_capacity_behaves_correctly() {
        let mut fifo = FifoSet::new(1000);

        // Insert 500 elements
        for i in 0..500 {
            fifo.insert(i);
        }

        assert_eq!(fifo.queue.len(), 500);
        assert_eq!(fifo.queue[0], 0);
        assert_eq!(fifo.queue[499], 499);
    }

    #[test]
    fn duplicate_values_are_inserted_without_deduplication() {
        let mut fifo = FifoSet::new(5);

        fifo.insert(1);
        fifo.insert(1);
        fifo.insert(1);

        assert_eq!(fifo.queue.len(), 3);
        assert_eq!(fifo.queue[0], 1);
        assert_eq!(fifo.queue[1], 1);
        assert_eq!(fifo.queue[2], 1);
    }

    #[test]
    fn insert_sequence_with_eviction_maintains_order() {
        let mut fifo = FifoSet::new(4);

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
        let mut fifo = FifoSet::new(2);

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
        let mut fifo = FifoSet::new(100);

        // Insert 10,000 elements
        for i in 0..10_000 {
            fifo.insert(i);
        }

        // Should contain last 100: [9900..10000)
        assert_eq!(fifo.queue.len(), 100);
        assert_eq!(fifo.queue[0], 9900);
        assert_eq!(fifo.queue[99], 9999);
    }
}
