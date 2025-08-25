/// A fixed-capacity set of boolean values packed into a contiguous
/// buffer of bytes.
///
/// Each bit can be individually set or queried.
///
/// # Examples
///
/// ```
/// use catapult::BitSet;
///
/// let mut bs = BitSet::new(10);
/// assert!(!bs.get_bit(3));
///
/// bs.set_bit(3);
/// assert!(bs.get_bit(3));
/// ```
pub struct BitSet {
    /*private*/ buffer: Box<[u8]>,
    /*private*/ capacity: usize,
}

impl BitSet {
    /// Constructs a new [`BitSet`] with space for `capacity` bits,
    /// all initialized to zero.
    ///
    /// # Examples
    /// ```
    /// let bs = BitSet::new(12);
    /// assert!(!bs.get_bit(0));
    /// ```
    pub fn new(capacity: usize) -> Self {
        let bytes_needed: usize = capacity.div_ceil(8);
        BitSet {
            buffer: vec![0u8; bytes_needed].into_boxed_slice(),
            capacity,
        }
    }

    /// Sets the bit at the given `index` to `1`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= capacity`.
    ///
    /// # Examples
    /// ```
    /// let mut bs = BitSet::new(4);
    /// bs.set_bit(2);
    /// assert!(bs.get_bit(2));
    /// ```
    pub fn set_bit(&mut self, index: usize) {
        assert!(index < self.capacity);

        let byte_index = index / 8;
        let bit_index = index % 8;

        self.buffer[byte_index] |= 1u8 << bit_index
    }

    /// Returns `true` if the bit at `index` is set.
    ///
    /// # Panics
    ///
    /// Panics if `index >= capacity`.
    ///
    /// # Examples
    /// ```
    /// let mut bs = BitSet::new(4);
    /// bs.set_bit(1);
    /// assert!(bs.get_bit(1));
    /// assert!(!bs.get_bit(0));
    /// ```
    pub fn get_bit(&self, index: usize) -> bool {
        assert!(index < self.capacity);

        let byte_index = index / 8;
        let bit_index = index % 8;

        self.buffer[byte_index] & (1u8 << bit_index) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_capacity_constructs() {
        // Just ensure it doesn't panic.
        let _bs = BitSet::new(0);
    }

    #[test]
    fn all_bits_start_cleared() {
        for cap in [1usize, 7, 8, 9, 16, 31, 32, 33] {
            let bs = BitSet::new(cap);
            for i in 0..cap {
                assert!(
                    !bs.get_bit(i),
                    "bit {} should start cleared for cap {}",
                    i,
                    cap
                );
            }
        }
    }

    #[test]
    fn set_and_get_single_bits_across_byte_boundaries() {
        let cap = 40; // >= 5 bytes
        let mut bs = BitSet::new(cap);

        // Set a bunch of positions, including boundaries
        let to_set = [0usize, 1, 7, 8, 15, 16, 31, 32, 39];
        for &i in &to_set {
            bs.set_bit(i);
            assert!(bs.get_bit(i), "bit {} should be set", i);
        }

        // Verify every position: set ones are 1, others are 0.
        let mut expected = vec![false; cap];
        for &i in &to_set {
            expected[i] = true;
        }

        for i in 0..cap {
            assert_eq!(
                bs.get_bit(i),
                expected[i],
                "bit {} expected {}, found {}",
                i,
                expected[i],
                bs.get_bit(i)
            );
        }
    }

    #[test]
    fn idempotent_sets() {
        let mut bs = BitSet::new(10);
        bs.set_bit(3);
        bs.set_bit(3);
        assert!(bs.get_bit(3));
        // Other bits unaffected
        for i in 0..10 {
            if i != 3 {
                assert!(!bs.get_bit(i));
            }
        }
    }

    #[test]
    fn non_multiple_of_8_capacity_last_bit_works() {
        // Capacity 10 => 2 bytes allocated, last valid index = 9
        let mut bs = BitSet::new(10);
        bs.set_bit(9);
        assert!(bs.get_bit(9));
        // Earlier bits still cleared
        for i in 0..9 {
            assert!(!bs.get_bit(i));
        }
    }

    #[test]
    fn dense_and_sparse_patterns() {
        let cap = 64;
        let mut bs = BitSet::new(cap);

        // Sparse: every 3rd bit
        for i in (0..cap).step_by(3) {
            bs.set_bit(i);
        }
        for i in 0..cap {
            let expected = i % 3 == 0;
            assert_eq!(bs.get_bit(i), expected, "mismatch at {}", i);
        }

        // Overwrite to dense: now set all bits
        for i in 0..cap {
            bs.set_bit(i);
        }
        for i in 0..cap {
            assert!(bs.get_bit(i));
        }
    }

    #[test]
    fn last_index_is_valid() {
        for cap in [1usize, 8, 9, 17, 31, 32, 33] {
            let last = cap - 1;
            let mut bs = BitSet::new(cap);
            bs.set_bit(last);
            assert!(
                bs.get_bit(last),
                "last bit {} should be set for cap {}",
                last,
                cap
            );
        }
    }

    #[test]
    #[should_panic]
    fn set_bit_out_of_bounds_panics_in_debug() {
        // capacity = 10 → valid indices are 0..9
        let mut bs = BitSet::new(10);
        bs.set_bit(10); // invalid
    }
}
