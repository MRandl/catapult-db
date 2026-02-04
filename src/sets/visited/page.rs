/// The number of bits stored in a single page (4096 bits = 512 bytes).
pub const PAGE_SIZE_BITS: usize = 4096;

/// The number of u64 words needed to store PAGE_SIZE_BITS (4096 / 64 = 64).
pub const PAGE_SIZE_U64: usize = PAGE_SIZE_BITS / 64;

/// A fixed-size bitmap page storing 4096 bits using 64 u64 words.
///
/// This structure provides efficient bit manipulation for tracking boolean flags
/// in a contiguous block of memory. Each page can track 4096 distinct indices.
///
/// # Memory Layout
/// - Total size: 512 bytes (64 u64 words × 8 bytes)
/// - Capacity: 4096 bits
/// - Bit indexing: bit `n` is stored at `bits[n/64]` with mask `1 << (n%64)`
pub struct Page {
    bits: [u64; PAGE_SIZE_U64],
}

impl Page {
    /// Creates a new page with all bits initialized to zero.
    ///
    /// # Returns
    /// A new `Page` with all 4096 bits cleared
    pub fn new() -> Self {
        Self {
            bits: [0; PAGE_SIZE_U64],
        }
    }

    /// Sets the bit at the given offset to 1.
    ///
    /// # Arguments
    /// * `offset` - The bit index to set (must be < 4096)
    ///
    /// # Panics
    /// May panic or produce incorrect results if `offset >= PAGE_SIZE_BITS`
    #[inline]
    pub fn set(&mut self, offset: usize) {
        self.bits[offset / 64] |= 1 << (offset % 64);
    }

    /// Returns whether the bit at the given offset is set.
    ///
    /// # Arguments
    /// * `offset` - The bit index to check (must be < 4096)
    ///
    /// # Returns
    /// `true` if the bit is set, `false` otherwise
    ///
    /// # Panics
    /// May panic or produce incorrect results if `offset >= PAGE_SIZE_BITS`
    #[inline]
    pub fn get(&self, offset: usize) -> bool {
        (self.bits[offset / 64] & (1 << (offset % 64))) != 0
    }

    /// Returns the count of set bits in this page.
    ///
    /// # Returns
    /// The number of bits set to 1 (between 0 and 4096)
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }
}

impl Default for Page {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_page_all_bits_cleared() {
        let page = Page::new();
        for i in 0..PAGE_SIZE_BITS {
            assert!(!page.get(i), "bit {i} should be cleared on new page");
        }
        let page_default = Page::default();
        for i in 0..PAGE_SIZE_BITS {
            assert!(!page_default.get(i));
        }
    }

    #[test]
    fn set_and_get_single_bit() {
        let mut page = Page::new();
        page.set(42);
        assert!(page.get(42));
        assert!(!page.get(41));
        assert!(!page.get(43));
        assert_eq!(page.len(), 1);
    }

    #[test]
    fn set_bits_across_u64_boundaries() {
        let mut page = Page::new();

        // Test boundaries between u64 chunks
        let boundary_positions = [0, 63, 64, 127, 128, 191, 192, PAGE_SIZE_BITS - 1];

        for &pos in &boundary_positions {
            page.set(pos);
            assert!(page.get(pos), "bit {pos} should be set");
        }

        // Verify all set positions
        for i in 0..PAGE_SIZE_BITS {
            let expected = boundary_positions.contains(&i);
            assert_eq!(page.get(i), expected, "bit {i} mismatch");
        }
    }

    #[test]
    fn idempotent_set_operations() {
        let mut page = Page::new();
        page.set(100);
        page.set(100);
        page.set(100);
        assert!(page.get(100));

        // Verify other bits unaffected
        assert!(!page.get(99));
        assert!(!page.get(101));
    }

    #[test]
    fn dense_bit_pattern() {
        let mut page = Page::new();

        // Set all bits in first u64
        for i in 0..64 {
            page.set(i);
        }

        // Verify first u64 all set
        for i in 0..64 {
            assert!(page.get(i), "bit {i} should be set");
        }

        // Verify rest still cleared
        for i in 64..PAGE_SIZE_BITS {
            assert!(!page.get(i), "bit {i} should be cleared");
        }
    }

    #[test]
    fn sparse_bit_pattern() {
        let mut page = Page::new();

        // Set every 100th bit
        for i in (0..PAGE_SIZE_BITS).step_by(100) {
            page.set(i);
        }

        for i in 0..PAGE_SIZE_BITS {
            let expected = i % 100 == 0;
            assert_eq!(page.get(i), expected, "bit {i} mismatch");
        }
    }

    #[test]
    fn set_all_bits_in_page() {
        let mut page = Page::new();

        for i in 0..PAGE_SIZE_BITS {
            page.set(i);
        }

        for i in 0..PAGE_SIZE_BITS {
            assert!(page.get(i));
        }
    }

    #[test]
    fn last_bit_in_page() {
        let mut page = Page::new();
        let last_bit = PAGE_SIZE_BITS - 1;

        page.set(last_bit);
        assert!(page.get(last_bit));
        assert!(!page.get(last_bit - 1));
    }

    #[test]
    fn first_bit_in_each_u64_chunk() {
        let mut page = Page::new();

        // Set first bit of each u64 chunk
        for chunk_idx in 0..PAGE_SIZE_U64 {
            page.set(chunk_idx * 64);
        }

        for chunk_idx in 0..PAGE_SIZE_U64 {
            let bit_pos = chunk_idx * 64;
            assert!(page.get(bit_pos));

            // Verify next bit in chunk is clear
            assert!(!page.get(bit_pos + 1))
        }
    }

    #[test]
    fn last_bit_in_each_u64_chunk() {
        let mut page = Page::new();

        // Set last bit of each u64 chunk
        for chunk_idx in 0..PAGE_SIZE_U64 {
            page.set(chunk_idx * 64 + 63);
        }

        for chunk_idx in 0..PAGE_SIZE_U64 {
            let bit_pos = chunk_idx * 64 + 63;
            assert!(page.get(bit_pos));

            // Verify previous bit is clear
            assert!(!page.get(bit_pos - 1));
        }
    }
}
