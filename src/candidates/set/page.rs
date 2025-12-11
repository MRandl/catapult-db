pub const PAGE_SIZE_BITS: usize = 4096;
pub const PAGE_SIZE_U64: usize = PAGE_SIZE_BITS / 64;

pub struct Page {
    bits: [u64; PAGE_SIZE_U64],
}

impl Page {
    pub fn new() -> Self {
        Self {
            bits: [0; PAGE_SIZE_U64],
        }
    }

    #[inline]
    pub fn set(&mut self, offset: usize) {
        self.bits[offset / 64] |= 1 << (offset % 64);
    }

    #[inline]
    pub fn get(&self, offset: usize) -> bool {
        (self.bits[offset / 64] & (1 << (offset % 64))) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_page_all_bits_cleared() {
        let page = Page::new();
        for i in 0..PAGE_SIZE_BITS {
            assert!(!page.get(i), "bit {} should be cleared on new page", i);
        }
    }

    #[test]
    fn set_and_get_single_bit() {
        let mut page = Page::new();
        page.set(42);
        assert!(page.get(42));
        assert!(!page.get(41));
        assert!(!page.get(43));
    }

    #[test]
    fn set_bits_across_u64_boundaries() {
        let mut page = Page::new();

        // Test boundaries between u64 chunks
        let boundary_positions = [0, 63, 64, 127, 128, 191, 192, PAGE_SIZE_BITS - 1];

        for &pos in &boundary_positions {
            page.set(pos);
            assert!(page.get(pos), "bit {} should be set", pos);
        }

        // Verify all set positions
        for i in 0..PAGE_SIZE_BITS {
            let expected = boundary_positions.contains(&i);
            assert_eq!(page.get(i), expected, "bit {} mismatch", i);
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
            assert!(page.get(i), "bit {} should be set", i);
        }

        // Verify rest still cleared
        for i in 64..PAGE_SIZE_BITS {
            assert!(!page.get(i), "bit {} should be cleared", i);
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
            assert_eq!(page.get(i), expected, "bit {} mismatch", i);
        }
    }

    #[test]
    fn set_all_bits_in_page() {
        let mut page = Page::new();

        for i in 0..PAGE_SIZE_BITS {
            page.set(i);
        }

        for i in 0..PAGE_SIZE_BITS {
            assert!(page.get(i), "bit {} should be set", i);
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
            assert!(
                page.get(bit_pos),
                "first bit of chunk {} should be set",
                chunk_idx
            );

            // Verify other bits in chunk are clear
            if bit_pos + 1 < PAGE_SIZE_BITS {
                assert!(!page.get(bit_pos + 1));
            }
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
            assert!(
                page.get(bit_pos),
                "last bit of chunk {} should be set",
                chunk_idx
            );

            // Verify previous bit is clear
            assert!(!page.get(bit_pos - 1));
        }
    }
}
