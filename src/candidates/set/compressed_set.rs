use crate::candidates::set::{
    integer_map::IntegerMap,
    page::{PAGE_SIZE_BITS, Page},
    visitor_set::VisitorSet,
};

#[derive(Default)]
pub struct CompressedBitset {
    pages: IntegerMap<Box<Page>>,
}

impl CompressedBitset {
    pub fn new() -> Self {
        Self {
            pages: IntegerMap::default(),
        }
    }
}

impl VisitorSet for CompressedBitset {
    fn set(&mut self, index: usize) {
        let page_idx = index / PAGE_SIZE_BITS;
        let bit_offset = index % PAGE_SIZE_BITS;

        // Because we use NoOpHasher, this lookup is now essentially just a memory offset calc
        let page = self
            .pages
            .entry(page_idx)
            .or_insert_with(|| Box::new(Page::new()));

        page.set(bit_offset);
    }

    fn get(&self, index: usize) -> bool {
        let page_idx = index / PAGE_SIZE_BITS;
        let bit_offset = index % PAGE_SIZE_BITS;

        match self.pages.get(&page_idx) {
            Some(page) => page.get(bit_offset),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bitset_is_empty() {
        let bitset = CompressedBitset::new();

        // Check various indices are all false
        for i in [0, 1, 100, 1000, 10000, 100000] {
            assert!(!bitset.get(i), "index {} should be false in new bitset", i);
        }
    }

    #[test]
    fn set_and_get_single_bit() {
        let mut bitset = CompressedBitset::new();

        bitset.set(42);
        assert!(bitset.get(42));
        assert!(!bitset.get(41));
        assert!(!bitset.get(43));
    }

    #[test]
    fn set_bits_in_same_page() {
        let mut bitset = CompressedBitset::new();

        // Set multiple bits within first page (indices < PAGE_SIZE_BITS)
        bitset.set(0);
        bitset.set(100);
        bitset.set(1000);
        bitset.set(PAGE_SIZE_BITS - 1);

        assert!(bitset.get(0));
        assert!(bitset.get(100));
        assert!(bitset.get(1000));
        assert!(bitset.get(PAGE_SIZE_BITS - 1));

        // Other bits in same page should be false
        assert!(!bitset.get(1));
        assert!(!bitset.get(101));
        assert!(!bitset.get(1001));
    }

    #[test]
    fn set_bits_across_multiple_pages() {
        let mut bitset = CompressedBitset::new();

        // Set bits in different pages
        let indices = [
            0,                         // Page 0
            PAGE_SIZE_BITS - 1,        // Last bit of page 0
            PAGE_SIZE_BITS,            // First bit of page 1
            PAGE_SIZE_BITS * 2,        // First bit of page 2
            PAGE_SIZE_BITS * 10 + 500, // Page 10
            PAGE_SIZE_BITS * 100,      // Page 100
        ];

        for &idx in &indices {
            bitset.set(idx);
        }

        // Verify all set bits
        for &idx in &indices {
            assert!(bitset.get(idx), "index {} should be set", idx);
        }

        // Verify some unset bits
        assert!(!bitset.get(1));
        assert!(!bitset.get(PAGE_SIZE_BITS + 1));
        assert!(!bitset.get(PAGE_SIZE_BITS * 50));
    }

    #[test]
    fn idempotent_set_operations() {
        let mut bitset = CompressedBitset::new();

        bitset.set(1000);
        bitset.set(1000);
        bitset.set(1000);

        assert!(bitset.get(1000));
        assert!(!bitset.get(999));
        assert!(!bitset.get(1001));
    }

    #[test]
    fn page_boundary_transitions() {
        let mut bitset = CompressedBitset::new();

        // Test bits right at page boundaries
        bitset.set(PAGE_SIZE_BITS - 1); // Last bit of page 0
        bitset.set(PAGE_SIZE_BITS); // First bit of page 1
        bitset.set(PAGE_SIZE_BITS + 1); // Second bit of page 1

        assert!(bitset.get(PAGE_SIZE_BITS - 1));
        assert!(bitset.get(PAGE_SIZE_BITS));
        assert!(bitset.get(PAGE_SIZE_BITS + 1));

        assert!(!bitset.get(PAGE_SIZE_BITS - 2));
        assert!(!bitset.get(PAGE_SIZE_BITS + 2));
    }

    #[test]
    fn sparse_page_allocation() {
        let mut bitset = CompressedBitset::new();

        // Set bits in very distant pages to test sparse allocation
        bitset.set(0);
        bitset.set(PAGE_SIZE_BITS * 1000);
        bitset.set(PAGE_SIZE_BITS * 1000000);

        assert!(bitset.get(0));
        assert!(bitset.get(PAGE_SIZE_BITS * 1000));
        assert!(bitset.get(PAGE_SIZE_BITS * 1000000));

        // Pages in between should not be allocated (get returns false)
        assert!(!bitset.get(PAGE_SIZE_BITS * 500));
        assert!(!bitset.get(PAGE_SIZE_BITS * 999));
        assert!(!bitset.get(PAGE_SIZE_BITS * 1001));
    }

    #[test]
    fn large_indices() {
        let mut bitset = CompressedBitset::new();

        let large_idx = usize::MAX / 2;
        bitset.set(large_idx);

        assert!(bitset.get(large_idx));
        assert!(!bitset.get(large_idx - 1));
        assert!(!bitset.get(large_idx + 1));
    }

    #[test]
    fn dense_bit_pattern_single_page() {
        let mut bitset = CompressedBitset::new();

        // Set many bits in a single page
        for i in 0..1000 {
            bitset.set(i);
        }

        for i in 0..1000 {
            assert!(bitset.get(i), "bit {} should be set", i);
        }

        // Bits beyond should be clear
        for i in 1000..2000 {
            assert!(!bitset.get(i), "bit {} should be clear", i);
        }
    }

    #[test]
    fn alternating_pages() {
        let mut bitset = CompressedBitset::new();

        // Set one bit in alternating pages
        for page_num in (0..10).step_by(2) {
            bitset.set(page_num * PAGE_SIZE_BITS);
        }

        // Verify even pages have bit set
        for page_num in (0..10).step_by(2) {
            assert!(
                bitset.get(page_num * PAGE_SIZE_BITS),
                "page {} should have bit set",
                page_num
            );
        }

        // Verify odd pages don't have that bit set
        for page_num in (1..10).step_by(2) {
            assert!(
                !bitset.get(page_num * PAGE_SIZE_BITS),
                "page {} should not have bit set",
                page_num
            );
        }
    }

    #[test]
    fn get_from_unallocated_pages_returns_false() {
        let bitset = CompressedBitset::new();

        // Getting from any page that hasn't been allocated should return false
        assert!(!bitset.get(0));
        assert!(!bitset.get(PAGE_SIZE_BITS));
        assert!(!bitset.get(PAGE_SIZE_BITS * 10));
        assert!(!bitset.get(PAGE_SIZE_BITS * 1000000));
    }

    #[test]
    fn mixed_operations_maintain_correctness() {
        let mut bitset = CompressedBitset::new();

        // Set some bits
        bitset.set(10);
        bitset.set(PAGE_SIZE_BITS + 20);
        bitset.set(PAGE_SIZE_BITS * 5 + 30);

        // Verify
        assert!(bitset.get(10));
        assert!(bitset.get(PAGE_SIZE_BITS + 20));
        assert!(bitset.get(PAGE_SIZE_BITS * 5 + 30));

        // Set more bits
        bitset.set(100);
        bitset.set(PAGE_SIZE_BITS * 5 + 40);

        // Verify all bits
        assert!(bitset.get(10));
        assert!(bitset.get(100));
        assert!(bitset.get(PAGE_SIZE_BITS + 20));
        assert!(bitset.get(PAGE_SIZE_BITS * 5 + 30));
        assert!(bitset.get(PAGE_SIZE_BITS * 5 + 40));

        // Verify unset bits
        assert!(!bitset.get(11));
        assert!(!bitset.get(PAGE_SIZE_BITS + 21));
        assert!(!bitset.get(PAGE_SIZE_BITS * 5 + 31));
    }
}
