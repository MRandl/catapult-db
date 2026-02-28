use std::collections::BTreeMap;
use std::ops::Bound;

use crate::numerics::AlignedBlock;
use crate::search::NodeId;
use crate::search::hash_start::pstable_hasher::PStableHashingBlock;

/// An ordered index of Z-order (Morton-coded) u128 hash signatures mapping to node IDs.
///
/// Insertions store one or more [`NodeId`]s under a given signature key. Queries find the
/// `k` entries whose signatures have the longest common bit-prefix (LLCP) with a target
/// signature, using bidirectional expansion from the target's position in the ordered tree.
///
/// # LLCP and Z-order
///
/// The Longest Common Prefix length between two u128 keys is `u128::leading_zeros(a ^ b)`.
/// Keys with a longer common prefix are considered closer. Bidirectional expansion from
/// the target's successor in sorted order visits keys in roughly LLCP-descending order
/// because a longer shared prefix implies a smaller absolute difference and vice versa —
/// not perfectly, but it is the standard approximation used in LSHAPG-style indices.
pub struct ZOrderIndex {
    tree: BTreeMap<u128, Vec<NodeId>>,
    hasher: PStableHashingBlock,
}

pub const LSH_APG_REDUNDANCY: usize = 1;

impl ZOrderIndex {
    pub fn new(num_hash: usize, stored_vectors_dim: usize, seed: u64, w: f32) -> Self {
        Self {
            tree: BTreeMap::new(),
            hasher: PStableHashingBlock::new_seeded(num_hash, stored_vectors_dim, seed, w),
        }
    }

    /// Insert `node` under `signature`. Multiple nodes may share the same signature.
    pub fn insert(&mut self, key: &[AlignedBlock], node: NodeId) {
        let target_signature = self.hasher.hash(key);
        self.tree.entry(target_signature).or_default().push(node);
    }

    /// Return the LLCP (number of shared leading bits) between two signatures.
    pub fn llcp(a: u128, b: u128) -> u32 {
        (a ^ b).leading_zeros()
    }

    /// Return up to `k` [`NodeId`]s whose signatures have the greatest LLCP with `target_signature`.
    ///
    /// The search starts at the first key >= `target_signature` and expands bidirectionally,
    /// always consuming whichever side (left or right) has the better (longer) LLCP
    /// with `target_signature` next. Ties are broken by taking the right side first.
    ///
    /// Returns fewer than `k` entries if the index contains fewer.
    fn query_k_closest_by_signature(&self, target_signature: u128, k: usize) -> Vec<NodeId> {
        if k == 0 || self.tree.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(k);

        // Build two iterators: one going right (>= target), one going left (< target).
        let mut right = self
            .tree
            .range((Bound::Included(target_signature), Bound::Unbounded))
            .peekable();
        let mut left = self
            .tree
            .range((Bound::Unbounded, Bound::Excluded(target_signature)))
            .rev()
            .peekable();

        loop {
            if result.len() >= k {
                break;
            }

            let right_llcp = right
                .peek()
                .map(|&(&sig, _)| Self::llcp(target_signature, sig));
            let left_llcp = left
                .peek()
                .map(|&(&sig, _)| Self::llcp(target_signature, sig));

            // Pick whichever side has the longer common prefix; prefer right on tie.
            let take_right = match (right_llcp, left_llcp) {
                (None, None) => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(r), Some(l)) => r >= l,
            };

            let iter: &mut dyn Iterator<Item = (&u128, &Vec<NodeId>)> =
                if take_right { &mut right } else { &mut left };

            if let Some((_, nodes)) = iter.next() {
                for &node in nodes {
                    result.push(node);
                    if result.len() >= k {
                        break;
                    }
                }
            }
        }

        result
    }

    /// Return up to `k` [`NodeId`]s whose signatures have the greatest LLCP with `target_vec`.
    ///
    /// Hashes `target_vec` and delegates to [`Self::query_k_closest_by_signature`].
    pub fn query_k_closest(&self, target_vec: &[AlignedBlock], k: usize) -> Vec<NodeId> {
        let target_signature = self.hasher.hash(target_vec);
        self.query_k_closest_by_signature(target_signature, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::AlignedBlock;

    fn node(id: usize) -> NodeId {
        NodeId { internal: id }
    }

    fn new_for_test() -> ZOrderIndex {
        ZOrderIndex::new(1, 16, 0, 1.0)
    }

    /// Build a 1-block vector with all 16 lanes set to `val`.
    fn vec_of(val: f32) -> Vec<AlignedBlock> {
        AlignedBlock::allocate_padded(vec![val; 16])
    }

    #[test]
    fn test_empty_index_returns_empty() {
        let idx = new_for_test();
        assert!(idx.query_k_closest(&vec_of(0.0), 5).is_empty());
    }

    #[test]
    fn test_k_zero_returns_empty() {
        let mut idx = new_for_test();
        idx.insert(&vec_of(1.0), node(1));
        assert!(idx.query_k_closest(&vec_of(1.0), 0).is_empty());
    }

    #[test]
    fn test_single_entry_always_returned() {
        let mut idx = new_for_test();
        let v = vec_of(1.0);
        idx.insert(&v, node(7));
        let result = idx.query_k_closest(&v, 3);
        assert_eq!(result, vec![node(7)]);
    }

    #[test]
    fn test_same_vector_inserted_twice_both_returned() {
        let mut idx = new_for_test();
        let v1 = vec_of(1.0);
        let v2 = vec_of(2.0);
        let v_query = vec_of(1.0);
        idx.insert(&v1, node(1));
        idx.insert(&v2, node(2));
        idx.insert(&v1, node(3)); // second node at same key as v1
        let result = idx.query_k_closest(&v_query, 5);
        // Both nodes at the v1 key should come back
        assert!(result.contains(&node(1)));
        assert!(result.contains(&node(3)));
    }

    #[test]
    fn test_llcp_identical_keys() {
        assert_eq!(ZOrderIndex::llcp(u128::MAX, u128::MAX), 128);
        assert_eq!(ZOrderIndex::llcp(0, 0), 128);
    }

    #[test]
    fn test_llcp_totally_different_keys() {
        // High bit differs
        assert_eq!(ZOrderIndex::llcp(0, u128::MAX), 0);
        assert_eq!(ZOrderIndex::llcp(1u128 << 127, 0), 0);
    }

    #[test]
    fn test_llcp_partial_match() {
        // Keys share top 4 bits (0xF... vs 0xE...)
        let a = 0xF000_0000_0000_0000_0000_0000_0000_0000u128;
        let b = 0xE000_0000_0000_0000_0000_0000_0000_0000u128;
        assert_eq!(ZOrderIndex::llcp(a, b), 3);
    }

    #[test]
    fn test_returns_at_most_k_results() {
        let mut idx = new_for_test();
        for i in 0..20 {
            idx.insert(&vec_of(i as f32), node(i));
        }
        let result = idx.query_k_closest(&vec_of(10.0), 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_returns_all_when_fewer_than_k() {
        let mut idx = new_for_test();
        idx.insert(&vec_of(1.0), node(1));
        idx.insert(&vec_of(2.0), node(2));
        let result = idx.query_k_closest(&vec_of(1.0), 100);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_query_vector_not_in_index_still_works() {
        let mut idx = new_for_test();
        idx.insert(&vec_of(1.0), node(1));
        idx.insert(&vec_of(2.0), node(2));
        // query with a vector not inserted
        let result = idx.query_k_closest(&vec_of(1.5), 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    pub fn test_1000_index() {
        let mut idx = ZOrderIndex::new(10, 16, 42, 1.0);

        for i in 0..10000 {
            idx.insert(&[AlignedBlock::new([i as f32; 16])], NodeId { internal: i });
        }

        let queried = idx.query_k_closest(&[AlignedBlock::new([5000.5; 16])], 5);

        // looking for the five closest known vectors in the area of 5000.5 should return a bunch
        // of vectors, including id 5000 and 5001
        assert!(queried.contains(&NodeId { internal: 5000 }));
        assert!(queried.contains(&NodeId { internal: 5001 }));
    }
}
