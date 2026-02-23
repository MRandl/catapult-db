use std::collections::BTreeMap;
use std::ops::Bound;

use crate::search::NodeId;

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
}

impl ZOrderIndex {
    pub fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
        }
    }

    /// Insert `node` under `signature`. Multiple nodes may share the same signature.
    pub fn insert(&mut self, signature: u128, node: NodeId) {
        self.tree.entry(signature).or_default().push(node);
    }

    /// Return the LLCP (number of shared leading bits) between two signatures.
    pub fn llcp(a: u128, b: u128) -> u32 {
        (a ^ b).leading_zeros()
    }

    /// Return up to `k` [`NodeId`]s whose signatures have the greatest LLCP with `target`.
    ///
    /// The search starts at the first key >= `target` and expands bidirectionally,
    /// always consuming whichever side (left or right) has the better (longer) LLCP
    /// with `target` next. Ties are broken by taking the right side first.
    ///
    /// Returns fewer than `k` entries if the index contains fewer.
    pub fn query_k_closest(&self, target: u128, k: usize) -> Vec<NodeId> {
        if k == 0 || self.tree.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(k);

        // Build two iterators: one going right (>= target), one going left (< target).
        let mut right = self
            .tree
            .range((Bound::Included(target), Bound::Unbounded))
            .peekable();
        let mut left = self
            .tree
            .range((Bound::Unbounded, Bound::Excluded(target)))
            .rev()
            .peekable();

        loop {
            if result.len() >= k {
                break;
            }

            let right_llcp = right.peek().map(|&(&sig, _)| Self::llcp(target, sig));
            let left_llcp = left.peek().map(|&(&sig, _)| Self::llcp(target, sig));

            // Pick whichever side has the longer common prefix; prefer right on tie.
            let take_right = match (right_llcp, left_llcp) {
                (None, None) => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(r), Some(l)) => r >= l,
            };

            let iter: &mut dyn Iterator<Item = (&u128, &Vec<NodeId>)> = if take_right {
                &mut right
            } else {
                &mut left
            };

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
}

impl Default for ZOrderIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: usize) -> NodeId {
        NodeId { internal: id }
    }

    #[test]
    fn test_empty_index_returns_empty() {
        let idx = ZOrderIndex::new();
        assert!(idx.query_k_closest(42, 5).is_empty());
    }

    #[test]
    fn test_k_zero_returns_empty() {
        let mut idx = ZOrderIndex::new();
        idx.insert(10, node(1));
        assert!(idx.query_k_closest(10, 0).is_empty());
    }

    #[test]
    fn test_single_entry_always_returned() {
        let mut idx = ZOrderIndex::new();
        idx.insert(0x1234, node(7));
        let result = idx.query_k_closest(0x1234, 3);
        assert_eq!(result, vec![node(7)]);
    }

    #[test]
    fn test_exact_match_returned_first() {
        let mut idx = ZOrderIndex::new();
        idx.insert(100, node(1));
        idx.insert(200, node(2));
        idx.insert(100, node(3)); // second node at same key
        let result = idx.query_k_closest(100, 5);
        // Both nodes at key 100 should come back, and before node 2
        assert!(result.contains(&node(1)));
        assert!(result.contains(&node(3)));
        let pos1 = result.iter().position(|&n| n == node(1)).unwrap();
        let pos2 = result.iter().position(|&n| n == node(2)).unwrap();
        assert!(pos1 < pos2);
        let pos3 = result.iter().position(|&n| n == node(3)).unwrap();
        assert!(pos3 < pos2);
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
    fn test_closer_signature_preferred() {
        // target = 0b1000, near = 0b1001 (LLCP 127), far = 0b0000 (LLCP 0)
        let target: u128 = 0b1000;
        let near: u128 = 0b1001;
        let far: u128 = 0b0000;

        let mut idx = ZOrderIndex::new();
        idx.insert(near, node(1));
        idx.insert(far, node(2));

        let result = idx.query_k_closest(target, 1);
        assert_eq!(result, vec![node(1)]);
    }

    #[test]
    fn test_returns_at_most_k_results() {
        let mut idx = ZOrderIndex::new();
        for i in 0..20u128 {
            idx.insert(i, node(i as usize));
        }
        let result = idx.query_k_closest(10, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_returns_all_when_fewer_than_k() {
        let mut idx = ZOrderIndex::new();
        idx.insert(1, node(1));
        idx.insert(2, node(2));
        let result = idx.query_k_closest(1, 100);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_bidirectional_expansion() {
        // target = 8; keys at 7 (left) and 9 (right) both have LLCP 125 with 8.
        // Both should appear in a k=2 query before more distant keys.
        let target: u128 = 8;
        let mut idx = ZOrderIndex::new();
        idx.insert(7, node(7));
        idx.insert(9, node(9));
        idx.insert(0, node(0)); // far left
        idx.insert(255, node(255)); // far right

        let result = idx.query_k_closest(target, 2);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&node(7)) || result.contains(&node(9)));
        assert!(!result.contains(&node(0)));
        assert!(!result.contains(&node(255)));
    }

    #[test]
    fn test_target_not_in_index_still_works() {
        let mut idx = ZOrderIndex::new();
        idx.insert(10, node(10));
        idx.insert(20, node(20));
        // target 15 is between them
        let result = idx.query_k_closest(15, 2);
        assert_eq!(result.len(), 2);
    }
}
