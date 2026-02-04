/// A trait for tracking visited nodes during graph traversal.
///
/// Provides a simple boolean flag interface for marking nodes as visited and
/// checking their visited status. Implementations may use different backing
/// structures (hash sets, bit vectors, etc.) with varying performance characteristics.
pub trait VisitorSet {
    /// Checks whether a node has been visited.
    ///
    /// # Arguments
    /// * `i` - The node index to check
    ///
    /// # Returns
    /// `true` if the node has been marked as visited, `false` otherwise
    fn get(&self, i: usize) -> bool;

    /// Marks a node as visited.
    ///
    /// # Arguments
    /// * `i` - The node index to mark as visited
    fn set(&mut self, i: usize);
}
