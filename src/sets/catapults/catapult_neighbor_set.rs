use crate::search::NodeId;

/// A trait for data structures that store catapult node indices with eviction policies.
///
/// Catapult structures maintain a bounded set of node indices that serve as cached
/// starting points for graph searches. When capacity is reached, implementations
/// must evict entries according to their specific policy (e.g., FIFO, LRU).
pub trait CatapultEvictingStructure {
    /// Inserts a node index into the catapult structure.
    ///
    /// If the structure is at capacity, the implementation's eviction policy determines
    /// which existing entry to remove. Inserting a duplicate may remove the old occurrence
    /// and add it fresh (implementation-specific).
    ///
    /// # Arguments
    /// * `neighbor` - The node index to insert as a catapult
    fn insert(&mut self, neighbor: NodeId);

    /// Creates a new empty catapult structure.
    ///
    /// # Returns
    /// A new empty instance of the implementing type
    fn new() -> Self;

    /// Returns all currently stored catapult node indices as a vector.
    ///
    /// The order may be significant depending on the implementation (e.g., FIFO order).
    ///
    /// # Returns
    /// A vector containing all stored node indices
    fn to_vec(&self) -> Vec<NodeId>;

    /// Removes all stored catapults, resetting the structure to empty.
    fn clear(&mut self);
}
