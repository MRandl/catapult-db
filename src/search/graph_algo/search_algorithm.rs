use crate::sets::{catapults::CatapultEvictingStructure, fixed::FixedSet};

/// A trait defining the components of a graph search algorithm strategy.
///
/// This trait specifies the associated types needed for a search strategy: catapult
/// configuration, starting point selection mechanism, and neighbor set representation.
///
/// Current implementations:
/// - `FlatSearch`: Single-layer proximity graph search (e.g., DiskANN-style)
pub trait GraphSearchAlgorithm {
    /// Configuration type controlling catapult behavior for this algorithm.
    ///
    /// For flat searches, this is a simple enabled/disabled flag.
    type CatapultChoice: Clone + Copy;

    /// The starting point selector type, parameterized by eviction strategy.
    ///
    /// Typically an `EngineStarter<T>` that uses LSH to map queries to catapult buckets.
    type StartingPointSelector<T: CatapultEvictingStructure>;

    /// The neighbor set representation type.
    ///
    /// For flat graphs, this is `FlatFixedSet`.
    type FixedSetType: FixedSet;
}
