use crate::sets::{catapults::CatapultEvictingStructure, fixed::FixedSet};

/// A trait defining the components of a graph search algorithm strategy.
///
/// This trait abstracts over different graph search approaches (flat vs hierarchical)
/// by specifying the associated types needed for each strategy: catapult configuration,
/// starting point selection mechanism, and neighbor set representation.
///
/// Implementations include:
/// - `FlatSearch`: Single-layer proximity graph search (e.g., DiskANN-style)
/// - Future: HNSW (Hierarchical Navigable Small World) with multi-layer graphs
pub trait GraphSearchAlgorithm {
    /// Configuration type controlling catapult behavior for this algorithm.
    ///
    /// For flat searches, this is a simple enabled/disabled flag.
    /// For hierarchical searches, this could control layer-specific catapult strategies.
    type CatapultChoice: Clone + Copy;

    /// The starting point selector type, parameterized by eviction strategy.
    ///
    /// Typically an `EngineStarter<T>` that uses LSH to map queries to catapult buckets.
    type StartingPointSelector<T: CatapultEvictingStructure>;

    /// The neighbor set representation type.
    ///
    /// For flat graphs, this is `FlatFixedSet`. For hierarchical graphs, this could be
    /// a multi-level structure supporting layer-specific neighbor access.
    type FixedSetType: FixedSet;
}

// search mode 1: Flat search, where one search is performed on a big graph.
// Typical example for this behavior is DiskANN. This does not make use of recursive graphs like HNSW.

// Search mode 2: HNSW, which makes use of stacked graphs which get progressively denser
// This one has a few option for catapult generation. SameLevel means that catapults are
// placed on a given level and cannot 'jump' from one level to the other. Finalizing means that
// catapults are placed from the entry point (max layer) to the final point of the run (layer 0)
// pub struct HNSWSearch {}

// pub struct HNSWEngineStarter {
//     hasher: EngineStarter,
//     pub max_level: usize,
// }

// impl HNSWEngineStarter {
//     pub fn new(hasher: EngineStarter, max_level: usize) -> Self {
//         HNSWEngineStarter { hasher, max_level }
//     }

//     pub fn select_starting_points(&self, query: &[AlignedBlock], k: usize) -> Vec<usize> {
//         self.hasher.select_starting_points(query, k)
//     }
// }

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum HNSWCatapultChoice {
    CatapultsDisabled = 0,
    SameLevelCatapults,
    FinalizingCatapults,
}

// impl GraphSearchAlgorithm for HNSWSearch {
//     type CatapultChoice = HNSWCatapultChoice;
//     type StartingPointSelector = HNSWEngineStarter;
//     type FixedSetType = HierarchicalFixedSet;

//     fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool {
//         strategy == HNSWCatapultChoice::SameLevelCatapults
//     }
// }
