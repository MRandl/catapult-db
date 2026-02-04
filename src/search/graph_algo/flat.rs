use crate::{
    search::{graph_algo::GraphSearchAlgorithm, hash_start::EngineStarter},
    sets::{catapults::CatapultEvictingStructure, fixed::FlatFixedSet},
};

/// A marker type for flat (single-layer) graph search algorithms.
///
/// Represents search strategies that operate on a single-layer proximity graph
/// without hierarchical navigation, similar to DiskANN or NSW (non-hierarchical).
pub struct FlatSearch;

/// Configuration for catapult usage in flat graph searches.
///
/// In flat searches, catapults are either globally enabled or disabled.
/// There are no per-layer or hierarchical catapult strategies.
#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum FlatCatapultChoice {
    /// Catapults are disabled; only the base starting node is used
    CatapultsDisabled = 0,

    /// Catapults are enabled; cached starting points from similar queries are used
    CatapultsEnabled = 1,
}

impl FlatCatapultChoice {
    /// Returns whether catapults are enabled for local (flat) search.
    ///
    /// # Returns
    /// `true` if catapults are enabled, `false` otherwise
    pub fn local_enabled(self) -> bool {
        self == FlatCatapultChoice::CatapultsEnabled
    }

    /// Creates a `FlatCatapultChoice` from a boolean flag.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable catapults
    ///
    /// # Returns
    /// `CatapultsEnabled` if `enabled` is true, `CatapultsDisabled` otherwise
    pub fn from_bool(enabled: bool) -> Self {
        if enabled {
            FlatCatapultChoice::CatapultsEnabled
        } else {
            FlatCatapultChoice::CatapultsDisabled
        }
    }
}

impl GraphSearchAlgorithm for FlatSearch {
    type CatapultChoice = FlatCatapultChoice;
    type StartingPointSelector<T: CatapultEvictingStructure> = EngineStarter<T>;
    type FixedSetType = FlatFixedSet;
}
