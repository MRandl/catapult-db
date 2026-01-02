use crate::{numerics::AlignedBlock, search::hash_start::EngineStarter};

pub trait GraphSearchAlgorithm {
    type LevelContext: Clone + Copy;
    type CatapultChoice: Clone + Copy;
    type StartingPointSelector;

    fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool;
}

// search mode 1: Flat search, where one search is performed on a big graph.
// Typical example for this behavior is DiskANN. This does not make use of recursive graphs like HNSW.
pub struct FlatSearch;

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum FlatCatapultChoice {
    // in non-HNSW workloads, catapults are either on or off
    CatapultsDisabled = 0,
    CatapultsEnabled = 1,
}

impl GraphSearchAlgorithm for FlatSearch {
    type LevelContext = (); // flat search has no concept of a graph level
    type CatapultChoice = FlatCatapultChoice;
    type StartingPointSelector = EngineStarter;

    fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool {
        strategy == FlatCatapultChoice::CatapultsEnabled
    }
}

// Search mode 2: HNSW, which makes use of stacked graphs which get progressively denser
// This one has a few option for catapult generation. SameLevel means that catapults are
// placed on a given level and cannot 'jump' from one level to the other. Finalizing means that
// catapults are placed from the entry point (max layer) to the final point of the run (layer 0)
pub struct HNSWSearch {}

pub struct HNSWEngineStarter {
    hasher: EngineStarter,
    pub max_level: u32,
}

impl HNSWEngineStarter {
    pub fn select_starting_points(&self, query: &[AlignedBlock], k: usize) -> Vec<usize> {
        self.hasher.select_starting_points(query, k)
    }
}

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum HNSWCatapultChoice {
    CatapultsDisabled = 0,
    SameLevelCatapults,
    FinalizingCatapults,
}

impl GraphSearchAlgorithm for HNSWSearch {
    type LevelContext = u32;
    type CatapultChoice = HNSWCatapultChoice;
    type StartingPointSelector = HNSWEngineStarter;

    fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool {
        strategy == HNSWCatapultChoice::SameLevelCatapults
    }
}
