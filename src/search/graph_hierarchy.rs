use crate::{
    search::hash_start::EngineStarter,
    sets::{
        catapults::CatapultEvictingStructure,
        fixed::{FixedSet, FlatFixedSet},
    },
};

pub trait GraphSearchAlgorithm {
    type CatapultChoice: Clone + Copy;
    type StartingPointSelector<T: CatapultEvictingStructure>;
    type FixedSetType: FixedSet;

    //fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool;
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
    type CatapultChoice = FlatCatapultChoice;
    type StartingPointSelector<T: CatapultEvictingStructure> = EngineStarter<T>;
    type FixedSetType = FlatFixedSet;

    // fn local_catapults_enabled(strategy: Self::CatapultChoice) -> bool {
    //     strategy == FlatCatapultChoice::CatapultsEnabled
    // }
}

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
