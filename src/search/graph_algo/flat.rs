use crate::{
    search::{graph_algo::GraphSearchAlgorithm, hash_start::EngineStarter},
    sets::{catapults::CatapultEvictingStructure, fixed::FlatFixedSet},
};

pub struct FlatSearch;

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum FlatCatapultChoice {
    // in non-HNSW workloads, catapults are either on or off
    CatapultsDisabled = 0,
    CatapultsEnabled = 1,
}

impl FlatCatapultChoice {
    pub fn local_enabled(self) -> bool {
        self == FlatCatapultChoice::CatapultsEnabled
    }

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
