pub trait GraphSearchAlgo {
    type LevelContext;
}

pub struct FlatSearch;

impl GraphSearchAlgo for FlatSearch {
    type LevelContext = ();
}

pub struct HNSWSearch {
    pub level: u32,
}

impl GraphSearchAlgo for HNSWSearch {
    type LevelContext = u32;
}
