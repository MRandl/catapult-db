use crate::search::hash_start::zorder_index::{LSH_APG_REDUNDANCY, ZOrderIndex};

pub enum SearchStrategy {
    Vanilla,
    Catapult,
    LshApg([ZOrderIndex; LSH_APG_REDUNDANCY]),
}
