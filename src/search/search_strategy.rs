use crate::search::hash_start::zorder_index::{LSH_APG_REDUNDANCY, ZOrderIndex};

pub enum SearchStrategy {
    Vanilla,
    Catapult,
    LshApg([ZOrderIndex; LSH_APG_REDUNDANCY]),
}

pub struct LshApgArgs {
    pub num_hash: usize,
    pub stored_vectors_dim: usize,
    pub seed: u64,
    pub w: f32,
}

impl SearchStrategy {
    pub fn from_string(s: &str, args: Option<LshApgArgs>) -> Self {
        if s == "vanilla" && args.is_none() {
            SearchStrategy::Vanilla
        } else if s == "catapult" && args.is_none() {
            SearchStrategy::Catapult
        } else if s == "lshapg"
            && let Some(args) = args
        {
            SearchStrategy::LshApg([ZOrderIndex::new(
                args.num_hash,
                args.stored_vectors_dim,
                args.seed,
                args.w,
            )])
        } else {
            panic!(
                "Invalid runtime mode: {}, lshapgargs: {}",
                s,
                args.is_some()
            )
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::search::{LshApgArgs, SearchStrategy};

    #[test]
    pub fn load_vanilla() {
        assert!(matches!(
            SearchStrategy::from_string("vanilla", None),
            SearchStrategy::Vanilla
        ))
    }

    #[test]
    pub fn load_catapult() {
        assert!(matches!(
            SearchStrategy::from_string("catapult", None),
            SearchStrategy::Catapult
        ))
    }

    #[test]
    #[should_panic]
    pub fn load_panic() {
        SearchStrategy::from_string("lshapg", None);
    }

    #[test]
    pub fn load_lshapg() {
        assert!(matches!(
            SearchStrategy::from_string(
                "lshapg",
                Some(LshApgArgs {
                    num_hash: 10,
                    stored_vectors_dim: 16,
                    seed: 0,
                    w: 1.0
                })
            ),
            SearchStrategy::LshApg(_)
        ))
    }
}
