//! Locality-sensitive hashing for query-to-starting-point mapping.
//!
//! This module provides LSH-based infrastructure for mapping query vectors to cached
//! catapult starting points, enabling fast warm starts for similar queries.

mod engine_starter;
mod hyperplane_hasher;
mod pstable_hasher;

pub use engine_starter::*;
