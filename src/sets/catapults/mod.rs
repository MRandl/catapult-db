//! Catapult storage structures for caching search starting points.
//!
//! Catapults are cached node indices from previous successful searches that serve
//! as accelerated starting points for similar future queries. This module provides
//! eviction-based data structures to store and manage these catapults efficiently.

mod catapult_neighbor_set;
mod fifo_set;

pub use catapult_neighbor_set::*;
pub use fifo_set::*;
