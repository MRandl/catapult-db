//! Performance statistics tracking for graph search operations.
//!
//! This module provides structures for collecting and aggregating metrics about
//! search performance, including number of searches, nodes visited, distances computed,
//! and catapult usage.

mod stats;
pub use stats::*;
