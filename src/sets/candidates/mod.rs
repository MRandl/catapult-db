//! Candidate management structures for graph search operations.
//!
//! This module provides data structures for tracking and managing candidate nodes
//! during nearest neighbor search, including specialized floating-point ordering
//! and bounded k-smallest tracking with deduplication.

mod candidate_entry;
mod ordered_float;
mod smallest_k;

pub use candidate_entry::*;
pub use ordered_float::*;
pub use smallest_k::*;
