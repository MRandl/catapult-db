//! Graph search algorithm abstractions and implementations.
//!
//! This module provides trait definitions and concrete implementations for
//! flat (single-layer) proximity graph search strategies.

mod flat;
mod search_algorithm;

pub use flat::*;
pub use search_algorithm::*;
