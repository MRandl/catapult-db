//! Graph search algorithm abstractions and implementations.
//!
//! This module provides trait definitions and concrete implementations for different
//! graph search strategies, including flat (single-layer) and hierarchical approaches.

mod flat;
mod search_algorithm;

pub use flat::*;
pub use search_algorithm::*;
