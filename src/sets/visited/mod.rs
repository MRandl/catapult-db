//! Visited node tracking structures for graph traversal.
//!
//! This module provides data structures for efficiently tracking which nodes have been
//! visited during graph search operations, preventing redundant expansions and cycles.
//! Multiple implementations are available with different space-time tradeoffs.

mod hashset;
mod integer_map;
mod page;
mod uncompressed_set;
mod visitor_set;

pub use integer_map::*;
pub use page::*;
pub use uncompressed_set::*;
pub use visitor_set::*;
