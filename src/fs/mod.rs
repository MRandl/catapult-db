//! File system I/O operations for loading graphs and queries.
//!
//! This module provides functionality for loading proximity graphs and query vectors
//! from disk, supporting NumPy format for queries and custom binary formats for graphs.

mod adjacency_load;
mod query_load;

pub use query_load::*;
