//! Specialized data structures for graph search operations.
//!
//! This module provides efficient data structures tailored for approximate nearest
//! neighbor search on proximity graphs.
//!
//! # Submodules
//!
//! - [`candidates`]: Priority queue structures for maintaining top-k search candidates
//! - [`catapults`]: FIFO evicting sets for storing cached starting points from previous searches
//! - [`fixed`]: Immutable neighbor sets for graph nodes
//! - [`visited`]: Bitmap-based visited node tracking for efficient duplicate detection

pub mod candidates;
pub mod catapults;
pub mod fixed;
pub mod visited;
