//! Graph search algorithms and data structures for approximate nearest neighbor search.
//!
//! This module provides the core functionality for beam search on flat proximity graphs,
//! including LSH-based starting point selection (catapults) and the main graph structure.
//!
//! # Main Components
//!
//! - [`AdjacencyGraph`]: The main graph structure storing nodes and supporting beam search
//! - [`Node`]: Individual graph nodes with vector payloads and neighbor connections
//! - [`NodeId`]: Type-safe wrapper for node indices
//! - [`graph_algo`]: Search algorithm trait definitions and implementations
//! - [`hash_start`]: LSH-based catapult management for starting point selection

pub mod graph_algo;
pub mod hash_start;

mod adjacency_graph;
mod node;
mod running_mode;

pub use adjacency_graph::*;
pub use node::*;
pub use running_mode::*;
