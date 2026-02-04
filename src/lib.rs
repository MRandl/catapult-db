#![feature(portable_simd)]

//! # Catapult: Approximate Nearest Neighbor Search with LSH-Cached Starting Points
//!
//! This library implements efficient approximate nearest neighbor (ANN) search on flat
//! proximity graphs using beam search with LSH-based catapult acceleration.
//!
//! ## Core Concepts
//!
//! - **Proximity Graph**: A graph where each node stores a vector embedding and connections
//!   to its nearest neighbors in the embedding space
//! - **Beam Search**: Best-first search maintaining a beam of `k` closest candidates,
//!   iteratively expanding the nearest unvisited node
//! - **Catapults**: Cached starting points from previous successful searches, stored in
//!   LSH buckets to provide good starting positions for similar queries
//! - **LSH (Locality-Sensitive Hashing)**: Maps similar queries to the same signature/bucket,
//!   enabling retrieval of relevant cached starting points
//!
//! ## Modules
//!
//! - [`numerics`]: SIMD-aligned vector operations and distance computations
//! - [`search`]: Core graph search algorithms, LSH, and node structures
//! - [`sets`]: Specialized data structures (candidates, catapults, visited tracking, fixed neighbors)
//! - [`fs`]: File I/O for loading graphs and query vectors
//! - [`statistics`]: Search performance metrics and statistics
//!

pub mod fs;
pub mod numerics;
pub mod search;
pub mod sets;
pub mod statistics;
