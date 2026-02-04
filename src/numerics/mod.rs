//! Numerical operations and SIMD-optimized data structures for vector computations.
//!
//! This module provides SIMD-accelerated distance computations and vector operations
//! using 64-byte aligned blocks of 16 f32 values for efficient parallel processing.

mod aligned_block;
mod f32slice;

pub use aligned_block::{AlignedBlock, SIMD_LANECOUNT};
pub use f32slice::VectorLike;
