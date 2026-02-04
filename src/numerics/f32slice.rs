use std::simd::{Simd, num::SimdFloat};

use crate::numerics::aligned_block::{AlignedBlock, SIMD_LANECOUNT};

type SimdF32 = Simd<f32, SIMD_LANECOUNT>;

/// A trait for vector‐like slices of `f32`, supporting common linear‐algebra
/// operations (dot product, L2 distance, normalization). The trait only has one
/// implementation, and exists because I could otherwise not add random Impl blocks
/// to the existing `[f32]` type from stdlib.
///
/// Implemented for `[f32]` using portable SIMD with lane-width [`SIMD_LANECOUNT`].
///
/// # Contract
///
/// - Operations involving two vectors (l2, dot product) require that they have the same length.
pub trait VectorLike {
    fn l2_squared(&self, othr: &Self) -> f32;
    fn l2(&self, othr: &Self) -> f32;
    fn dot(&self, othr: &Self) -> f32;
}

impl VectorLike for [AlignedBlock] {
    /// Computes the squared L2 (Euclidean) distance between two vectors using SIMD operations.
    ///
    /// Calculates `Σ_i (self[i] - other[i])²` across all aligned blocks in parallel.
    /// This is more efficient than computing the actual L2 distance when only comparing
    /// relative distances, since `dist(a,b) < dist(c,d)` ⟺ `dist²(a,b) < dist²(c,d)`.
    ///
    /// # Arguments
    /// * `othr` - The other vector to compute distance to, must have same length as `self`
    ///
    /// # Returns
    /// The squared L2 distance as an f32
    ///
    /// # Panics
    /// Panics if the two vectors have different lengths
    #[inline]
    fn l2_squared(&self, othr: &[AlignedBlock]) -> f32 {
        assert_eq!(self.len(), othr.len());

        let mut intermediate_sum_lanes = SimdF32::splat(0.0);

        for (&slice_self, &slice_othr) in self.iter().zip(othr.iter()) {
            let f32simd_slf = SimdF32::from_array(slice_self.data);
            let f32simd_oth = SimdF32::from_array(slice_othr.data);
            let diff = f32simd_slf - f32simd_oth;
            intermediate_sum_lanes += diff * diff;
        }

        intermediate_sum_lanes.reduce_sum() // 8-to-1 sum
    }

    /// Computes the L2 (Euclidean) distance between two vectors using SIMD operations.
    ///
    /// Calculates `√(Σ_i (self[i] - other[i])²)` by computing the squared distance
    /// and taking the square root.
    ///
    /// # Arguments
    /// * `other` - The other vector to compute distance to, must have same length as `self`
    ///
    /// # Returns
    /// The L2 distance as an f32
    ///
    /// # Panics
    /// Panics if the two vectors have different lengths
    #[inline]
    fn l2(&self, other: &[AlignedBlock]) -> f32 {
        self.l2_squared(other).sqrt()
    }

    /// Computes the dot product of two vectors using SIMD operations.
    ///
    /// Calculates `Σ_i (self[i] * other[i])` across all aligned blocks in parallel.
    ///
    /// # Arguments
    /// * `othr` - The other vector to compute dot product with, must have same length as `self`
    ///
    /// # Returns
    /// The dot product as an f32
    ///
    /// # Panics
    /// Panics if the two vectors have different lengths
    #[inline]
    fn dot(&self, othr: &[AlignedBlock]) -> f32 {
        assert_eq!(self.len(), othr.len());

        let mut intermediate_sum_lanes = SimdF32::splat(0.0);

        for (&slice_self, &slice_othr) in self.iter().zip(othr.iter()) {
            let f32simd_slf = SimdF32::from_array(slice_self.data);
            let f32simd_oth = SimdF32::from_array(slice_othr.data);
            intermediate_sum_lanes += f32simd_slf * f32simd_oth;
        }

        intermediate_sum_lanes.reduce_sum() // 8-to-1 sum
    }
}

// todo at some point convert these tests to quickcheck for better testing range
#[cfg(test)]
mod tests {
    use super::*;
    use core::f32;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        let diff = (a - b).abs();
        diff < eps
    }

    fn scalar_l2_sq(x: &[AlignedBlock], y: &[AlignedBlock]) -> f32 {
        x.iter()
            .zip(y)
            .map(|(ba, bb)| {
                let mut block_l2 = 0.0;
                for (a, b) in ba.data.iter().zip(bb.data.iter()) {
                    let d = a - b;
                    block_l2 += d * d;
                }
                block_l2
            })
            .sum()
    }

    #[test]
    fn l2_squared_matches_scalar_multiple_of_lanes() {
        let x = vec![AlignedBlock::new([
            1.0, -2.0, 3.5, 0.0, 0.125, 4.0, -7.0, 2.0, 1.0, -2.0, 3.5, 0.0, 0.125, 4.0, -7.0, 2.0,
        ])];
        let y = vec![AlignedBlock::new([
            0.5, 2.0, 3.0, 1.0, -0.5, 1.5, 2.0, 1.0, 0.5, 2.0, 3.0, 1.0, -0.5, 1.5, 2.0, 1.0,
        ])];

        let simd = x.l2_squared(&y);
        let scalar = scalar_l2_sq(&x, &y);
        assert!(approx_eq(simd, scalar, EPS), "simd={simd} scalar={scalar}");
    }

    #[test]
    fn l2_is_sqrt_of_l2_squared() {
        let x = [AlignedBlock::new([
            1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 0.25, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 0.25,
        ])];
        let y = [AlignedBlock::new([
            0.0, 1.0, 1.5, 4.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.5, 4.0, 1.0, 0.0, 0.0, 2.,
        ])];

        let d2 = x.l2_squared(&y);
        let d = x.l2(&y);
        assert!(approx_eq(d, d2.sqrt(), EPS), "d={d} sqrt(d2)={}", d2.sqrt());
    }

    #[test]
    fn identical_vectors_have_zero_distance() {
        let x = vec![AlignedBlock::new([
            0.25, -1.0, 3.0, 4.0, 0.0, 2.0, -3.5, 1.0, 0.25, -1.0, 3.0, 4.0, 0.0, 2.0, -3.5, 1.0,
        ])];
        let d2 = x.l2_squared(&x);
        let d = x.l2(&x);
        assert!(approx_eq(d2, 0.0, EPS));
        assert!(approx_eq(d, 0.0, EPS));
    }
}
