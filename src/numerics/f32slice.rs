use std::simd::{Simd, num::SimdFloat};

pub const SIMD_LANECOUNT: usize = 8;
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
/// - The slice length must be a multiple of [`SIMD_LANECOUNT`].
/// - Operations involving two vectors (l2, dot product) require that they have the same length.
/// - Mismatched lengths or vectors with SIMD-awkward lengths will panic via `assert!` in both debug and release builds.

pub trait VectorLike {
    fn l2_squared(&self, othr: &Self) -> f32;
    fn l2(&self, othr: &Self) -> f32;
    fn dot(&self, othr: &Self) -> f32;
    fn normalized(&self) -> Vec<f32>;
}

impl VectorLike for [f32] {
    /// # Usage
    /// Computes the **SQUARED** L2 distance between two vectors:
    ///
    /// ```text
    /// L2^2(x, y) = Σ_i (x[i] - y[i]) ** 2
    /// ```
    ///
    /// This is typically useful when comparing two distances :
    ///
    /// dist(u,v) < dist(w, x) ⇔ dist(u,v) ** 2 < dist(w,x) ** 2
    ///
    /// We are usually interested in the left side of the equivalence,
    /// but the right side is slightly cheaper to compute.
    ///
    /// # Panics
    ///
    /// Panics if the two vectors have different lengths or if the length
    /// is not a multiple of [`SIMD_LANECOUNT`].
    #[inline]
    fn l2_squared(&self, othr: &[f32]) -> f32 {
        assert!(self.len() == othr.len());
        assert!(self.len().is_multiple_of(SIMD_LANECOUNT));

        let mut intermediate_sum_lanes = Simd::<f32, SIMD_LANECOUNT>::splat(0.0);

        let self_chunks = self.as_chunks::<SIMD_LANECOUNT>().0.iter();
        let othr_chunks = othr.as_chunks::<SIMD_LANECOUNT>().0.iter();

        for (&slice_self, &slice_othr) in self_chunks.zip(othr_chunks) {
            let f32simd_slf = SimdF32::from_array(slice_self);
            let f32simd_oth = SimdF32::from_array(slice_othr);
            let diff = f32simd_slf - f32simd_oth;
            intermediate_sum_lanes += diff * diff;
        }

        intermediate_sum_lanes.reduce_sum() // 8-to-1 sum
    }

    /// # Usage
    /// Computes the L2 distance between two vectors.
    ///
    /// # Panics
    ///
    /// Panics if the two vectors have different lengths or if the length
    /// is not a multiple of [`SIMD_LANECOUNT`].
    #[inline]
    fn l2(&self, other: &[f32]) -> f32 {
        self.l2_squared(other).sqrt()
    }

    /// # Usage
    /// Computes the **dot product** between two vectors:
    ///
    /// ```text
    /// x·y = Σ_i x[i] * y[i]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths or if the length
    /// is not a multiple of [`SIMD_LANECOUNT`].
    #[inline]
    fn dot(&self, othr: &[f32]) -> f32 {
        assert!(self.len() == othr.len());
        assert!(self.len().is_multiple_of(SIMD_LANECOUNT));

        let mut intermediate_sum_lanes = Simd::<f32, SIMD_LANECOUNT>::splat(0.0);

        let self_chunks = self.as_chunks::<SIMD_LANECOUNT>().0.iter();
        let othr_chunks = othr.as_chunks::<SIMD_LANECOUNT>().0.iter();

        for (&slice_self, &slice_othr) in self_chunks.zip(othr_chunks) {
            let f32simd_slf = SimdF32::from_array(slice_self);
            let f32simd_oth = SimdF32::from_array(slice_othr);
            intermediate_sum_lanes += f32simd_slf * f32simd_oth;
        }

        // 8-to-1
        intermediate_sum_lanes.reduce_sum()
    }

    /// Returns a new vector that is the **L2-normalized** version of `self`.
    ///
    /// Each element is divided by the Euclidean norm of the vector:
    ///
    /// ```text
    /// normalized(x) = x / ||x||_2
    /// ```
    ///
    /// If the norm is zero, returns a vector of zeros with the same length.
    ///
    /// # Examples
    /// ```
    /// # use catapult::numerics::{VectorLike, SIMD_LANECOUNT};
    /// let v = [1.0f32; SIMD_LANECOUNT];
    /// let n = v.normalized();
    /// let len = n.dot(&n).sqrt();
    /// assert!((len - 1.0).abs() < 1e-6);
    /// ```
    /// # Panics
    ///
    /// Panics if the vector length is not a multiple of [`SIMD_LANECOUNT`].
    #[inline]
    fn normalized(&self) -> Vec<f32> {
        assert!(self.len().is_multiple_of(SIMD_LANECOUNT));

        let norm = self.dot(self).sqrt();
        if norm == 0.0 {
            // avoid division by zero; return zero vector
            return vec![0.0; self.len()];
        }
        let inv_norm = 1.0 / norm;

        let mut out = Vec::with_capacity(self.len());

        for &chunk in self.as_chunks::<SIMD_LANECOUNT>().0.iter() {
            let simd_chunk = SimdF32::from_array(chunk);
            let scaled = simd_chunk * SimdF32::splat(inv_norm);
            out.extend(&scaled.to_array());
        }

        out
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

    fn scalar_dot(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }

    fn scalar_l2_sq(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    #[test]
    fn dot_matches_scalar_multiple_of_lanes() {
        let x = [1.0, -2.0, 3.5, 0.0, 0.125, 4.0, -7.0, 2.0];
        let y = [0.5, 2.0, 3.0, 1.0, -0.5, 1.5, 2.0, 1.0];

        let simd = x.dot(&y);
        let scalar = scalar_dot(&x, &y);
        assert!(approx_eq(simd, scalar, EPS), "simd={simd} scalar={scalar}");
    }

    #[test]
    fn l2_squared_matches_scalar_multiple_of_lanes() {
        let x = [1.0, -2.0, 3.5, 0.0, 0.125, 4.0, -7.0, 2.0];
        let y = [0.5, 2.0, 3.0, 1.0, -0.5, 1.5, 2.0, 1.0];

        let simd = x.l2_squared(&y);
        let scalar = scalar_l2_sq(&x, &y);
        assert!(approx_eq(simd, scalar, EPS), "simd={simd} scalar={scalar}");
    }

    #[test]
    fn l2_is_sqrt_of_l2_squared() {
        let x = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 0.25];
        let y = [0.0, 1.0, 1.5, 4.0, 1.0, 0.0, 0.0, 2.0];

        let d2 = x.l2_squared(&y);
        let d = x.l2(&y);
        assert!(approx_eq(d, d2.sqrt(), EPS), "d={d} sqrt(d2)={}", d2.sqrt());
    }

    #[test]
    fn parallelogram_identity() {
        // ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        let x = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 5.0, 0.0];
        let y = [2.0, 2.0, 1.0, -1.0, 0.5, 4.0, 0.0, 1.0];

        let xdotx = x.dot(&x);
        let ydoty = y.dot(&y);
        let xdoty = x.dot(&y);
        let d2 = x.l2_squared(&y);

        let rhs = xdotx + ydoty - 2.0 * xdoty;
        assert!(approx_eq(d2, rhs, 1e-4), "d2={d2} rhs={rhs}");
    }

    #[test]
    fn zero_vector_normalized_is_zero() {
        let z = [0.0f32; SIMD_LANECOUNT];
        let n = z.normalized();
        assert!(n.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn normalized_has_unit_length_or_zero() {
        let x = [1.0, 2.0, 2.0, -1.0, 0.5, -0.5, 4.0, 3.5];
        let n = x.normalized();

        // length ~ 1
        let len = n.dot(&n).sqrt();
        assert!(approx_eq(len, 1.0, 1e-5), "len={len}");
    }

    #[test]
    fn normalization_preserves_direction() {
        // For any component i where x_i != 0, n_i / x_i is constant.
        let x = [1.0, -2.0, 0.0, 4.0, -3.0, 0.0, 8.0, -0.5];
        let n = x.normalized();

        // Find first nonzero to compute scale
        let mut scale = None;
        for (&xi, &ni) in x.iter().zip(&n) {
            if xi != 0.0 {
                scale = Some(ni / xi);
                break;
            }
        }
        let s = scale.expect("nonzero component expected");
        for (&xi, &ni) in x.iter().zip(&n) {
            if xi != 0.0 {
                assert!(approx_eq(ni, s * xi, 1e-5), "ni={ni} s*xi={}", s * xi);
            } else {
                assert!(ni == 0.0);
            }
        }
    }

    #[test]
    fn identical_vectors_have_zero_distance() {
        let x = [0.25, -1.0, 3.0, 4.0, 0.0, 2.0, -3.5, 1.0];
        let d2 = x.l2_squared(&x);
        let d = x.l2(&x);
        assert!(approx_eq(d2, 0.0, EPS));
        assert!(approx_eq(d, 0.0, EPS));
    }
}
