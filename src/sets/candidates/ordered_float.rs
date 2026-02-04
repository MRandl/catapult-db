use std::{cmp::Ordering, hash::Hash, hash::Hasher};

/// A wrapper around f32 that provides total ordering and proper equality semantics.
///
/// Standard f32 does not implement `Ord` or `Eq` due to NaN values and signed zeros.
/// This wrapper uses bit-level comparison to ensure that any floating-point numbers are
/// compared using a total order, including NaN values and signed zeros.
///
/// This enables f32 values to be used in sorted collections and as hash keys.
#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct TotalF32(pub f32);

impl PartialEq for TotalF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for TotalF32 {}

impl PartialOrd for TotalF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TotalF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl From<f32> for TotalF32 {
    fn from(x: f32) -> Self {
        TotalF32(x)
    }
}

impl Hash for TotalF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    #[test]
    fn test_equality() {
        let a = TotalF32(1.0);
        let b = TotalF32(1.0);
        let c = TotalF32(2.0);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_nan_equality() {
        let nan1 = TotalF32(f32::NAN);
        let nan2 = TotalF32(f32::NAN);

        assert_eq!(nan1, nan2);
    }

    #[test]
    fn test_negative_zero_equality() {
        let pos_zero = TotalF32(0.0);
        let neg_zero = TotalF32(-0.0);

        assert_ne!(pos_zero, neg_zero);
    }

    #[test]
    fn test_ordering() {
        let a = TotalF32(1.0);
        let b = TotalF32(2.0);
        let c = TotalF32(3.0);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
        assert!(c > b);
        assert!(b > a);
    }

    #[test]
    fn test_ordering_with_negative() {
        let neg = TotalF32(-1.0);
        let pos = TotalF32(1.0);
        let zero = TotalF32(0.0);

        assert!(neg < zero);
        assert!(zero < pos);
        assert!(neg < pos);
    }

    #[test]
    fn test_ordering_with_nan() {
        let nan = TotalF32(f32::NAN);
        let normal = TotalF32(1.0);
        let infinity = TotalF32(f32::INFINITY);
        let neg_infinity = TotalF32(f32::NEG_INFINITY);

        assert!(nan > infinity);
        assert!(nan > normal);
        assert!(nan > neg_infinity);
        assert!(infinity > normal);
        assert!(normal > neg_infinity);
    }

    #[test]
    fn test_ordering_zero() {
        let pos_zero = TotalF32(0.0);
        let neg_zero = TotalF32(-0.0);

        assert!(pos_zero > neg_zero);
    }

    #[test]
    fn test_from_f32() {
        let f = 3.5f32;
        let total: TotalF32 = f.into();

        assert_eq!(total.0, f);
    }

    #[test]
    fn test_hash_consistency() {
        let a = TotalF32(1.0);
        let b = TotalF32(1.0);

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();

        a.hash(&mut hasher_a);
        b.hash(&mut hasher_b);

        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn test_hash_different_values() {
        let a = TotalF32(1.0);
        let b = TotalF32(2.0);

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();

        a.hash(&mut hasher_a);
        b.hash(&mut hasher_b);

        assert_ne!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn test_hash_nan() {
        let nan1 = TotalF32(f32::NAN);
        let nan2 = TotalF32(f32::NAN);

        let mut hasher_1 = DefaultHasher::new();
        let mut hasher_2 = DefaultHasher::new();

        nan1.hash(&mut hasher_1);
        nan2.hash(&mut hasher_2);

        assert_eq!(hasher_1.finish(), hasher_2.finish());
    }

    #[test]
    fn test_hash_zeros() {
        let pos_zero = TotalF32(0.0);
        let neg_zero = TotalF32(-0.0);

        let mut hasher_pos = DefaultHasher::new();
        let mut hasher_neg = DefaultHasher::new();

        pos_zero.hash(&mut hasher_pos);
        neg_zero.hash(&mut hasher_neg);

        assert_ne!(hasher_pos.finish(), hasher_neg.finish());
    }

    #[test]
    fn test_partial_cmp() {
        let a = TotalF32(1.0);
        let b = TotalF32(2.0);

        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
    }

    #[test]
    fn test_copy_clone() {
        let a = TotalF32(1.0);
        let b = a;
        #[allow(clippy::clone_on_copy)]
        let c = a.clone();

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
    }

    #[test]
    fn test_debug() {
        let a = TotalF32(1.5);
        let debug_str = format!("{a:?}");

        assert!(debug_str.contains("TotalF32"));
        assert!(debug_str.contains("1.5"));
    }

    #[test]
    fn test_sort() {
        let mut values = [
            TotalF32(3.0),
            TotalF32(1.0),
            TotalF32(f32::NAN),
            TotalF32(2.0),
            TotalF32(f32::INFINITY),
            TotalF32(f32::NEG_INFINITY),
            TotalF32(0.0),
            TotalF32(-0.0),
        ];

        values.sort();

        assert_eq!(values[0], TotalF32(f32::NEG_INFINITY));
        assert_eq!(values[1], TotalF32(-0.0));
        assert_eq!(values[2], TotalF32(0.0));
        assert!(values[3].0 > 0.0 && values[3].0 < 2.0);
        assert!(values[4].0 >= 2.0 && values[4].0 < 3.0);
        assert!(values[5].0 >= 3.0 && values[5].0 < f32::INFINITY);
        assert_eq!(values[6], TotalF32(f32::INFINITY));
        assert_eq!(values[7], TotalF32(f32::NAN));
    }
}
