use crate::sets::visited::{IntegerSet, VisitorSet};

impl VisitorSet for IntegerSet {
    fn get(&self, i: usize) -> bool {
        self.contains(&i)
    }

    fn set(&mut self, i: usize) {
        self.insert(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_set_is_empty() {
        let set = IntegerSet::default();
        assert!(!VisitorSet::get(&set, 0));
        assert!(!VisitorSet::get(&set, 1));
        assert!(!VisitorSet::get(&set, 100));
    }

    #[test]
    fn test_set_and_get_single_value() {
        let mut set = IntegerSet::default();
        VisitorSet::set(&mut set, 5);
        assert!(VisitorSet::get(&set, 5));
        assert!(!VisitorSet::get(&set, 4));
        assert!(!VisitorSet::get(&set, 6));
    }

    #[test]
    fn test_set_multiple_values() {
        let mut set = IntegerSet::default();
        VisitorSet::set(&mut set, 1);
        VisitorSet::set(&mut set, 10);
        VisitorSet::set(&mut set, 100);

        assert!(VisitorSet::get(&set, 1));
        assert!(VisitorSet::get(&set, 10));
        assert!(VisitorSet::get(&set, 100));
        assert!(!VisitorSet::get(&set, 0));
        assert!(!VisitorSet::get(&set, 50));
    }

    #[test]
    fn test_set_same_value_twice() {
        let mut set = IntegerSet::default();
        VisitorSet::set(&mut set, 42);
        VisitorSet::set(&mut set, 42);

        assert!(VisitorSet::get(&set, 42));
    }

    #[test]
    fn test_set_zero() {
        let mut set = IntegerSet::default();
        VisitorSet::set(&mut set, 0);
        assert!(VisitorSet::get(&set, 0));
        assert!(!VisitorSet::get(&set, 1));
    }

    #[test]
    fn test_set_large_values() {
        let mut set = IntegerSet::default();
        VisitorSet::set(&mut set, usize::MAX);
        VisitorSet::set(&mut set, usize::MAX - 1);

        assert!(VisitorSet::get(&set, usize::MAX));
        assert!(VisitorSet::get(&set, usize::MAX - 1));
        assert!(!VisitorSet::get(&set, usize::MAX - 2));
    }

    #[test]
    fn test_sequential_values() {
        let mut set = IntegerSet::default();
        for i in 0..100 {
            VisitorSet::set(&mut set, i);
        }

        for i in 0..100 {
            assert!(VisitorSet::get(&set, i));
        }
        assert!(!VisitorSet::get(&set, 100));
        assert!(!VisitorSet::get(&set, 101));
    }

    #[test]
    fn test_sparse_values() {
        let mut set = IntegerSet::default();
        let sparse_values = vec![0, 100, 1000, 10000, 100000];

        for &val in &sparse_values {
            VisitorSet::set(&mut set, val);
        }

        for &val in &sparse_values {
            assert!(VisitorSet::get(&set, val));
        }

        assert!(!VisitorSet::get(&set, 50));
        assert!(!VisitorSet::get(&set, 500));
        assert!(!VisitorSet::get(&set, 5000));
    }
}
