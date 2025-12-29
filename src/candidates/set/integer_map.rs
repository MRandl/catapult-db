use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

#[derive(Default)]
pub struct NoOpHasher {
    hash: u64,
}

impl Hasher for NoOpHasher {
    fn write(&mut self, _bytes: &[u8]) {
        panic!("This hasher only accepts u64/usize keys");
    }

    fn write_usize(&mut self, i: usize) {
        self.hash = i as u64;
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type IntegerMap<V> = HashMap<usize, V, BuildHasherDefault<NoOpHasher>>;

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn noop_hasher_with_usize() {
        let mut hasher = NoOpHasher::default();
        hasher.write_usize(42);
        assert_eq!(hasher.finish(), 42);

        let mut hasher2 = NoOpHasher::default();
        hasher2.write_usize(12345);
        assert_eq!(hasher2.finish(), 12345);
    }

    #[test]
    fn noop_hasher_with_u64() {
        let mut hasher = NoOpHasher::default();
        hasher.write_u64(99);
        assert_eq!(hasher.finish(), 99);

        let mut hasher2 = NoOpHasher::default();
        hasher2.write_u64(u64::MAX);
        assert_eq!(hasher2.finish(), u64::MAX);
    }

    #[test]
    fn noop_hasher_identity_function() {
        // The hasher should return the exact value written
        for value in [0, 1, 100, 1000, 10000, usize::MAX / 2] {
            let mut hasher = NoOpHasher::default();
            hasher.write_usize(value);
            assert_eq!(hasher.finish(), value as u64);
        }
    }

    #[test]
    #[should_panic(expected = "This hasher only accepts u64/usize keys")]
    fn noop_hasher_panics_on_write() {
        let mut hasher = NoOpHasher::default();
        hasher.write(&[1, 2, 3]);
    }

    #[test]
    fn integer_map_basic_operations() {
        let mut map: IntegerMap<String> = IntegerMap::default();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());

        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
        assert_eq!(map.get(&3), Some(&"three".to_string()));
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn integer_map_insert_and_update() {
        let mut map: IntegerMap<i32> = IntegerMap::default();

        map.insert(10, 100);
        assert_eq!(map.get(&10), Some(&100));

        map.insert(10, 200);
        assert_eq!(map.get(&10), Some(&200));
    }

    #[test]
    fn integer_map_remove() {
        let mut map: IntegerMap<&str> = IntegerMap::default();

        map.insert(5, "value");
        assert_eq!(map.get(&5), Some(&"value"));

        map.remove(&5);
        assert_eq!(map.get(&5), None);
    }

    #[test]
    fn integer_map_with_large_keys() {
        let mut map: IntegerMap<bool> = IntegerMap::default();

        let large_keys = [0, 1000, 10000, 100000, 1000000, usize::MAX / 2];

        for &key in &large_keys {
            map.insert(key, true);
        }

        for &key in &large_keys {
            assert_eq!(map.get(&key), Some(&true));
        }
    }

    #[test]
    fn integer_map_sparse_keys() {
        let mut map: IntegerMap<usize> = IntegerMap::default();

        // Insert values at very sparse positions
        map.insert(0, 10);
        map.insert(1000000, 20);
        map.insert(2000000, 30);

        assert_eq!(map.get(&0), Some(&10));
        assert_eq!(map.get(&1000000), Some(&20));
        assert_eq!(map.get(&2000000), Some(&30));

        // Keys in between should not exist
        assert_eq!(map.get(&500000), None);
        assert_eq!(map.get(&1500000), None);
    }

    #[test]
    fn integer_map_entry_api() {
        let mut map: IntegerMap<Vec<i32>> = IntegerMap::default();

        map.entry(1).or_default().push(10);
        map.entry(1).or_default().push(20);
        map.entry(2).or_default().push(30);

        assert_eq!(map.get(&1), Some(&vec![10, 20]));
        assert_eq!(map.get(&2), Some(&vec![30]));
    }

    #[test]
    fn integer_map_iteration() {
        let mut map: IntegerMap<char> = IntegerMap::default();

        map.insert(1, 'a');
        map.insert(2, 'b');
        map.insert(3, 'c');

        let mut collected: Vec<_> = map.iter().collect();
        collected.sort_by_key(|(k, _)| *k);

        assert_eq!(collected, vec![(&1, &'a'), (&2, &'b'), (&3, &'c')]);
    }

    #[test]
    fn integer_map_len_and_is_empty() {
        let mut map: IntegerMap<i32> = IntegerMap::default();

        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert(1, 100);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);

        map.insert(2, 200);
        assert_eq!(map.len(), 2);

        map.remove(&1);
        assert_eq!(map.len(), 1);

        map.remove(&2);
        assert!(map.is_empty());
    }

    #[test]
    fn integer_map_clear() {
        let mut map: IntegerMap<String> = IntegerMap::default();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        assert_eq!(map.len(), 2);

        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn integer_map_contains_key() {
        let mut map: IntegerMap<f64> = IntegerMap::default();

        map.insert(42, f64::consts::PI);

        assert!(map.contains_key(&42));
        assert!(!map.contains_key(&43));
    }
}
