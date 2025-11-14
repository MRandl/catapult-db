pub trait Queries {
    fn load_from_npy(path: &str) -> Self;
}

impl Queries for Vec<Vec<f32>> {
    fn load_from_npy(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
        assert!(npy.shape().len() == 2);
        let (d1, d2) = (npy.shape()[0] as usize, npy.shape()[1] as usize);

        let mut iter = npy.data::<f32>().unwrap();
        let mut result = Vec::with_capacity(d1);
        for _ in 0..d1 {
            let mut row = Vec::with_capacity(d2);
            for _ in 0..d2 {
                row.push(iter.next().unwrap().unwrap());
            }
            result.push(row);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_4vecs() {
        let vectors = Vec::<Vec<f32>>::load_from_npy("test_index/4vecs/4vecs.npy");

        println!("Loaded {} vectors", vectors.len());
        for (i, vec) in vectors.iter().enumerate() {
            println!("Vector {}: {:?}", i, vec);
        }
    }
}
