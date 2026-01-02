use crate::numerics::{AlignedBlock, SIMD_LANECOUNT};

pub trait Queries {
    fn load_from_npy(path: &str) -> Self;
}

impl Queries for Vec<Vec<AlignedBlock>> {
    fn load_from_npy(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
        assert!(npy.shape().len() == 2);
        let (d1, d2) = (npy.shape()[0] as usize, npy.shape()[1] as usize);

        assert!(d2.is_multiple_of(SIMD_LANECOUNT));

        let mut iter = npy.data::<f32>().unwrap();
        let mut result = Vec::with_capacity(d1);
        for _ in 0..d1 {
            let d2_capacity = d2 / SIMD_LANECOUNT;
            let mut row = Vec::with_capacity(d2_capacity);
            for _ in 0..d2_capacity {
                let mut buffer = [0.0; SIMD_LANECOUNT];
                for entry in buffer.iter_mut() {
                    *entry = iter.next().unwrap().unwrap();
                }
                row.push(AlignedBlock::new(buffer));
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
        let _ = Vec::<Vec<AlignedBlock>>::load_from_npy("test_index/4vecs/4vecs.npy");
    }
}
