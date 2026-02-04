use crate::numerics::{AlignedBlock, SIMD_LANECOUNT};

/// A trait for loading query vectors from NumPy format files.
///
/// Implementations handle parsing .npy files and converting flat f32 arrays
/// into SIMD-aligned block representations suitable for distance computations.
pub trait Queries {
    /// Loads query vectors from a NumPy .npy file.
    ///
    /// # Arguments
    /// * `path` - Path to the .npy file containing query vectors
    ///
    /// # Returns
    /// The loaded queries in the implementing type's format
    ///
    /// # Panics
    /// * Panics if the file cannot be read
    /// * Panics if the .npy format is invalid
    /// * Panics if the shape is not 2-dimensional
    /// * Panics if the vector dimension is not a multiple of `SIMD_LANECOUNT`
    fn load_from_npy(path: &str) -> Self;
}

impl Queries for Vec<Vec<AlignedBlock>> {
    /// Loads a 2D array of f32 vectors from a NumPy file and converts to aligned blocks.
    ///
    /// Expects a 2D NumPy array where each row is a query vector. Vectors are converted
    /// to sequences of `AlignedBlock` instances for SIMD-accelerated distance computation.
    ///
    /// # Arguments
    /// * `path` - Path to the .npy file
    ///
    /// # Returns
    /// A vector of queries, where each query is a vector of `AlignedBlock` instances
    ///
    /// # Panics
    /// * Panics if the file cannot be read
    /// * Panics if the .npy data is not 2-dimensional
    /// * Panics if the vector dimension is not a multiple of `SIMD_LANECOUNT`
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
        let _ = Vec::<Vec<AlignedBlock>>::load_from_npy("test_index/vectors.npy");
    }
}
