use std::hash::{Hash, Hasher};

use pyo3::{
    Bound, FromPyObject, IntoPyObject, PyErr,
    types::{PyAnyMethods, PyList},
};

/// A Python-compatible wrapper around a vector of f32 values.
///
/// This type bridges between Python lists and Rust Vec<f32>
pub struct VecPy {
    pub inner: Vec<f32>,
}

impl PartialEq for VecPy {
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len()
            && self
                .inner
                .iter()
                .zip(&other.inner)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for VecPy {}

impl Hash for VecPy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &val in &self.inner {
            state.write_u32(val.to_bits());
        }
    }
}

impl AsRef<[f32]> for VecPy {
    fn as_ref(&self) -> &[f32] {
        self.inner.as_ref()
    }
}

/// Explain to Rust how to parse some random python object into an actual Rust vector
/// This involves new allocations because Python cannot be trusted to keep this
/// reference alive.
///
/// This can fail if the random object in question is not a list,
/// in which case it is automatically reported by raising a TypeError exception
/// in the Python code
impl<'a> FromPyObject<'a> for VecPy {
    fn extract_bound(ob: &pyo3::Bound<'a, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let list: Vec<f32> = ob.downcast::<PyList>()?.extract()?;
        Ok(VecPy { inner: list })
    }
}

// Cast back the list of f32's to a Python list
impl<'a> IntoPyObject<'a> for VecPy {
    type Target = PyList;
    type Output = Bound<'a, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'a>) -> Result<Self::Output, Self::Error> {
        let internal = self.inner;
        PyList::new(py, internal)
    }
}

impl Clone for VecPy {
    fn clone(&self) -> Self {
        VecPy {
            inner: self.inner.clone(),
        }
    }
}
