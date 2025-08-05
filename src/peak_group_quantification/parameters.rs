use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct QuantificationParameters {
    /// Mass tolerance in ppm for fragment matching
    #[pyo3(get, set)]
    pub tolerance_ppm: f32,

    /// Maximum number of fragments to use for quantification per precursor
    #[pyo3(get, set)]
    pub top_k_fragments: usize,
}

#[pymethods]
impl QuantificationParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            tolerance_ppm: 20.0,
            top_k_fragments: 50,
        }
    }
}

impl Default for QuantificationParameters {
    fn default() -> Self {
        Self::new()
    }
}
