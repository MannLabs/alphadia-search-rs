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
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            tolerance_ppm: 7.0,
            // maximum number of fragments to use for quantification per precursor. depends on the number of fragments in the precursor.
            top_k_fragments: 100,
        }
    }
}

impl Default for QuantificationParameters {
    fn default() -> Self {
        Self::new()
    }
}
