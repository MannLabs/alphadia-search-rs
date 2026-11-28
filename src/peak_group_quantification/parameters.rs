use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct QuantificationParameters {
    /// Mass tolerance in ppm for fragment matching
    #[pyo3(get, set)]
    pub tolerance_ppm: f32,

    /// Maximum number of fragments to use for quantification per precursor
    #[pyo3(get, set)]
    pub top_k_fragments: usize,

    /// Whether the M+1 XIC of each isotope-eligible fragment adds to its fragment XIC
    #[pyo3(get, set)]
    pub fragment_isotopes: bool,
}

#[pymethods]
impl QuantificationParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            tolerance_ppm: 7.0,
            // maximum number of fragments to use for quantification per precursor. depends on the number of fragments in the precursor.
            // very large number to capture them all by default
            top_k_fragments: 10000,
            fragment_isotopes: false,
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("tolerance_ppm")? {
            self.tolerance_ppm = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("fragment_isotopes")? {
            self.fragment_isotopes = value.extract::<bool>()?;
        }
        Ok(())
    }
}

impl Default for QuantificationParameters {
    fn default() -> Self {
        Self::new()
    }
}
