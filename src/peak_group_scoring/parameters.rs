use crate::traits::{check_mobility_tolerance, DEFAULT_MOBILITY_TOLERANCE};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ScoringParameters {
    #[pyo3(get)]
    pub mass_tolerance: f32,
    #[pyo3(get)]
    pub mobility_tolerance: f32,
    #[pyo3(get)]
    pub top_k_fragments: usize,
}

#[pymethods]
impl ScoringParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            mass_tolerance: 7.0,
            // maximum ion mobility (1/K0) deviation from the precursor's library value, ignored for data without mobility.
            mobility_tolerance: DEFAULT_MOBILITY_TOLERANCE,
            // maximum number of fragments to use for scoring per precursor. depends on the number of fragments in the precursor.
            top_k_fragments: 100,
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("mass_tolerance")? {
            self.mass_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("mobility_tolerance")? {
            self.mobility_tolerance =
                check_mobility_tolerance(value.extract::<f32>()?).map_err(PyValueError::new_err)?;
        }
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        Ok(())
    }
}

impl Default for ScoringParameters {
    fn default() -> Self {
        Self::new()
    }
}
