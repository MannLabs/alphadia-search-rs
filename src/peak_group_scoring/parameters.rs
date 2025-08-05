use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Clone)]
pub struct ScoringParameters {
    #[pyo3(get)]
    pub mass_tolerance: f32,
    #[pyo3(get)]
    pub top_k_fragments: usize,
}

#[pymethods]
impl ScoringParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            mass_tolerance: 7.0,
            top_k_fragments: 12,
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("mass_tolerance")? {
            self.mass_tolerance = value.extract::<f32>()?;
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
