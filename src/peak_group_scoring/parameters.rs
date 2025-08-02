use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Clone)]
pub struct ScoringParameters {
    #[pyo3(get)]
    pub mass_tolerance: f32,

}

#[pymethods]
impl ScoringParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            fwhm_rt: 3.0,
            kernel_size: 20,
            peak_length: 5,
            mass_tolerance: 7.0,
            rt_tolerance: 200.0,
            candidate_count: 3,
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("fwhm_rt")? {
            self.fwhm_rt = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("kernel_size")? {
            self.kernel_size = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("peak_length")? {
            self.peak_length = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("mass_tolerance")? {
            self.mass_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("rt_tolerance")? {
            self.rt_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("candidate_count")? {
            self.candidate_count = value.extract::<usize>()?;
        }
        Ok(())
    }
}

impl Default for SelectionParameters {
    fn default() -> Self {
        Self::new()
    }
}
