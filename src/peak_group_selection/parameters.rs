use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct SelectionParameters {
    #[pyo3(get)]
    pub fwhm_rt: f32,
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub peak_length: usize,
    #[pyo3(get)]
    pub mass_tolerance: f32,
    #[pyo3(get)]
    pub rt_tolerance: f32,
    #[pyo3(get)]
    pub candidate_count: usize,
    #[pyo3(get)]
    pub top_k_fragments: usize,
    #[pyo3(get)]
    pub rt_prior_scale: f32,
    #[pyo3(get)]
    pub rt_prior_df: f32,
}

#[pymethods]
impl SelectionParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            // FWHM of the RT gaussian kernel in seconds.
            fwhm_rt: 3.0,
            // dimensionality of the dense gaussian kernel in number of cycles. Does not need to be changed.
            kernel_size: 20,
            // Length of the peak in number of dia_cycles from the apex. similar to window parameter in DIA-NN. A peak length of 5 means center plus/minus 5 cycles.
            peak_length: 5,
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            mass_tolerance: 7.0,
            // maximum retention time error expected for fragment matching in seconds.
            rt_tolerance: 200.0,
            // maximum number of peak group candidates to select per precursor.
            candidate_count: 3,
            // maximum number of fragments to use for selecting precursors from a DIAData object.
            top_k_fragments: 12,
            // Scale in seconds and degrees of freedom of a Student-t prior on the deviation of a peak from the predicted
            // retention time. Its log density, zero at the predicted RT, is added to the selection score. 0 turns it off.
            rt_prior_scale: 0.0,
            rt_prior_df: 0.0,
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
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("rt_prior_scale")? {
            self.rt_prior_scale = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("rt_prior_df")? {
            self.rt_prior_df = value.extract::<f32>()?;
        }
        Ok(())
    }
}

impl Default for SelectionParameters {
    fn default() -> Self {
        Self::new()
    }
}
