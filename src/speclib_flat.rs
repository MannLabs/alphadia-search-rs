use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use crate::precursor::Precursor;

#[pyclass]
pub struct SpecLibFlat {
    precursor_idx: Vec<usize>,
    precursor_mz: Vec<f32>,
    precursor_rt: Vec<f32>,
    precursor_start_idx: Vec<usize>,
    precursor_stop_idx: Vec<usize>,
    fragment_mz: Vec<f32>,
    fragment_intensity: Vec<f32>,
}

#[pymethods]
impl SpecLibFlat {
    #[new]
    fn new() -> Self {
        Self {
            precursor_idx: Vec::new(),
            precursor_mz: Vec::new(),
            precursor_rt: Vec::new(),
            precursor_start_idx: Vec::new(),
            precursor_stop_idx: Vec::new(),
            fragment_mz: Vec::new(),
            fragment_intensity: Vec::new(),
        }
    }

    #[staticmethod]
    fn from_arrays(
        precursor_idx: PyReadonlyArray1<'_, usize>,
        precursor_mz: PyReadonlyArray1<'_, f32>,
        precursor_rt: PyReadonlyArray1<'_, f32>,
        precursor_start_idx: PyReadonlyArray1<'_, usize>,
        precursor_stop_idx: PyReadonlyArray1<'_, usize>,
        fragment_mz: PyReadonlyArray1<'_, f32>,
        fragment_intensity: PyReadonlyArray1<'_, f32>,
    ) -> Self {
        Self {
            precursor_idx: precursor_idx.as_array().to_vec(),
            precursor_mz: precursor_mz.as_array().to_vec(),
            precursor_rt: precursor_rt.as_array().to_vec(),
            precursor_start_idx: precursor_start_idx.as_array().to_vec(),
            precursor_stop_idx: precursor_stop_idx.as_array().to_vec(),
            fragment_mz: fragment_mz.as_array().to_vec(),
            fragment_intensity: fragment_intensity.as_array().to_vec(),
        }
    }

    #[getter]
    pub fn num_precursors(&self) -> usize {
        self.precursor_mz.len()
    }

    #[getter]
    pub fn num_fragments(&self) -> usize {
        self.fragment_mz.len()
    }
}

// Regular Rust implementation (not exposed to Python)
impl SpecLibFlat {
    pub fn get_precursor(&self, index: usize) -> Precursor {
        let precursor_idx = self.precursor_idx[index];
        let precursor_mz = self.precursor_mz[index];
        let precursor_rt = self.precursor_rt[index];
        let start_idx = self.precursor_start_idx[index];
        let stop_idx = self.precursor_stop_idx[index];

        let fragment_mz = self.fragment_mz[start_idx..stop_idx].to_vec();
        let fragment_intensity = self.fragment_intensity[start_idx..stop_idx].to_vec();

        Precursor {
            idx: precursor_idx,
            mz: precursor_mz,
            rt: precursor_rt,
            fragment_mz,
            fragment_intensity,
        }
    }
} 