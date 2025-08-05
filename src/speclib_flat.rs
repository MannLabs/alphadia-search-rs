use crate::precursor::Precursor;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

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
        // Convert arrays to vectors
        let precursor_idx_vec = precursor_idx.as_array().to_vec();
        let precursor_mz_vec = precursor_mz.as_array().to_vec();
        let precursor_rt_vec = precursor_rt.as_array().to_vec();
        let precursor_start_idx_vec = precursor_start_idx.as_array().to_vec();
        let precursor_stop_idx_vec = precursor_stop_idx.as_array().to_vec();
        let fragment_mz_vec = fragment_mz.as_array().to_vec();
        let fragment_intensity_vec = fragment_intensity.as_array().to_vec();

        // Create indices for sorting
        let mut indices: Vec<usize> = (0..precursor_idx_vec.len()).collect();

        // Sort indices by precursor_idx values
        indices.sort_by_key(|&i| precursor_idx_vec[i]);

        // Reorder all precursor arrays according to sorted indices
        let sorted_precursor_idx: Vec<usize> =
            indices.iter().map(|&i| precursor_idx_vec[i]).collect();
        let sorted_precursor_mz: Vec<f32> = indices.iter().map(|&i| precursor_mz_vec[i]).collect();
        let sorted_precursor_rt: Vec<f32> = indices.iter().map(|&i| precursor_rt_vec[i]).collect();
        let sorted_precursor_start_idx: Vec<usize> = indices
            .iter()
            .map(|&i| precursor_start_idx_vec[i])
            .collect();
        let sorted_precursor_stop_idx: Vec<usize> =
            indices.iter().map(|&i| precursor_stop_idx_vec[i]).collect();

        Self {
            precursor_idx: sorted_precursor_idx,
            precursor_mz: sorted_precursor_mz,
            precursor_rt: sorted_precursor_rt,
            precursor_start_idx: sorted_precursor_start_idx,
            precursor_stop_idx: sorted_precursor_stop_idx,
            fragment_mz: fragment_mz_vec,
            fragment_intensity: fragment_intensity_vec,
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

    pub fn get_precursor_by_idx(&self, precursor_idx: usize) -> Option<Precursor> {
        // Use binary search since precursor_idx is now sorted
        match self.precursor_idx.binary_search(&precursor_idx) {
            Ok(array_index) => Some(self.get_precursor(array_index)),
            Err(_) => None,
        }
    }
}
