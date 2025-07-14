use pyo3::prelude::*;
use numpy::ndarray::{ArrayBase, ViewRepr, Dim};
use numpy::PyReadonlyArray1;

use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::dia_data_builder::DIADataBuilder;



const _TMP_PATH: &str = "/Users/georgwallmann/Documents/data/alpha-rs/";

pub struct AlphaRawView<'py> {
    pub spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
}

impl<'py> AlphaRawView<'py> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            spectrum_rt,
            peak_mz,
            peak_intensity,
        }
    }
}

#[pyclass]
pub struct DIAData {
    pub mz_index: MZIndex,
    pub rt_index: RTIndex,
    pub quadrupole_observations: Vec<QuadrupoleObservation>,
}

impl Default for DIAData {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl DIAData {
    #[new]
    pub fn new() -> Self {
        Self {
            mz_index: MZIndex::new(),
            rt_index: RTIndex::new(),
            quadrupole_observations: Vec::new(),
        }
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays<'py>(
        spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
        isolation_lower_mz: PyReadonlyArray1<'py, f32>,
        isolation_upper_mz: PyReadonlyArray1<'py, f32>,
        spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
        spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
        spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
        spectrum_rt: PyReadonlyArray1<'py, f32>,
        peak_mz: PyReadonlyArray1<'py, f32>,
        peak_intensity: PyReadonlyArray1<'py, f32>,
        _py: Python<'py>
    ) -> PyResult<Self> {
        let alpha_raw_view = AlphaRawView::new(
            spectrum_delta_scan_idx.as_array(),
            isolation_lower_mz.as_array(),
            isolation_upper_mz.as_array(),
            spectrum_peak_start_idx.as_array(),
            spectrum_peak_stop_idx.as_array(),
            spectrum_cycle_idx.as_array(),
            spectrum_rt.as_array(),
            peak_mz.as_array(),
            peak_intensity.as_array(),
        );

        let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        Ok(dia_data)
    }

    #[getter]
    pub fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    pub fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        let mut valid_observations = Vec::new();
        for (i, obs) in self.quadrupole_observations.iter().enumerate() {
            if obs.isolation_window[0] <= precursor_mz && obs.isolation_window[1] >= precursor_mz {
                valid_observations.push(i);
            }
        }
        valid_observations
    }
} 