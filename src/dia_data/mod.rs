use numpy::ndarray::{Array1, Array4};
use numpy::{PyArray1, PyArray4, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::{prelude::*, Bound};
mod alpha_raw_view;
use crate::dia_data_builder::DIADataBuilder;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;
pub use alpha_raw_view::AlphaRawView;

/// DIAData structure using optimized memory layout for peptide centric querying.
///
/// In data independant acquisition processing (DIA), the data is acquired in spectra with defined cycles of isolation windows.
/// Data is acquired in a spectrum centric way which makes it cheap to get a whole spectrum as it's continous in memory.
///
/// Isolation windows [mz]:
/// spectrum 0: 400-410
/// spectrum 1: 410-420
/// spectrum 2: 420-430
/// ...
/// spectrum 10: 410-420
/// spectrum 11: 420-430
///
/// For DIA data processing, we want to query all spectral information for a single peptide in a single or few isolation windwos.
/// We therefore need a datastructure which allows to query a mass slice fast. DIAData therefore implements an optimized transposed representation of the data.
///
/// Instead of spectra, we group isolation windows into a single QuadrupoleObservation and track their cycle indices.
///
/// QuadrupoleObservation 0: [400-410 cycle 0, 400-410 cycle 1, 400-410 cycle 2, ...]
/// QuadrupoleObservation 1: [410-420 cycle 0, 410-420 cycle 1, 410-420 cycle 2, ...]
/// ...
///
/// We can thereby select the relevant isolation windows by selecting the corresponding QuadrupoleObservation.
/// Within each QuadrupoleObservation, we build a transposed mass index representation.
///
/// Each spectrum consists of tuples of (mz, intensity).
///
/// spectrum 0: 400-410 cycle 0
/// mz: 201.321 intensity: 100.0
/// mz: 254.234 intensity: 200.0
/// ...
/// mz: 821.321 intensity: 300.0
///
/// spectrum 10: 400-410 cycle 1
/// mz: 201.321 intensity: 100.0
/// mz: 259.234 intensity: 500.0
/// ...
/// mz: 725.321 intensity: 100.0
///
/// we map the mz values to a resolution optimized MZIndex with 1 ppm resolution.
///
/// spectrum 0: 400-410 cycle 0
/// mz_index: 1020 intensity: 100.0,
/// mz_index: 2540 intensity: 200.0,
/// ...
/// mz_index: 8210 intensity: 300.0,
///
/// spectrum 10: 400-410 cycle 1
/// mz_index: 1020 intensity: 100.0,
/// mz_index: 2590 intensity: 500.0,
/// ...
/// mz_index: 7250 intensity: 100.0,
///
///
/// then we build a transposed mass index representation which tracks the start anbd the stop of each mz_index slice.
///
/// slice_start[1020] = 0
/// slice_start[1021] = 2
/// slice_start[2540] = 3
/// slice_start[2590] = 4
/// slice_start[7250] = 5
/// slice_start[8210] = 6
///
/// cycle: [0,1,0,1,0,1,]
/// intensity: [100.0, 500.0, 100.0, 500.0, 100.0, 500.0]
///
/// This allows retrieving a slice of the data for a single mz_indexing by reading a constinuous array of cycle indices and intensities.
///
/// my_slice_start = slice_start[1020] = 0
/// my_slice_stop = slice_start[1020 + 1] = 2
///
/// cycle: [0,1]
/// intensity: [100.0, 500.0]
///
#[pyclass]
pub struct DIAData {
    pub rt_index: RTIndex,
    pub quadrupole_observations: Vec<QuadrupoleObservation>,
    pub rt_values: Array1<f32>,
    pub cycle: Array4<f32>,
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
            rt_index: RTIndex::new(),
            quadrupole_observations: Vec::new(),
            rt_values: Array1::zeros((0,)),
            cycle: Array4::zeros((0, 0, 0, 0)),
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
        cycle: PyReadonlyArray4<'py, f32>,
        _py: Python<'py>,
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
            cycle.as_array(),
        );

        // Use optimized builder
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

    /// Returns the memory footprint of the optimized DIAData structure in bytes
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total_size = 0;

        // Size of RTIndex (MZIndex is global and not owned by this struct)
        total_size += self.rt_index.rt.len() * std::mem::size_of::<f32>();

        // Size of quadrupole_observations Vec overhead
        total_size += std::mem::size_of::<Vec<QuadrupoleObservation>>();

        // Size of each optimized QuadrupoleObservation
        for obs in &self.quadrupole_observations {
            total_size += obs.memory_footprint_bytes();
        }

        total_size
    }

    /// Returns the memory footprint in megabytes for easier reading
    pub fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes() as f64 / (1024.0 * 1024.0)
    }

    #[getter]
    pub fn has_mobility(&self) -> bool {
        false
    }

    #[getter]
    pub fn has_ms1(&self) -> bool {
        false
    }

    #[getter]
    pub fn mobility_values(&self) -> Vec<f32> {
        vec![1e-6, 0.0]
    }

    #[getter]
    pub fn rt_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_array(py, &self.rt_values)
    }

    #[getter]
    pub fn cycle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        PyArray4::from_array(py, &self.cycle)
    }
}

// Implement the DIADataTrait for DIAData
impl crate::traits::DIADataTrait for DIAData {
    type QuadrupoleObservation = crate::quadrupole_observation::QuadrupoleObservation;

    fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        self.get_valid_observations(precursor_mz)
    }

    fn mz_index(&self) -> &crate::mz_index::MZIndex {
        MZIndex::global()
    }

    fn rt_index(&self) -> &crate::rt_index::RTIndex {
        &self.rt_index
    }

    fn quadrupole_observations(&self) -> &[Self::QuadrupoleObservation] {
        &self.quadrupole_observations
    }

    fn memory_footprint_bytes(&self) -> usize {
        self.memory_footprint_bytes()
    }
}

#[cfg(test)]
mod tests;
