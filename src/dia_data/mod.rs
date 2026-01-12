use numpy::ndarray::{Array1, Array4};
use numpy::{PyArray1, PyArray4, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::{prelude::*, Bound};
mod alpha_raw_view;
use crate::dia_data_builder::DIADataBuilder;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;
pub use alpha_raw_view::AlphaRawView;

/// DIAData structure using optimized memory layout for peptide-centric querying.
///
/// # Problem
///
/// In data-independent acquisition (DIA), spectra are acquired in cycles of isolation windows.
/// Raw data is spectrum-centric (contiguous per spectrum), but peptide-centric analysis needs
/// all observations of a given m/z across cycles. DIAData provides a transposed representation
/// optimized for this access pattern.
///
/// # Data Transformation
///
/// ## Input: AlphaRawView (spectrum-centric)
///
/// Consider 1 isolation window (400-425 m/z) acquired over 3 cycles. All spectra with
/// `spectrum_delta_scan_idx == 1` share this window:
///
/// ```text
/// spectrum_idx | spectrum_delta_scan_idx | spectrum_cycle_idx | isolation_window
/// -------------|-------------------------|--------------------|-----------------
///      0       |           1             |         0          |    400-425
///      1       |           1             |         1          |    400-425
///      2       |           1             |         2          |    400-425
///
/// peak_mz and peak_intensity (referenced via spectrum_peak_start/stop_idx):
///
/// Spectrum 0 (cycle 0):        Spectrum 1 (cycle 1):        Spectrum 2 (cycle 2):
///   peak_mz=405.2, int=100       peak_mz=405.2, int=150       peak_mz=410.0, int=80
///   peak_mz=410.0, int=200       peak_mz=420.5, int=300
/// ```
///
/// ## Step 1: Group by spectrum_delta_scan_idx into QuadrupoleObservation
///
/// All spectra with the same `spectrum_delta_scan_idx` are grouped:
///
/// ```text
/// quadrupole_observations[1] = QuadrupoleObservation {
///     isolation_window: [400.0, 425.0],
///     num_cycles: 3,
///     ...
/// }
/// ```
///
/// ## Step 2: Map peak_mz to mz_index via MZIndex::find_closest_index
///
/// Each m/z is mapped to the nearest index in the global MZIndex.
/// For clarity, this example uses small abstract indices:
///
/// ```text
/// MZIndex::find_closest_index(405.2) -> mz_index 0
/// MZIndex::find_closest_index(410.0) -> mz_index 1
/// MZIndex::find_closest_index(420.5) -> mz_index 2
/// ```
///
/// ## Step 3: Build transposed arrays sorted by (mz_index, cycle_indices)
///
/// Peaks are sorted first by mz_index, then by cycle within each mz_index:
///
/// ```text
/// Position | mz_index | cycle_indices | intensities
/// ---------|----------|---------------|------------
///    0     |    0     |       0       |    100
///    1     |    0     |       1       |    150
///    2     |    1     |       0       |    200
///    3     |    1     |       2       |     80
///    4     |    2     |       1       |    300
/// ```
///
/// ## Step 4: Create slice_starts index
///
/// `slice_starts[i]` marks where data for mz_index `i` begins in cycle_indices/intensities.
/// Length is `MZIndex.len() + 1`. For this 3-mz_index example:
///
/// ```text
/// slice_starts: [0, 2, 4, 5]
///                │  │  │  └── end sentinel
///                │  │  └── mz_index 2 starts at position 4
///                │  └── mz_index 1 starts at position 2
///                └── mz_index 0 starts at position 0
///
/// cycle_indices: [0, 1, 0, 2, 1]
/// intensities:   [100, 150, 200, 80, 300]
/// ```
///
/// # Query Example
///
/// To retrieve all observations of m/z ~405.2 (mz_index 0):
///
/// ```text
/// let (cycles, ints) = observation.get_slice_data(0);
/// // start = slice_starts[0] = 0
/// // stop  = slice_starts[1] = 2
///
/// cycles = &cycle_indices[0..2] = [0, 1]   // Observed in cycles 0 and 1
/// ints   = &intensities[0..2]   = [100, 150]
/// ```
///
/// # Benefits
///
/// - Memory and allocation optimized storage of sparse arrays instead of list of structs
/// - O(1) slice lookup with contiguous memory reads, replacing scattered access across spectra
///
/// # Constraints
///
/// - Requires building transposed structure once upfront
/// - Upper limit of resolution needs to be known upfront
/// - The DIA cycle needs to be consistent across all spectra
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

    /// Returns the memory footprint of the optimized DIAData structure in bytes.
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

    /// Returns the memory footprint in megabytes for easier reading.
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
