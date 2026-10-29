//! timsTOF diaPASEF data, read from a Bruker `.d` folder with timsrust.
//!
//! Same layout as `DIAData` (peaks bucketed on the global 1 ppm m/z index and sorted
//! by cycle) with one observation per diaPASEF window group and the scan of every
//! peak kept, so XICs stay 2D (fragments × cycles) while peaks outside the
//! precursor's ion mobility range are left out.

use numpy::ndarray::{Array1, Array4};
use numpy::{PyArray1, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use timsrust::tdf::TdfFrameReader;
use timsrust::TimsTofPath;

mod builder;
mod layout;
mod observation;

use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use crate::traits::{DIADataTrait, MobilityFilter};
use layout::RunLayout;
pub use observation::TimsQuadrupoleObservation;

/// A timsTOF diaPASEF run (diaPASEF, synchro-/slicePASEF) indexed for peptide-centric search.
///
/// Peaks are raw (not centroided) and their intensities are not corrected for the
/// frame accumulation time. MS1 frames only define the cycles; their peaks are not read.
#[pyclass]
pub struct TimsDIAData {
    pub(crate) rt_index: RTIndex,
    pub(crate) quadrupole_observations: Vec<TimsQuadrupoleObservation>,
    /// 1/K0 of every scan, decreasing with the scan index.
    pub(crate) scan_mobility: Vec<f32>,
    /// RT of every frame slot in cycle order (MS1, then one frame per window group).
    pub(crate) rt_values: Array1<f32>,
    pub(crate) cycle: Array4<f32>,
}

#[pymethods]
impl TimsDIAData {
    /// Reads a diaPASEF `.d` folder.
    ///
    /// Raises `ValueError` if the folder cannot be read or is not a diaPASEF run.
    #[staticmethod]
    pub fn from_bruker(path: &str, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            // timsrust can panic on malformed metadata; report it like any read error
            std::panic::catch_unwind(|| Self::read(path))
                .unwrap_or_else(|_| Err(format!("timsrust failed to read {path}")))
        })
        .map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    #[getter]
    pub fn num_peaks(&self) -> usize {
        self.quadrupole_observations
            .iter()
            .map(TimsQuadrupoleObservation::num_peaks)
            .sum()
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.rt_index.len() + self.scan_mobility.len() + self.rt_values.len() + self.cycle.len())
            * std::mem::size_of::<f32>()
            + self
                .quadrupole_observations
                .iter()
                .map(TimsQuadrupoleObservation::memory_footprint_bytes)
                .sum::<usize>()
    }

    pub fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes() as f64 / (1024.0 * 1024.0)
    }

    #[getter]
    pub fn has_mobility(&self) -> bool {
        true
    }

    /// MS1 peaks are not read.
    #[getter]
    pub fn has_ms1(&self) -> bool {
        false
    }

    #[getter]
    pub fn mobility_values(&self) -> Vec<f32> {
        self.scan_mobility.clone()
    }

    #[getter]
    pub fn rt_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_array(py, &self.rt_values)
    }

    /// Isolation windows as `[1, frames per cycle, scans, 2]`, -1 where nothing was isolated
    /// (the MS1 frame and unset scans), like alphaDIA's Bruker cycle.
    #[getter]
    pub fn cycle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        PyArray4::from_array(py, &self.cycle)
    }
}

impl TimsDIAData {
    pub fn read(path: &str) -> Result<Self, String> {
        let path = TimsTofPath::new(path).map_err(|e| e.to_string())?;
        let reader: TdfFrameReader = path.frame_reader().map_err(|e| e.to_string())?;
        let first = reader.iter_indices().min().ok_or("no frames")?;
        let num_scans = reader
            .get_ions(first)
            .map_err(|e| e.to_string())?
            .scan_count();

        let layout = RunLayout::read(&path, &reader, num_scans)?;
        let quadrupole_observations = builder::build_observations(&path, &reader, &layout)?;

        let mut cycle = Array4::from_elem((1, layout.cycle_len(), num_scans, 2), -1.0f32);
        for (group, windows) in layout.scan_windows.iter().enumerate() {
            for (scan, window) in windows.iter().enumerate() {
                if let Some([lo, hi]) = window {
                    cycle[[0, group + 1, scan, 0]] = *lo;
                    cycle[[0, group + 1, scan, 1]] = *hi;
                }
            }
        }

        Ok(Self {
            rt_index: RTIndex::from_rt(Array1::from_vec(layout.cycle_rt_seconds())),
            quadrupole_observations,
            scan_mobility: layout.scan_mobility,
            rt_values: Array1::from_vec(layout.frame_rt_seconds),
            cycle,
        })
    }

    /// Scans `[start, stop)` within the mobility filter; all scans when it has no mobility.
    fn scan_range(&self, mobility: MobilityFilter) -> std::ops::Range<usize> {
        // a library without (valid) mobility does not restrict the scans
        let usable = mobility.mobility.is_finite()
            && mobility.mobility > 0.0
            && mobility.tolerance.is_finite()
            && mobility.tolerance >= 0.0;
        if !usable {
            return 0..self.scan_mobility.len();
        }
        let lower = mobility.mobility - mobility.tolerance;
        let upper = mobility.mobility + mobility.tolerance;
        let start = self.scan_mobility.partition_point(|m| *m > upper);
        let stop = self.scan_mobility.partition_point(|m| *m >= lower);
        start..stop.max(start)
    }
}

impl DIADataTrait for TimsDIAData {
    type QuadrupoleObservation = TimsQuadrupoleObservation;

    fn select_observations(
        &self,
        precursor_mz: f32,
        mobility: MobilityFilter,
    ) -> Vec<(usize, Vec<bool>)> {
        let scans = self.scan_range(mobility);
        self.quadrupole_observations
            .iter()
            .enumerate()
            .filter_map(|(obs_idx, obs)| {
                obs.scan_mask(precursor_mz, scans.clone())
                    .map(|mask| (obs_idx, mask))
            })
            .collect()
    }

    fn mz_index(&self) -> &MZIndex {
        MZIndex::global()
    }

    fn rt_index(&self) -> &RTIndex {
        &self.rt_index
    }

    fn quadrupole_observations(&self) -> &[TimsQuadrupoleObservation] {
        &self.quadrupole_observations
    }

    fn memory_footprint_bytes(&self) -> usize {
        self.memory_footprint_bytes()
    }
}

#[cfg(test)]
mod tests;
