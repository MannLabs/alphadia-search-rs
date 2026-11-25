//! timsTOF diaPASEF data, read from a Bruker `.d` folder with timsrust.
//!
//! Same layout as `DIAData` (peaks bucketed on the global 1 ppm m/z index and sorted
//! by cycle) with one observation per diaPASEF window group and the scan of every
//! peak kept, so XICs stay 2D (fragments × cycles) while peaks outside the
//! precursor's ion mobility range are left out.
//!
//! Built only with the `tims` cargo feature, which pulls in timsrust. The Python wheel
//! enables it (`pyproject.toml`); plain `cargo build`/`cargo test` leave it out.

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
use crate::traits::{DIADataTrait, MobilityFilter, MobilityMoments};
use layout::RunLayout;
pub use observation::TimsQuadrupoleObservation;

/// A timsTOF diaPASEF run (diaPASEF, synchro-/slicePASEF) indexed for peptide-centric search.
///
/// Peaks are raw (not centroided) and their intensities are not corrected for the
/// frame accumulation time. MS1 frames define the cycles; their peaks, if read, are one observation after the window
/// groups.
#[pyclass]
pub struct TimsDIAData {
    pub(crate) rt_index: RTIndex,
    pub(crate) quadrupole_observations: Vec<TimsQuadrupoleObservation>,
    /// 1/K0 of every scan, decreasing with the scan index.
    pub(crate) scan_mobility: Vec<f32>,
    /// RT of every frame slot in cycle order (MS1, then one frame per window group).
    pub(crate) rt_values: Array1<f32>,
    pub(crate) cycle: Array4<f32>,
    /// Number of diaPASEF window groups; the observations after them hold the MS1 frames.
    pub(crate) num_groups: usize,
}

#[pymethods]
impl TimsDIAData {
    /// Reads a diaPASEF `.d` folder, with its MS1 frames if `read_ms1`.
    ///
    /// Raises `ValueError` if the folder cannot be read or is not a diaPASEF run.
    #[staticmethod]
    #[pyo3(signature = (path, read_ms1 = true))]
    pub fn from_bruker(path: &str, read_ms1: bool, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            // timsrust can panic on malformed metadata; report it like any read error
            std::panic::catch_unwind(|| Self::read(path, read_ms1)).unwrap_or_else(|payload| {
                Err(format!(
                    "timsrust failed to read {path}: {}",
                    panic_message(payload.as_ref())
                ))
            })
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

    /// Whether the MS1 frames were read.
    #[getter]
    pub fn has_ms1(&self) -> bool {
        self.quadrupole_observations.len() > self.num_groups
    }

    #[getter]
    pub fn mobility_values(&self) -> Vec<f32> {
        self.scan_mobility.clone()
    }

    #[getter]
    pub fn rt_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_array(py, &self.rt_values)
    }

    /// Per-fragment intensity patches around given apexes, for offline analyses of the raw evidence: for each
    /// (precursor, apex cycle, apex mobility) the `top_k_fragments` most intense library fragments' summed peak
    /// intensity on each of `2 * cycle_radius + 1` cycles x `n_scans` scans centred on the apex (zero outside the run,
    /// outside the precursor's isolation windows and for missing fragments), and those fragments' library intensities.
    /// Returns (patches [n, top_k, cycles, scans], library intensities [n, top_k], 1/K0 of the patch's first scan [n]).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (speclib, precursor_idx, cycle_center, mobility, cycle_radius, n_scans, mass_tolerance, top_k_fragments))]
    pub fn mobility_patches<'py>(
        &self,
        py: Python<'py>,
        speclib: &crate::SpecLibFlat,
        precursor_idx: numpy::PyReadonlyArray1<'_, usize>,
        cycle_center: numpy::PyReadonlyArray1<'_, usize>,
        mobility: numpy::PyReadonlyArray1<'_, f32>,
        cycle_radius: usize,
        n_scans: usize,
        mass_tolerance: f32,
        top_k_fragments: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, numpy::PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        use rayon::prelude::*;
        let precursor_idx = precursor_idx.as_slice()?;
        let cycle_center = cycle_center.as_slice()?;
        let mobility = mobility.as_slice()?;
        let n = precursor_idx.len();
        if cycle_center.len() != n || mobility.len() != n {
            return Err(PyValueError::new_err(
                "precursor_idx, cycle_center and mobility need the same length",
            ));
        }
        let n_cycles = 2 * cycle_radius + 1;
        let num_cycles = self.rt_index.rt.len();
        let scan_width = (self.scan_mobility[0] - self.scan_mobility[self.scan_mobility.len() - 1])
            .abs()
            / (self.scan_mobility.len() - 1) as f32;
        let half_width = 0.5 * n_scans as f32 * scan_width;
        let results: Vec<Result<(Vec<f32>, Vec<f32>, f32), String>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let position = speclib.position_of(precursor_idx[i]).ok_or_else(|| {
                    format!("precursor_idx {} is not in the library", precursor_idx[i])
                })?;
                let precursor =
                    speclib.get_precursor_filtered(position, true, true, false, top_k_fragments);
                let filter = MobilityFilter::Window {
                    mobility: mobility[i],
                    tolerance: half_width,
                };
                let range = self.scan_range(filter);
                let scan_start = range.start;
                let observations = self.select_observations(precursor.mz, filter);
                let first_cycle = cycle_center[i] as i64 - cycle_radius as i64;
                let mut patches = vec![0.0f32; top_k_fragments * n_cycles * n_scans];
                let mut library = vec![0.0f32; top_k_fragments];
                for (f, &mz) in precursor
                    .fragment_mz
                    .iter()
                    .enumerate()
                    .take(top_k_fragments)
                {
                    library[f] = precursor.fragment_intensity[f];
                    let lo = first_cycle.max(0) as usize;
                    let hi = ((first_cycle + n_cycles as i64).max(0) as usize).min(num_cycles);
                    if hi <= lo {
                        continue;
                    }
                    let mut patch = numpy::ndarray::Array2::<f32>::zeros((hi - lo, n_scans));
                    self.fill_mobility_patch(
                        &observations,
                        lo,
                        scan_start,
                        mass_tolerance,
                        mz,
                        &mut patch,
                    );
                    let row_offset = (lo as i64 - first_cycle) as usize;
                    for r in 0..(hi - lo) {
                        for s in 0..n_scans {
                            patches[(f * n_cycles + r + row_offset) * n_scans + s] = patch[[r, s]];
                        }
                    }
                }
                let first_mobility = self
                    .scan_mobility
                    .get(scan_start)
                    .copied()
                    .unwrap_or(f32::NAN);
                Ok((patches, library, first_mobility))
            })
            .collect();
        let mut all_patches = Vec::with_capacity(n * top_k_fragments * n_cycles * n_scans);
        let mut all_library = Vec::with_capacity(n * top_k_fragments);
        let mut first = Vec::with_capacity(n);
        for result in results {
            let (patches, library, first_mobility) = result.map_err(PyValueError::new_err)?;
            all_patches.extend(patches);
            all_library.extend(library);
            first.push(first_mobility);
        }
        let patches = Array4::from_shape_vec((n, top_k_fragments, n_cycles, n_scans), all_patches)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let library = numpy::ndarray::Array2::from_shape_vec((n, top_k_fragments), all_library)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            PyArray4::from_array(py, &patches),
            numpy::PyArray2::from_array(py, &library),
            PyArray1::from_vec(py, first),
        ))
    }

    /// Isolation windows as `[1, frames per cycle, scans, 2]`, -1 where nothing was isolated
    /// (the MS1 frame and unset scans), like alphaDIA's Bruker cycle.
    #[getter]
    pub fn cycle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        PyArray4::from_array(py, &self.cycle)
    }
}

/// The text of a caught panic, when it carries one.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no panic message")
}

impl TimsDIAData {
    pub fn read(path: &str, read_ms1: bool) -> Result<Self, String> {
        let path = TimsTofPath::new(path).map_err(|e| e.to_string())?;
        let reader: TdfFrameReader = path.frame_reader().map_err(|e| e.to_string())?;
        let num_scans = builder::first_scan_count(&reader)?;
        if num_scans > usize::from(u16::MAX) + 1 {
            return Err(format!(
                "{num_scans} scans per frame; more than 65,536 are not supported"
            ));
        }

        let layout = RunLayout::read(&path, &reader, num_scans)?;
        let mut quadrupole_observations = builder::build_observations(&path, &reader, &layout)?;
        let num_groups = quadrupole_observations.len();
        if read_ms1 {
            quadrupole_observations.push(builder::build_ms1_observation(&path, &reader, &layout)?);
        }

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
            num_groups,
        })
    }

    /// Scans `[start, stop)` within the mobility filter; all scans for `MobilityFilter::Any`.
    fn scan_range(&self, filter: MobilityFilter) -> std::ops::Range<usize> {
        let (mobility, tolerance) = match filter {
            MobilityFilter::Any => return 0..self.scan_mobility.len(),
            MobilityFilter::Window {
                mobility,
                tolerance,
            } => (mobility, tolerance),
        };
        let lower = mobility - tolerance;
        let upper = mobility + tolerance;
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

    fn mobility_moments(
        &self,
        observations: &[(usize, Vec<bool>)],
        cycle_start: usize,
        cycle_stop: usize,
        mass_tolerance: f32,
        mz: f32,
        reference: f32,
        outer: f32,
    ) -> Option<MobilityMoments> {
        let mut moments = MobilityMoments::default();
        for (obs_idx, scan_mask) in observations {
            self.quadrupole_observations[*obs_idx].for_each_peak(
                MZIndex::global(),
                scan_mask,
                cycle_start,
                cycle_stop,
                mass_tolerance,
                mz,
                |_, intensity, _, scan| {
                    let w = intensity as f64;
                    let m = self.scan_mobility[scan] as f64;
                    moments.sums[0] += w;
                    moments.sums[1] += w * m;
                    moments.sums[2] += w * m * m;
                    if (m - reference as f64).abs() > outer as f64 {
                        moments.outer_weight += w;
                    }
                },
            );
        }
        Some(moments)
    }

    /// The MS1 frame opens the cycle (slot 0), window group `g` is slot `g + 1`.
    fn cycle_position(&self, obs_idx: usize) -> f32 {
        let slot = if obs_idx >= self.num_groups {
            0
        } else {
            obs_idx + 1
        };
        slot as f32 / (self.num_groups + 1) as f32
    }

    fn mobility_scans(&self, mobility: MobilityFilter) -> Option<(usize, usize, Vec<f32>)> {
        let range = self.scan_range(mobility);
        Some((range.start, range.end, self.scan_mobility[range].to_vec()))
    }

    fn fill_mobility_patch(
        &self,
        observations: &[(usize, Vec<bool>)],
        cycle_start: usize,
        scan_start: usize,
        mass_tolerance: f32,
        mz: f32,
        patch: &mut numpy::ndarray::Array2<f32>,
    ) {
        let (n_cycles, n_scans) = patch.dim();
        for (obs_idx, scan_mask) in observations {
            self.quadrupole_observations[*obs_idx].for_each_peak(
                MZIndex::global(),
                scan_mask,
                cycle_start,
                cycle_start + n_cycles,
                mass_tolerance,
                mz,
                |cycle, intensity, _, scan| {
                    if scan >= scan_start && scan - scan_start < n_scans {
                        patch[[cycle, scan - scan_start]] += intensity;
                    }
                },
            );
        }
    }

    fn ms1_scan_filter(&self, _obs_idx: usize, mobility: MobilityFilter) -> Vec<bool> {
        let mut mask = vec![false; self.scan_mobility.len()];
        mask[self.scan_range(mobility)].fill(true);
        mask
    }
}

#[cfg(test)]
mod tests;
