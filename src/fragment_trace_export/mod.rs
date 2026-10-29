//! Per-fragment raw traces of scored candidates, exported for offline feature work (replay dumps).
//!
//! For each candidate: its top fragments in alphaDIA's riding-slot order (non-zero library intensity, no y1, by
//! library intensity), and over a fixed window of cycles around the scored cycles the fragment XICs, their
//! intensity-weighted m/z error, the XICs of each fragment's M+1 isotope and of a half-spacing companion, plus at the
//! candidate's apex cycle the number of occupied m/z bins near each fragment and the TIC of its isolation window(s).
//! Nothing here feeds the search.

#[cfg(test)]
mod tests;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::DIAData;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::speclib_flat::SpecLibFlat;
use crate::traits::DIADataTrait;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use rayon::prelude::*;

/// Mass difference between 13C and 12C in Da.
pub const C13_MASS_DELTA: f32 = 1.003_354_8;

/// Traces exported per candidate: the fragment itself, its M+1 isotope and its half-spacing companion.
const TRACES: usize = 3;

#[derive(Debug, Clone)]
pub struct TraceParameters {
    /// Fragment extraction tolerance in ppm (the scoring's)
    pub mass_tolerance: f32,
    /// Fragments per candidate
    pub top_k: usize,
    /// Cycles exported before the scored window's start
    pub extend: usize,
    /// Exported cycles in total, starting `extend` cycles before the scored window's start
    pub window: usize,
    /// Half-width in ppm around a fragment in which occupied m/z bins are counted at the apex
    pub neighbour_ppm: f32,
}

/// The exported values of one candidate, flattened [fragment, cycle] or [fragment].
#[derive(Debug, Clone)]
pub struct CandidateTraces {
    pub window_start: i32,
    pub fragment_mz: Vec<f32>,
    pub fragment_mz_library: Vec<f32>,
    pub library_intensity: Vec<f32>,
    pub fragment_charge: Vec<u8>,
    pub companion_offset: Vec<f32>,
    pub xic: Vec<f32>,
    pub mz_error_ppm: Vec<f32>,
    pub isotope_xic: Vec<f32>,
    pub companion_xic: Vec<f32>,
    pub apex_neighbours: Vec<u16>,
    pub apex_tic: f32,
}

/// m/z offset of the half-spacing companion: for a 1+ fragment the would-be 2+ isotope (+C13/2), for a 2+ or higher
/// fragment the would-be 1+ isotope (+C13).
pub fn companion_offset(charge: u8) -> f32 {
    if charge <= 1 {
        C13_MASS_DELTA / 2.0
    } else {
        C13_MASS_DELTA
    }
}

/// Summed intensity per (observation, cycle) over all m/z bins.
pub fn observation_tic(dia_data: &DIAData, num_cycles: usize) -> Array2<f32> {
    let observations = dia_data.quadrupole_observations();
    let rows: Vec<Vec<f32>> = observations
        .par_iter()
        .map(|obs| {
            let mut tic = vec![0.0f32; num_cycles];
            for (&cycle, &intensity) in obs.cycle_indices.iter().zip(&obs.intensities) {
                if (cycle as usize) < num_cycles {
                    tic[cycle as usize] += intensity;
                }
            }
            tic
        })
        .collect();
    let mut out = Array2::<f32>::zeros((observations.len(), num_cycles));
    for (i, row) in rows.iter().enumerate() {
        out.row_mut(i).assign(&Array1::from(row.clone()));
    }
    out
}

/// Occupied m/z bins with intensity at `cycle` within `ppm` of `mz`.
fn occupied_bins(
    obs: &QuadrupoleObservation,
    mz_index: &MZIndex,
    mz: f32,
    ppm: f32,
    cycle: usize,
) -> u16 {
    let delta = mz * ppm * 1e-6;
    let mut count = 0u16;
    for mz_idx in mz_index.mz_range_indices(mz - delta, mz + delta) {
        let (cycles, intensities) = obs.get_slice_data(mz_idx);
        if let Ok(pos) = cycles.binary_search(&(cycle as u16)) {
            if intensities[pos] > 0.0 {
                count = count.saturating_add(1);
            }
        }
    }
    count
}

pub fn candidate_traces(
    dia_data: &DIAData,
    lib: &SpecLibFlat,
    candidate: &Candidate,
    tic: &numpy::ndarray::ArrayView2<f32>,
    params: &TraceParameters,
) -> CandidateTraces {
    let k = params.top_k;
    let w = params.window;
    let num_cycles = dia_data.rt_index().rt.len();
    let window_start = candidate.cycle_start as i64 - params.extend as i64;
    let mut out = CandidateTraces {
        window_start: window_start as i32,
        fragment_mz: vec![0.0; k],
        fragment_mz_library: vec![0.0; k],
        library_intensity: vec![0.0; k],
        fragment_charge: vec![0; k],
        companion_offset: vec![0.0; k],
        xic: vec![0.0; k * w],
        mz_error_ppm: vec![f32::NAN; k * w],
        isotope_xic: vec![0.0; k * w],
        companion_xic: vec![0.0; k * w],
        apex_neighbours: vec![0; k],
        apex_tic: 0.0,
    };
    let Some((precursor_mz, fragments)) =
        lib.top_fragments_by_intensity(candidate.precursor_idx, k)
    else {
        return out;
    };
    let n = fragments.len();
    let mut trace_mz = Vec::with_capacity(TRACES * n);
    for (slot, &(mz, mz_library, intensity, charge)) in fragments.iter().enumerate() {
        out.fragment_mz[slot] = mz;
        out.fragment_mz_library[slot] = mz_library;
        out.library_intensity[slot] = intensity;
        out.fragment_charge[slot] = charge;
        out.companion_offset[slot] = companion_offset(charge);
        trace_mz.push(mz);
    }
    for &(mz, _, _, charge) in &fragments {
        trace_mz.push(mz + C13_MASS_DELTA / charge.max(1) as f32);
    }
    for &(mz, _, _, charge) in &fragments {
        trace_mz.push(mz + companion_offset(charge));
    }

    let first = window_start.max(0) as usize;
    let last = ((window_start + w as i64).max(0) as usize).min(num_cycles);
    let valid = dia_data.get_valid_observations(precursor_mz);
    if first < last && n > 0 {
        let dense = DenseXICMZObservation::from_observations(
            dia_data,
            valid.clone(),
            first,
            last,
            params.mass_tolerance,
            &trace_mz,
        );
        let offset = (first as i64 - window_start) as usize;
        for slot in 0..n {
            for c in 0..(last - first) {
                let at = slot * w + offset + c;
                let intensity = dense.dense_xic[[slot, c]];
                out.xic[at] = intensity;
                if intensity > 0.0 {
                    let mz = fragments[slot].0;
                    out.mz_error_ppm[at] = (dense.dense_mz[[slot, c]] - mz) / mz * 1e6;
                }
                out.isotope_xic[at] = dense.dense_xic[[n + slot, c]];
                out.companion_xic[at] = dense.dense_xic[[2 * n + slot, c]];
            }
        }
    }

    let apex = candidate.cycle_center;
    if apex < num_cycles {
        let observations = dia_data.quadrupole_observations();
        for &obs_idx in &valid {
            out.apex_tic += tic[[obs_idx, apex]];
            for (slot, &(mz, _, _, _)) in fragments.iter().enumerate() {
                out.apex_neighbours[slot] =
                    out.apex_neighbours[slot].saturating_add(occupied_bins(
                        &observations[obs_idx],
                        dia_data.mz_index(),
                        mz,
                        params.neighbour_ppm,
                        apex,
                    ));
            }
        }
    }
    out
}

/// Exports per-fragment traces of scored candidates for offline analysis.
#[pyclass]
pub struct FragmentTraceExport {
    params: TraceParameters,
}

#[pymethods]
impl FragmentTraceExport {
    /// * `mass_tolerance` - fragment extraction tolerance in ppm.
    /// * `top_k` - fragments per candidate.
    /// * `extend` - cycles exported before the scored window's start.
    /// * `window` - exported cycles in total.
    /// * `neighbour_ppm` - half-width in ppm of the occupied-bin count at the apex.
    #[new]
    #[pyo3(signature = (mass_tolerance, top_k = 12, extend = 6, window = 19, neighbour_ppm = 50.0))]
    fn new(
        mass_tolerance: f32,
        top_k: usize,
        extend: usize,
        window: usize,
        neighbour_ppm: f32,
    ) -> Self {
        Self {
            params: TraceParameters {
                mass_tolerance,
                top_k,
                extend,
                window,
                neighbour_ppm,
            },
        }
    }

    /// Summed intensity per (observation, cycle), the `tic` input of `export`.
    #[staticmethod]
    fn observation_tic<'py>(
        py: Python<'py>,
        dia_data: &DIAData,
    ) -> Bound<'py, numpy::PyArray2<f32>> {
        let num_cycles = dia_data.rt_index().rt.len();
        observation_tic(dia_data, num_cycles).into_pyarray(py)
    }

    /// A dict of arrays with one row per candidate, in the candidate order of the caller.
    fn export(
        &self,
        py: Python<'_>,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
        tic: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Py<PyAny>> {
        let tic = tic.as_array();
        let candidate_list: Vec<&Candidate> = candidates.iter().collect();
        let traces: Vec<CandidateTraces> = candidate_list
            .par_iter()
            .map(|c| candidate_traces(dia_data, lib, c, &tic, &self.params))
            .collect();

        let n = traces.len();
        let k = self.params.top_k;
        let w = self.params.window;
        let flat2 = |get: &dyn Fn(&CandidateTraces) -> &Vec<f32>| -> Array2<f32> {
            Array2::from_shape_vec(
                (n, k),
                traces.iter().flat_map(|t| get(t).iter().copied()).collect(),
            )
            .unwrap()
        };
        let flat3 = |get: &dyn Fn(&CandidateTraces) -> &Vec<f32>| -> Array3<f32> {
            Array3::from_shape_vec(
                (n, k, w),
                traces.iter().flat_map(|t| get(t).iter().copied()).collect(),
            )
            .unwrap()
        };

        let dict = PyDict::new(py);
        let precursor_idx: Vec<u64> = candidate_list
            .iter()
            .map(|c| c.precursor_idx as u64)
            .collect();
        let rank: Vec<u64> = candidate_list.iter().map(|c| c.rank as u64).collect();
        let cycle_start: Vec<i32> = candidate_list
            .iter()
            .map(|c| c.cycle_start as i32)
            .collect();
        let cycle_stop: Vec<i32> = candidate_list.iter().map(|c| c.cycle_stop as i32).collect();
        let cycle_center: Vec<i32> = candidate_list
            .iter()
            .map(|c| c.cycle_center as i32)
            .collect();
        dict.set_item("precursor_idx", precursor_idx.into_pyarray(py))?;
        dict.set_item("rank", rank.into_pyarray(py))?;
        dict.set_item("cycle_start", cycle_start.into_pyarray(py))?;
        dict.set_item("cycle_stop", cycle_stop.into_pyarray(py))?;
        dict.set_item("cycle_center", cycle_center.into_pyarray(py))?;
        let window_start: Vec<i32> = traces.iter().map(|t| t.window_start).collect();
        dict.set_item("window_start", window_start.into_pyarray(py))?;
        let apex_tic: Vec<f32> = traces.iter().map(|t| t.apex_tic).collect();
        dict.set_item("apex_tic", apex_tic.into_pyarray(py))?;
        dict.set_item("fragment_mz", flat2(&|t| &t.fragment_mz).into_pyarray(py))?;
        dict.set_item(
            "fragment_mz_library",
            flat2(&|t| &t.fragment_mz_library).into_pyarray(py),
        )?;
        dict.set_item(
            "library_intensity",
            flat2(&|t| &t.library_intensity).into_pyarray(py),
        )?;
        dict.set_item(
            "companion_offset",
            flat2(&|t| &t.companion_offset).into_pyarray(py),
        )?;
        let charge = Array2::from_shape_vec(
            (n, k),
            traces
                .iter()
                .flat_map(|t| t.fragment_charge.iter().copied())
                .collect(),
        )
        .unwrap();
        dict.set_item("fragment_charge", charge.into_pyarray(py))?;
        let neighbours = Array2::from_shape_vec(
            (n, k),
            traces
                .iter()
                .flat_map(|t| t.apex_neighbours.iter().copied())
                .collect(),
        )
        .unwrap();
        dict.set_item("apex_neighbours", neighbours.into_pyarray(py))?;
        dict.set_item("xic", flat3(&|t| &t.xic).into_pyarray(py))?;
        dict.set_item("mz_error_ppm", flat3(&|t| &t.mz_error_ppm).into_pyarray(py))?;
        dict.set_item("isotope_xic", flat3(&|t| &t.isotope_xic).into_pyarray(py))?;
        dict.set_item(
            "companion_xic",
            flat3(&|t| &t.companion_xic).into_pyarray(py),
        )?;
        Ok(dict.into())
    }
}
