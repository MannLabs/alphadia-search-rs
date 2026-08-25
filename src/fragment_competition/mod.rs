//! Fragment competition: remove candidates that share fragment signal with a
//! higher-priority candidate observed nearby in retention time. Port of
//! `alphadia/fragcomp/fragcomp.py::FragmentCompetition`.
//!
//! The numeric work lives here in Rust — DIA-window assignment, priority ranking
//! and the overlap sweep. Python keeps the dataframe bookkeeping (candidate
//! hashing and fragment index lookup).

mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::WindowBounds;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-facing fragment-competition sweep (thin arrays-in / arrays-out interface).
#[pyclass]
pub struct FragmentCompetition {
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
}

#[pymethods]
impl FragmentCompetition {
    /// Create a fragment-competition sweep.
    ///
    /// * `rt_tol_seconds` — candidates within this RT distance compete.
    /// * `mass_tol_ppm` — fragments within this ppm tolerance are considered the same
    ///   ion.
    #[new]
    fn new(rt_tol_seconds: f32, mass_tol_ppm: f32) -> Self {
        Self {
            rt_tol_seconds,
            mass_tol_ppm,
        }
    }

    /// Sweep all candidates and return a `valid` mask in the order they were passed.
    ///
    /// Candidates may be passed in any order. They are grouped by the DIA isolation
    /// window containing `precursor_mz` and, within a window, ranked by ascending
    /// `proba` (ties broken by ascending `precursor_idx`); of two candidates that
    /// compete for the same fragments, the higher-ranked one survives.
    ///
    /// `fragment_mz` is the full fragment array and `frag_start_idx`/`frag_stop_idx`
    /// are offsets into it. `cycle` is the DIA cycle array, shape
    /// `(1, n_windows, n_scans, 2)`.
    ///
    /// Arrays are borrowed, not copied, so all of them must be C-contiguous.
    ///
    /// Raises `ValueError` if the per-candidate arrays have mismatched lengths or if
    /// a fragment index range falls outside `fragment_mz`, and `TypeError` if an
    /// array is not contiguous.
    #[allow(clippy::too_many_arguments)]
    fn compete<'py>(
        &self,
        py: Python<'py>,
        precursor_mz: PyReadonlyArray1<'_, f32>,
        precursor_idx: PyReadonlyArray1<'_, i64>,
        proba: PyReadonlyArray1<'_, f64>,
        rt_observed: PyReadonlyArray1<'_, f32>,
        frag_start_idx: PyReadonlyArray1<'_, i64>,
        frag_stop_idx: PyReadonlyArray1<'_, i64>,
        fragment_mz: PyReadonlyArray1<'_, f32>,
        cycle: PyReadonlyArray4<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bounds = WindowBounds::from_cycle(cycle.as_array()).map_err(PyValueError::new_err)?;
        let valid = algorithm::compete_for_fragments(
            precursor_mz.as_slice()?,
            precursor_idx.as_slice()?,
            proba.as_slice()?,
            rt_observed.as_slice()?,
            frag_start_idx.as_slice()?,
            frag_stop_idx.as_slice()?,
            fragment_mz.as_slice()?,
            &bounds,
            self.rt_tol_seconds,
            self.mass_tol_ppm,
        )
        .map_err(PyValueError::new_err)?;
        Ok(valid.into_pyarray(py))
    }
}
