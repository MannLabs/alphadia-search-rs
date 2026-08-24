//! Fragment competition: remove candidates that share fragment signal with a
//! higher-priority candidate observed nearby in retention time. Port of
//! `alphadia/fragcomp/fragcomp.py::FragmentCompetition`.
//!
//! The numeric sweep lives here in Rust; Python keeps the dataframe bookkeeping
//! (candidate hashing, fragment index lookup, DIA-window assignment, sorting).

mod algorithm;
#[cfg(test)]
mod tests;

use numpy::{IntoPyArray, PyArray1};
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

    /// Sweep all candidates and return a `valid` mask.
    ///
    /// `window_idx`, `rt_observed`, `frag_start_idx` and `frag_stop_idx` are per-PSM
    /// arrays and must already be sorted by `(window_idx, proba, precursor_idx)`
    /// ascending: `window_idx` groups must be contiguous, and the order within a
    /// group determines who wins a conflict (the earlier entry survives). `fragment_mz`
    /// is the full fragment array; `frag_start_idx`/`frag_stop_idx` are offsets into
    /// it.
    ///
    /// Raises `ValueError` if the per-PSM arrays have mismatched lengths.
    fn compete<'py>(
        &self,
        py: Python<'py>,
        window_idx: Vec<i64>,
        rt_observed: Vec<f32>,
        frag_start_idx: Vec<i64>,
        frag_stop_idx: Vec<i64>,
        fragment_mz: Vec<f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let valid = algorithm::compete_for_fragments(
            &window_idx,
            &rt_observed,
            &frag_start_idx,
            &frag_stop_idx,
            &fragment_mz,
            self.rt_tol_seconds,
            self.mass_tol_ppm,
        )
        .map_err(PyValueError::new_err)?;
        Ok(valid.into_pyarray(py))
    }
}
