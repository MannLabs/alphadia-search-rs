//! Fragment competition. Only one candidate can own a fragment signal. This module removes a
//! candidate if a better candidate near it in retention time claims the same ions.
//!
//! Python does only the dataframe work: it makes the candidate hashes and finds the fragment
//! indexes. All calculations occur here.

mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::WindowBounds;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Arrays in, mask out.
///
/// The description below comes from `src/fragment_competition/DOCS.md`. It gives the problem,
/// the input and output arrays, and the algorithm.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct FragmentCompetition {
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
}

#[pymethods]
impl FragmentCompetition {
    /// * `rt_tol_seconds` - two candidates compete if their retention times are closer than
    ///   this value.
    /// * `mass_tol_ppm` - two fragments are the same ion if their m/z are closer than this
    ///   value.
    #[new]
    fn new(rt_tol_seconds: f32, mass_tol_ppm: f32) -> Self {
        Self {
            rt_tol_seconds,
            mass_tol_ppm,
        }
    }

    /// Gives a `valid` mask in the candidate order of the caller.
    ///
    /// Candidates compete only inside their own DIA window, and the lower `proba` wins. The
    /// candidate order has no effect, thus the caller does not sort the arrays.
    ///
    /// `fragment_mz` holds the ions of all candidates. `frag_start_idx` and `frag_stop_idx`
    /// give the range of each candidate. `cycle` has the shape `(1, n_windows, n_scans, 2)`.
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the array lengths do not agree, if a fragment range is outside
    /// `fragment_mz`, or if a float input contains NaN. Raises `TypeError` if an array is not
    /// contiguous.
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
