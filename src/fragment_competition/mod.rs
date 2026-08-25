//! Fragment competition: drop candidates whose fragment signal is already claimed
//! by a better-scoring candidate nearby in retention time.
//!
//! Python keeps only the dataframe bookkeeping (candidate hashing and fragment
//! index lookup); everything numeric happens here.

mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::WindowBounds;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Arrays in, mask out.
#[pyclass]
pub struct FragmentCompetition {
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
}

#[pymethods]
impl FragmentCompetition {
    /// * `rt_tol_seconds` - candidates closer than this in RT compete.
    /// * `mass_tol_ppm` - fragments closer than this count as the same ion.
    #[new]
    fn new(rt_tol_seconds: f32, mass_tol_ppm: f32) -> Self {
        Self {
            rt_tol_seconds,
            mass_tol_ppm,
        }
    }

    /// Returns a `valid` mask in the order the candidates were passed.
    ///
    /// Candidates only compete inside their own DIA window, and the better `proba`
    /// wins. Order does not matter, so callers need not sort anything.
    ///
    /// `fragment_mz` holds every candidate's ions and `frag_start_idx`/`frag_stop_idx`
    /// slice into it. `cycle` has shape `(1, n_windows, n_scans, 2)`.
    ///
    /// Arrays are borrowed rather than copied, so they must be C-contiguous.
    ///
    /// Raises `ValueError` on mismatched lengths or an out-of-range fragment slice,
    /// `TypeError` on a non-contiguous array.
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
