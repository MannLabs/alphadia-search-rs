//! Intensity drift correction. Removes the retention time dependent intensity drift of every
//! run from the fragment intensities before label-free quantification.
//!
//! Python does only the dataframe work: it turns the fragment tables and the identifications
//! into run-major matrices. All calculations occur here.

mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::{
    correct_drift, validate_shapes, DriftParameters, DEFAULT_BIN_SECONDS, DEFAULT_MIN_CORRELATION,
    DEFAULT_MIN_IONS_PER_BIN, DEFAULT_MIN_IONS_PER_RUN, DEFAULT_MIN_OBSERVATIONS,
};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

/// Matrices in, corrected in place, one summary row per run out.
///
/// The description below comes from `src/intensity_drift/DOCS.md`. It gives the problem, the
/// input and output arrays, the terms and the algorithm.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct IntensityDriftCorrection {
    params: DriftParameters,
}

#[pymethods]
impl IntensityDriftCorrection {
    /// * `bin_seconds` - width of an RT bin before merging, in seconds.
    /// * `min_correlation` - minimum mean cross-run correlation of a usable ion.
    /// * `min_observations` - minimum number of runs a usable ion is observed in.
    /// * `min_ions_per_bin` - minimum number of usable ions observed in the run per bin.
    /// * `min_ions_per_run` - runs with fewer usable ions are not corrected.
    ///
    /// Raises `ValueError` if `bin_seconds` is not positive, `min_correlation` is not a
    /// number, or `min_observations` or `min_ions_per_bin` is zero.
    #[new]
    #[pyo3(signature = (
        bin_seconds = DEFAULT_BIN_SECONDS,
        min_correlation = DEFAULT_MIN_CORRELATION,
        min_observations = DEFAULT_MIN_OBSERVATIONS,
        min_ions_per_bin = DEFAULT_MIN_IONS_PER_BIN,
        min_ions_per_run = DEFAULT_MIN_IONS_PER_RUN,
    ))]
    fn new(
        bin_seconds: f64,
        min_correlation: f64,
        min_observations: usize,
        min_ions_per_bin: usize,
        min_ions_per_run: usize,
    ) -> PyResult<Self> {
        let params = DriftParameters {
            bin_seconds,
            min_correlation,
            min_observations,
            min_ions_per_bin,
            min_ions_per_run,
            ..DriftParameters::DEFAULT
        };
        params.validate().map_err(PyValueError::new_err)?;
        Ok(Self { params })
    }

    /// Corrects `intensity` in place and gives a dict of arrays with one row per run:
    /// `corrected` and `max_abs_correction` (log2).
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the shapes disagree or a precursor index is out of range, and
    /// `TypeError` if an array is not contiguous.
    fn correct(
        &self,
        py: Python<'_>,
        mut intensity: PyReadwriteArray2<'_, f32>,
        correlation: PyReadonlyArray2<'_, f32>,
        rt: PyReadonlyArray2<'_, f32>,
        precursor: PyReadonlyArray1<'_, usize>,
    ) -> PyResult<Py<PyAny>> {
        let shape = |array: &[usize]| (array[0], array[1]);
        let n_runs = intensity.shape()[0];
        let precursor = precursor.as_slice()?;
        validate_shapes(
            shape(intensity.shape()),
            shape(correlation.shape()),
            shape(rt.shape()),
            precursor,
        )
        .map_err(PyValueError::new_err)?;

        let curves = correct_drift(
            intensity.as_slice_mut()?,
            correlation.as_slice()?,
            rt.as_slice()?,
            precursor,
            n_runs,
            &self.params,
        );

        let corrected: Vec<bool> = curves.iter().map(Option::is_some).collect();
        let max_abs_correction: Vec<f64> = curves
            .iter()
            .map(|curve| {
                curve.as_ref().map_or(0.0, |curve| {
                    curve.correction.iter().fold(0.0, |max, c| c.abs().max(max))
                })
            })
            .collect();
        let dict = PyDict::new(py);
        dict.set_item("corrected", corrected.into_pyarray(py))?;
        dict.set_item("max_abs_correction", max_abs_correction.into_pyarray(py))?;
        Ok(dict.into())
    }
}
