//! Calibration: LOESS-based per-run calibration of m/z, RT and mobility.
//!
//! The numeric algorithm (LOESS fit/predict, deviation transform, metrics) lives
//! here in Rust. Python keeps only a thin array-passing manager and the plotting
//! code. The set of estimators and their configuration is hardcoded (see
//! [`EstimatorKind`]); Python selects one by its string identifier.

pub mod estimator;
pub mod loess;

#[cfg(test)]
mod tests;

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::calibration::estimator::CalibrationEstimator as CalibrationEstimatorInner;

/// Python-facing calibration estimator (thin arrays-in / arrays-out interface).
#[pyclass]
pub struct CalibrationEstimator {
    inner: CalibrationEstimatorInner,
}

#[pymethods]
impl CalibrationEstimator {
    /// Create an estimator.
    ///
    /// * `n_kernels` — number of LOESS kernels.
    /// * `transform_deviation` — deviation transform factor (e.g. `1e6` for ppm),
    ///   or `None` for absolute deviations.
    #[new]
    #[pyo3(signature = (n_kernels, transform_deviation=None))]
    fn new(n_kernels: usize, transform_deviation: Option<f32>) -> Self {
        Self {
            inner: CalibrationEstimatorInner::new(n_kernels, transform_deviation),
        }
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Deviation transform factor (`1e6` for ppm) or `None` for absolute.
    #[getter]
    fn transform_deviation(&self) -> Option<f32> {
        self.inner.transform_deviation()
    }

    /// Fit on `input` (uncalibrated property) and `target` (observed property).
    /// Raises `ValueError` on degenerate input (Python leaves it unfitted).
    fn fit(&mut self, input: Vec<f32>, target: Vec<f32>) -> PyResult<()> {
        self.inner
            .fit(&input, &target)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Predict calibrated values from `input`.
    fn predict<'py>(&self, py: Python<'py>, input: Vec<f32>) -> Bound<'py, PyArray1<f32>> {
        self.inner.predict(&input).into_pyarray(py)
    }

    /// Return `(observed, calibrated, residual)` deviation arrays for plotting.
    fn deviation<'py>(
        &self,
        py: Python<'py>,
        input: Vec<f32>,
        target: Vec<f32>,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    ) {
        let dev = self.inner.deviation(&input, &target);
        (
            dev.observed.into_pyarray(py),
            dev.calibrated.into_pyarray(py),
            dev.residual.into_pyarray(py),
        )
    }

    /// `(median_bias, median_variance)` if fitted, else `None`.
    fn metrics(&self) -> Option<(f32, f32)> {
        self.inner.metrics()
    }

    /// Residual deviation at the given confidence interval (e.g. `0.95`).
    fn ci(&self, input: Vec<f32>, target: Vec<f32>, ci: f32) -> f32 {
        self.inner.ci(&input, &target, ci)
    }
}
