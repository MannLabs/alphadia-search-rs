//! Intensity drift correction: removes the retention time dependent part of every run's
//! intensity deviation before label-free quantification.
//!
//! The estimator lives here in Rust (see [`algorithm`]). Python keeps only the dataframe
//! bookkeeping: it places every ion in retention time per run, hands the matrices over
//! run-major and writes the corrected intensities back.

mod algorithm;
#[cfg(test)]
mod tests;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use crate::intensity_drift::algorithm::DriftCurve;
use crate::intensity_drift::algorithm::{DriftParameters, Statistic};

/// The corrected intensities and the curve of every run, `None` where a run was unchanged.
type CorrectionResult<'py> = (Bound<'py, PyArray2<f64>>, Vec<Option<DriftCurve>>);

/// Python-facing intensity drift correction (thin arrays-in / arrays-out interface).
#[pyclass]
pub struct IntensityDriftCorrection {
    parameters: DriftParameters,
}

#[pymethods]
impl IntensityDriftCorrection {
    /// * `bin_seconds` — width of the fixed retention time bins the curve is estimated in.
    /// * `min_correlation` — minimum mean cross-run correlation for an ion to shape a curve.
    /// * `min_observations` — minimum number of runs an ion must be observed in to shape a
    ///   curve.
    /// * `min_ions_per_bin` — consecutive sparse bins are merged until they hold this many
    ///   ions.
    /// * `min_ions_per_run` — runs with fewer usable ions are left unchanged.
    /// * `statistic` — per-bin location estimator, `"mode"` or `"median"`.
    ///
    /// Raises `ValueError` for an unknown `statistic`.
    #[new]
    fn new(
        bin_seconds: f64,
        min_correlation: f64,
        min_observations: usize,
        min_ions_per_bin: usize,
        min_ions_per_run: usize,
        statistic: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            parameters: DriftParameters {
                bin_seconds,
                min_correlation,
                // the across-run level of an ion needs at least one observation
                min_observations: min_observations.max(1),
                min_ions_per_bin,
                min_ions_per_run,
                statistic: Statistic::from_name(statistic).map_err(PyValueError::new_err)?,
            },
        })
    }

    /// Divides the drift of every run out of `intensity`.
    ///
    /// The three matrices have the run-major shape `(n_runs, n_ions)`: `intensity` holds the
    /// linear fragment intensities where zero marks a missing value, `correlation` the
    /// cross-run fragment correlations and `rt` the observed retention time of every ion's
    /// precursor, NaN where a run has no identification for it.
    ///
    /// Gives the corrected intensities in the same layout, and the curve of every run with
    /// `None` for the runs that were left unchanged.
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the shapes do not agree. Raises `TypeError` if an array is not
    /// contiguous.
    fn correct<'py>(
        &self,
        py: Python<'py>,
        intensity: PyReadonlyArray2<'_, f64>,
        correlation: PyReadonlyArray2<'_, f64>,
        rt: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<CorrectionResult<'py>> {
        let shape = intensity.shape();
        if correlation.shape() != shape || rt.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "intensity, correlation and rt must have the same shape, got {:?}, {:?} and {:?}",
                shape,
                correlation.shape(),
                rt.shape()
            )));
        }
        let (n_runs, n_ions) = (shape[0], shape[1]);
        // the run-major matrices are chunked by ion, an empty table has nothing to correct
        if n_ions == 0 {
            return Ok((
                Array2::zeros((n_runs, 0)).into_pyarray(py),
                (0..n_runs).map(|_| None).collect(),
            ));
        }

        let (corrected, curves) = algorithm::correct_drift(
            intensity.as_slice()?,
            correlation.as_slice()?,
            rt.as_slice()?,
            n_ions,
            &self.parameters,
        );
        let corrected = Array2::from_shape_vec((n_runs, n_ions), corrected)
            .expect("the corrected matrix keeps the shape of the intensity matrix");
        Ok((corrected.into_pyarray(py), curves))
    }
}
