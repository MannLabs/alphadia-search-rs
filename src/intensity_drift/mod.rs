//! Intensity drift correction. Removes the retention time dependent intensity drift of every
//! run before label-free quantification, from the fragment intensities
//! (`IntensityDriftCorrection`) or from the precursor quantities along the retention time and
//! the precursor mass (`PrecursorDriftCorrection`).
//!
//! Python does only the dataframe work: it turns the tables and the identifications into
//! run-major matrices. All calculations occur here.

mod algorithm;
mod precursor;
#[cfg(test)]
mod precursor_tests;
#[cfg(test)]
mod tests;

use crate::utils::require_c_contiguous;
use algorithm::{
    correct_drift, validate_shapes, DriftParameters, DEFAULT_BIN_SECONDS, DEFAULT_MIN_CORRELATION,
    DEFAULT_MIN_IONS_PER_BIN, DEFAULT_MIN_IONS_PER_RUN,
};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods,
};
use precursor::{
    correct_precursor_drift, validate_precursor_shapes, PrecursorDriftParameters,
    DEFAULT_MIN_POINTS_PER_CELL, DEFAULT_MIN_REFERENCE_PER_RUN, DEFAULT_MIN_RT_BINS_PER_STRATUM,
    DEFAULT_N_STRATA, DEFAULT_POINTS_PER_CELL, DEFAULT_SIGMA_MASS, DEFAULT_SIGMA_RT,
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
    /// * `min_ions_per_bin` - minimum number of usable ions observed in the run per bin.
    /// * `min_ions_per_run` - runs with fewer usable ions are not corrected.
    ///
    /// Raises `ValueError` if `bin_seconds` is not positive, `min_correlation` is not a
    /// number, or `min_ions_per_bin` is zero.
    #[new]
    #[pyo3(signature = (
        bin_seconds = DEFAULT_BIN_SECONDS,
        min_correlation = DEFAULT_MIN_CORRELATION,
        min_ions_per_bin = DEFAULT_MIN_IONS_PER_BIN,
        min_ions_per_run = DEFAULT_MIN_IONS_PER_RUN,
    ))]
    fn new(
        bin_seconds: f64,
        min_correlation: f64,
        min_ions_per_bin: usize,
        min_ions_per_run: usize,
    ) -> PyResult<Self> {
        let params = DriftParameters {
            bin_seconds,
            min_correlation,
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
    /// `TypeError` if an array is not C-contiguous.
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
        require_c_contiguous(&**intensity, "intensity")?;
        require_c_contiguous(&*correlation, "correlation")?;
        require_c_contiguous(&*rt, "rt")?;

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

/// Precursor quantities in, corrected in place, one summary row per run out.
///
/// The description below comes from `src/intensity_drift/DOCS.md`. It gives the problem, the
/// input and output arrays, the terms and the algorithm of both corrections.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct PrecursorDriftCorrection {
    params: PrecursorDriftParameters,
}

#[pymethods]
impl PrecursorDriftCorrection {
    /// * `points_per_cell` - target number of reference precursors per cell.
    /// * `n_strata` - maximum number of mass strata.
    /// * `min_rt_bins_per_stratum` - fewer strata are used when the reference precursors cannot
    ///   fill this many RT bins per stratum.
    /// * `sigma_rt` - smoothing along the RT bins, in bins; 0 disables it.
    /// * `sigma_mass` - smoothing along the mass strata, in strata; 0 disables it.
    /// * `min_points_per_cell` - a cell with fewer reference precursors gets no location of its own.
    /// * `min_reference_per_run` - runs with fewer reference precursors are not corrected.
    ///
    /// Raises `ValueError` if a count is zero or a sigma is negative or not a number.
    #[new]
    #[pyo3(signature = (
        points_per_cell = DEFAULT_POINTS_PER_CELL,
        n_strata = DEFAULT_N_STRATA,
        min_rt_bins_per_stratum = DEFAULT_MIN_RT_BINS_PER_STRATUM,
        sigma_rt = DEFAULT_SIGMA_RT,
        sigma_mass = DEFAULT_SIGMA_MASS,
        min_points_per_cell = DEFAULT_MIN_POINTS_PER_CELL,
        min_reference_per_run = DEFAULT_MIN_REFERENCE_PER_RUN,
    ))]
    fn new(
        points_per_cell: usize,
        n_strata: usize,
        min_rt_bins_per_stratum: usize,
        sigma_rt: f64,
        sigma_mass: f64,
        min_points_per_cell: usize,
        min_reference_per_run: usize,
    ) -> PyResult<Self> {
        let params = PrecursorDriftParameters {
            points_per_cell,
            n_strata,
            min_rt_bins_per_stratum,
            sigma_rt,
            sigma_mass,
            min_points_per_cell,
            min_reference_per_run,
        };
        params.validate().map_err(PyValueError::new_err)?;
        Ok(Self { params })
    }

    /// Corrects `intensity` in place and gives a dict with the arrays `corrected` and
    /// `max_abs_correction` (log2), one row per run, and `n_strata`, the number of mass strata
    /// used.
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the shapes disagree and `TypeError` if an array is not
    /// C-contiguous.
    fn correct(
        &self,
        py: Python<'_>,
        mut intensity: PyReadwriteArray2<'_, f64>,
        rt: PyReadonlyArray2<'_, f64>,
        mass: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        let shape = |array: &[usize]| (array[0], array[1]);
        let n_runs = intensity.shape()[0];
        let mass = mass.as_slice()?;
        validate_precursor_shapes(shape(intensity.shape()), shape(rt.shape()), mass.len())
            .map_err(PyValueError::new_err)?;
        require_c_contiguous(&**intensity, "intensity")?;
        require_c_contiguous(&*rt, "rt")?;

        let summary = correct_precursor_drift(
            intensity.as_slice_mut()?,
            rt.as_slice()?,
            mass,
            n_runs,
            &self.params,
        );

        let dict = PyDict::new(py);
        dict.set_item("corrected", summary.corrected.into_pyarray(py))?;
        dict.set_item(
            "max_abs_correction",
            summary.max_abs_correction.into_pyarray(py),
        )?;
        dict.set_item("n_strata", summary.n_strata)?;
        Ok(dict.into())
    }
}
