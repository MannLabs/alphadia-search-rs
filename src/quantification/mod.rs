//! Library-anchored precursor quantification. Estimates the run quantities of every precursor
//! from its fragment intensities anchored to the library fragment pattern.
//!
//! Python does only the dataframe work: it turns the fragment tables into run-major matrices.
//! All calculations occur here.

mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::{
    fit, validate_shapes, FitParameters, DEFAULT_MAX_SWEEPS, DEFAULT_MIN_IONS, DEFAULT_TOLERANCE,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

/// Fragment matrices in, precursor quantities out.
///
/// The description below comes from `src/quantification/DOCS.md`. It gives the problem, the
/// input and output arrays, the model and the algorithm.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct LibraryAnchoredFit {
    params: FitParameters,
}

#[pymethods]
impl LibraryAnchoredFit {
    /// * `weighting` - base weight of a cell, `"variance"` (inverse residual variance at its
    ///   expected intensity) or `"share"` (library share of the fragment).
    /// * `tau` - scale of the Cauchy loss in log2, None to derive it from the data.
    /// * `lam` - shrinkage of the fragment offsets towards the library pattern, None to derive it.
    /// * `max_sweeps` - maximum number of reweighting sweeps per precursor.
    /// * `min_ions` - minimum observed fragments per precursor and run.
    /// * `tolerance` - largest change of a quantity (log2) at which the sweeps stop.
    ///
    /// Raises `ValueError` if `weighting` is unknown, `tau` is not positive, `lam` or `tolerance` is negative, or
    /// `min_ions` is zero.
    #[new]
    #[pyo3(signature = (
        weighting = "variance",
        tau = None,
        lam = None,
        max_sweeps = DEFAULT_MAX_SWEEPS,
        min_ions = DEFAULT_MIN_IONS,
        tolerance = DEFAULT_TOLERANCE,
    ))]
    fn new(
        weighting: &str,
        tau: Option<f64>,
        lam: Option<f64>,
        max_sweeps: usize,
        min_ions: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        let params = FitParameters {
            weighting: weighting.parse().map_err(PyValueError::new_err)?,
            tau,
            lambda: lam,
            max_sweeps,
            min_ions,
            tolerance,
        };
        params.validate().map_err(PyValueError::new_err)?;
        Ok(Self { params })
    }

    /// Gives a dict with the `(n_runs, n_precursors)` log2 `quantity` matrix and the scalars
    /// `residual_scale`, `loss_scale` and `offset_shrinkage` of the fit.
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the shapes disagree or a precursor index is out of range, and
    /// `TypeError` if an array is not contiguous.
    fn fit(
        &self,
        py: Python<'_>,
        intensity: PyReadonlyArray2<'_, f32>,
        library: PyReadonlyArray1<'_, f32>,
        precursor: PyReadonlyArray1<'_, usize>,
        n_precursors: usize,
    ) -> PyResult<Py<PyAny>> {
        let shape = intensity.shape();
        let n_runs = shape[0];
        let precursor = precursor.as_slice()?;
        validate_shapes((shape[0], shape[1]), library.len(), precursor, n_precursors)
            .map_err(PyValueError::new_err)?;

        let intensity = intensity.as_slice()?;
        let library = library.as_slice()?;
        let result = fit(
            intensity,
            library,
            precursor,
            n_runs,
            n_precursors,
            &self.params,
        );

        let quantity = Array2::from_shape_vec((n_runs, n_precursors), result.quantity)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py);
        let dict = PyDict::new(py);
        dict.set_item("quantity", quantity)?;
        dict.set_item("residual_scale", result.residual_scale)?;
        dict.set_item("loss_scale", result.loss_scale)?;
        dict.set_item("offset_shrinkage", result.offset_shrinkage)?;
        Ok(dict.into())
    }
}
