mod algorithm;
#[cfg(test)]
mod tests;

use algorithm::{fit, validate_shapes, FitConfig};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct VarModelQuantification;

#[pymethods]
impl VarModelQuantification {
    /// The fit runs with fixed settings; its loss scale and offset shrinkage are derived from
    /// the data.
    #[new]
    fn new() -> Self {
        Self
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
        intensity_library: PyReadonlyArray1<'_, f32>,
        precursor: PyReadonlyArray1<'_, usize>,
        n_precursors: usize,
    ) -> PyResult<Py<PyAny>> {
        let shape = intensity.shape();
        let n_runs = shape[0];
        let precursor = precursor.as_slice()?;
        validate_shapes(
            (shape[0], shape[1]),
            intensity_library.len(),
            precursor,
            n_precursors,
        )
        .map_err(PyValueError::new_err)?;

        let intensity = intensity.as_slice()?;
        let intensity_library = intensity_library.as_slice()?;
        let result = fit(
            intensity,
            intensity_library,
            precursor,
            n_runs,
            n_precursors,
            &FitConfig::DEFAULT,
        );

        let quantity = Array2::from_shape_vec((n_runs, n_precursors), result.quantity)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py);
        let dict = PyDict::new(py);
        dict.set_item("quantity", quantity)?;
        dict.set_item("residual_scale", result.parameters.sigma)?;
        dict.set_item("loss_scale", result.parameters.tau)?;
        dict.set_item("offset_shrinkage", result.parameters.lambda)?;
        Ok(dict.into())
    }
}
