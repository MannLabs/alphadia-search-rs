mod algorithm;
mod dropout;
mod noise;
mod numeric;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_censored;

use crate::utils::require_c_contiguous;
use algorithm::{fit, validate_correlation, validate_shapes, FitConfig};
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
    /// `residual_scale`, `loss_scale` and `offset_shrinkage` of the fit; with `censored` also
    /// the per-run dropout curves `detection_limit`, `dropout_width` and `dropout_plateau`
    /// (NaN for a run without missing fragments).
    ///
    /// `correlation` has the shape of `intensity` and values in [0, 1]; it only matters when
    /// `quality_exponent` > 0.
    ///
    /// This method borrows the arrays and does not copy them. The arrays must therefore be
    /// C-contiguous.
    ///
    /// Raises `ValueError` if the shapes disagree, a precursor index is out of range or an
    /// option is invalid, and `TypeError` if an array is not C-contiguous.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        intensity,
        intensity_library,
        precursor,
        n_precursors,
        correlation,
        *,
        censored=false,
        quality_exponent=0.0,
    ))]
    fn fit(
        &self,
        py: Python<'_>,
        intensity: PyReadonlyArray2<'_, f32>,
        intensity_library: PyReadonlyArray1<'_, f32>,
        precursor: PyReadonlyArray1<'_, usize>,
        n_precursors: usize,
        correlation: PyReadonlyArray2<'_, f32>,
        censored: bool,
        quality_exponent: f64,
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
        if correlation.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "correlation has shape {:?}, intensity {:?}",
                correlation.shape(),
                shape
            )));
        }
        if quality_exponent < 0.0 {
            return Err(PyValueError::new_err(format!(
                "quality_exponent must be >= 0, not {quality_exponent}"
            )));
        }
        require_c_contiguous(&*intensity, "intensity")?;
        require_c_contiguous(&*correlation, "correlation")?;
        let intensity = intensity.as_slice()?;
        let intensity_library = intensity_library.as_slice()?;
        let correlation = correlation.as_slice()?;
        validate_correlation(correlation).map_err(PyValueError::new_err)?;
        let result = fit(
            intensity,
            intensity_library,
            precursor,
            correlation,
            n_runs,
            n_precursors,
            &FitConfig {
                censored,
                quality_exponent,
                ..FitConfig::DEFAULT
            },
        );

        let quantity = Array2::from_shape_vec((n_runs, n_precursors), result.quantity)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py);
        let dict = PyDict::new(py);
        dict.set_item("quantity", quantity)?;
        dict.set_item("residual_scale", result.parameters.sigma)?;
        dict.set_item("loss_scale", result.parameters.tau)?;
        dict.set_item("offset_shrinkage", result.parameters.lambda)?;
        if let Some(dropout) = result.dropout {
            let per_run = |value: fn(&dropout::Curve) -> f64| -> Vec<f64> {
                dropout
                    .curves
                    .iter()
                    .map(|curve| curve.as_ref().map_or(f64::NAN, value))
                    .collect()
            };
            dict.set_item("detection_limit", per_run(|c| c.limit).into_pyarray(py))?;
            dict.set_item("dropout_width", per_run(|c| c.width).into_pyarray(py))?;
            dict.set_item("dropout_plateau", per_run(|c| c.plateau).into_pyarray(py))?;
        }
        Ok(dict.into())
    }
}
