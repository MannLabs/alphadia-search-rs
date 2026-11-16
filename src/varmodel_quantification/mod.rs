mod algorithm;
mod censored;
mod noise;
mod numeric;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_censored;

use crate::utils::require_c_contiguous;
use algorithm::{fit, validate_shapes, FitConfig};
use censored::{
    fit_censored, CensoredConfig, CensoredResult, MissingFragments, DEFAULT_CURVE_MAX_CELLS,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

const MISSING_DROP: &str = "drop";
const MISSING_CENSORED: &str = "censored";

/// The plain fit unless missing fragments are censored or the measurement quality weights the
/// variances.
fn quantify(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    correlation: Option<&[f32]>,
    n_runs: usize,
    n_precursors: usize,
    config: &CensoredConfig,
) -> CensoredResult {
    if config.missing_fragments == MissingFragments::Drop && config.quality_exponent <= 0.0 {
        let result = fit(
            intensity,
            intensity_library,
            precursor,
            n_runs,
            n_precursors,
            &FitConfig::DEFAULT,
        );
        return CensoredResult {
            quantity: result.quantity,
            parameters: result.parameters,
            dropout: None,
        };
    }
    fit_censored(
        intensity,
        intensity_library,
        precursor,
        correlation,
        n_runs,
        n_precursors,
        config,
    )
}

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
    /// `residual_scale`, `loss_scale` and `offset_shrinkage` of the fit; with censored missing
    /// fragments also the per-run dropout curves `detection_limit`, `dropout_width` and
    /// `dropout_plateau`.
    ///
    /// `missing_fragments` is "drop" or "censored". `correlation` has the shape of
    /// `intensity` and is needed when `quality_exponent` > 0. `curve_max_cells` bounds the
    /// measurements each dropout curve is fitted on.
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
        *,
        correlation=None,
        missing_fragments=MISSING_DROP,
        quality_exponent=0.0,
        curve_max_cells=DEFAULT_CURVE_MAX_CELLS,
    ))]
    fn fit(
        &self,
        py: Python<'_>,
        intensity: PyReadonlyArray2<'_, f32>,
        intensity_library: PyReadonlyArray1<'_, f32>,
        precursor: PyReadonlyArray1<'_, usize>,
        n_precursors: usize,
        correlation: Option<PyReadonlyArray2<'_, f32>>,
        missing_fragments: &str,
        quality_exponent: f64,
        curve_max_cells: usize,
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
        let missing_fragments = match missing_fragments {
            MISSING_DROP => MissingFragments::Drop,
            MISSING_CENSORED => MissingFragments::Censored,
            other => {
                return Err(PyValueError::new_err(format!(
                    "missing_fragments must be \"{MISSING_DROP}\" or \"{MISSING_CENSORED}\", not \"{other}\""
                )))
            }
        };
        if quality_exponent > 0.0 && correlation.is_none() {
            return Err(PyValueError::new_err(
                "quality_exponent > 0 needs the correlation of every fragment measurement",
            ));
        }
        if let Some(correlation) = &correlation {
            if correlation.shape() != shape {
                return Err(PyValueError::new_err(format!(
                    "correlation has shape {:?}, intensity {:?}",
                    correlation.shape(),
                    shape
                )));
            }
            require_c_contiguous(&**correlation, "correlation")?;
        }

        require_c_contiguous(&*intensity, "intensity")?;
        let intensity = intensity.as_slice()?;
        let intensity_library = intensity_library.as_slice()?;
        let correlation = correlation
            .as_ref()
            .map(|correlation| correlation.as_slice())
            .transpose()?;
        let result = quantify(
            intensity,
            intensity_library,
            precursor,
            correlation,
            n_runs,
            n_precursors,
            &CensoredConfig {
                missing_fragments,
                quality_exponent,
                curve_max_cells,
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
            dict.set_item("detection_limit", dropout.limit.into_pyarray(py))?;
            dict.set_item("dropout_width", dropout.width.into_pyarray(py))?;
            dict.set_item("dropout_plateau", dropout.plateau.into_pyarray(py))?;
        }
        Ok(dict.into())
    }
}
