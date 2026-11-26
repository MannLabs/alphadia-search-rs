//! FDR classifier: quantile feature transform.
//!
//! The Python side keeps all state as numpy arrays (the fitted quantiles), so the functions here
//! are stateless. Thus the classifiers pickle and deep-copy like any Python object. Computation
//! runs on the rayon thread pool (`set_num_threads`) with the GIL released.

pub(crate) mod quantile;
#[cfg(test)]
mod tests;

use crate::utils::require_c_contiguous;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const DEFAULT_N_QUANTILES: usize = 1001;

fn matrix<'a>(x: &'a PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<(&'a [f32], usize, usize)> {
    require_c_contiguous(&**x, name)?;
    let shape = x.shape();
    Ok((x.as_slice()?, shape[0], shape[1]))
}

/// Fits a quantile transform with a standard normal output, like sklearn's
/// `QuantileTransformer(output_distribution="normal")`.
///
/// Returns the quantiles as `[n_features, n_quantiles]` (float64). With `subsample`, the
/// quantiles come from that many randomly chosen rows.
#[pyfunction]
#[pyo3(signature = (x, n_quantiles = DEFAULT_N_QUANTILES, subsample = None, seed = 0))]
pub fn fdr_quantile_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    n_quantiles: usize,
    subsample: Option<usize>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (values, n_rows, n_features) = matrix(&x, "x")?;
    let quantiles = py
        .detach(|| quantile::fit(values, n_rows, n_features, n_quantiles, subsample, seed))
        .map_err(PyValueError::new_err)?;
    let quantiles = Array2::from_shape_vec((n_features, n_quantiles), quantiles)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(quantiles.into_pyarray(py))
}

/// Applies fitted quantiles (from `fdr_quantile_fit`) to `x` and returns float32 normal scores.
#[pyfunction]
pub fn fdr_quantile_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    quantiles: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (values, n_rows, n_features) = matrix(&x, "x")?;
    require_c_contiguous(&*quantiles, "quantiles")?;
    if quantiles.shape()[0] != n_features {
        return Err(PyValueError::new_err(format!(
            "quantiles are fitted for {} features, x has {n_features}",
            quantiles.shape()[0]
        )));
    }
    let n_quantiles = quantiles.shape()[1];
    let quantiles = quantiles.as_slice()?;
    let out = py
        .detach(|| quantile::transform(values, n_rows, n_features, quantiles, n_quantiles))
        .map_err(PyValueError::new_err)?;
    let out = Array2::from_shape_vec((n_rows, n_features), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}
