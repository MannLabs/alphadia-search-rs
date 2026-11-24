//! FDR classifier: quantile feature transform and a small fully connected network.
//!
//! The Python side keeps all state as numpy arrays (quantiles, flat network parameters), so the
//! functions here are stateless: classifiers pickle and deep-copy like any Python object, and a
//! warm start passes the previous parameters back in. Computation runs on the rayon thread pool
//! (`set_num_threads`) with the GIL released.

mod network;
mod ops;
mod quantile;
#[cfg(test)]
mod tests;

use crate::utils::require_c_contiguous;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use ops::Backend;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const DEFAULT_N_QUANTILES: usize = 1001;
const DEFAULT_EPOCHS: usize = 10;
const DEFAULT_BATCH_SIZE: usize = 4096;
const DEFAULT_LEARNING_RATE: f32 = 0.001;
const DEFAULT_WEIGHT_DECAY: f32 = 1e-5;

fn backend(name: Option<&str>) -> PyResult<Backend> {
    name.map_or(Ok(Backend::default_for_target()), |n| {
        Backend::from_name(n).map_err(PyValueError::new_err)
    })
}

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

/// Applies fitted quantiles (from `fdr_quantile_fit`) to `x`; returns float32 normal scores.
///
/// `mid_rank_bounds` maps ties at the lowest / highest quantile to the middle of their tie group
/// instead of clamping them to the clip value as sklearn does (see `quantile::transform`).
#[pyfunction]
#[pyo3(signature = (x, quantiles, mid_rank_bounds = false))]
pub fn fdr_quantile_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    quantiles: PyReadonlyArray2<'py, f64>,
    mid_rank_bounds: bool,
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
        .detach(|| {
            quantile::transform(
                values,
                n_rows,
                n_features,
                quantiles,
                n_quantiles,
                mid_rank_bounds,
            )
        })
        .map_err(PyValueError::new_err)?;
    let out = Array2::from_shape_vec((n_rows, n_features), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

/// Trains the FDR network on `x` (float32 `[rows, features]`) and binary labels `y` (float32).
///
/// Returns the flat float32 parameter vector. `params` continues training from earlier
/// parameters (warm start). `backend` overrides the matmul backend ("accelerate" on macOS,
/// "simd"); the default is the fastest one for the platform.
#[pyfunction]
#[pyo3(signature = (
    x,
    y,
    hidden_layers,
    epochs = DEFAULT_EPOCHS,
    batch_size = DEFAULT_BATCH_SIZE,
    learning_rate = DEFAULT_LEARNING_RATE,
    weight_decay = DEFAULT_WEIGHT_DECAY,
    seed = 0,
    params = None,
    backend = None,
    cosine_schedule = false,
    average_last_epoch = false,
    input_batch_norm = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn fdr_mlp_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
    hidden_layers: Vec<usize>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
    weight_decay: f32,
    seed: u64,
    params: Option<PyReadonlyArray1<'py, f32>>,
    backend: Option<&str>,
    cosine_schedule: bool,
    average_last_epoch: bool,
    input_batch_norm: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (values, n_rows, n_features) = matrix(&x, "x")?;
    let labels = y.as_slice()?;
    let init = params
        .map(|p| p.as_slice().map(<[f32]>::to_vec))
        .transpose()?;
    let config = network::TrainConfig {
        hidden_layers,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        seed,
        backend: self::backend(backend)?,
        cosine_schedule,
        average_last_epoch,
        input_batch_norm,
    };
    let trained = py
        .detach(|| network::train(values, n_rows, n_features, labels, &config, init))
        .map_err(PyValueError::new_err)?;
    Ok(trained.into_pyarray(py))
}

/// Output logits (float32) of the network with `params` for every row of `x`.
#[pyfunction]
#[pyo3(signature = (x, hidden_layers, params, backend = None, input_batch_norm = false))]
pub fn fdr_mlp_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    hidden_layers: Vec<usize>,
    params: PyReadonlyArray1<'py, f32>,
    backend: Option<&str>,
    input_batch_norm: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (values, n_rows, n_features) = matrix(&x, "x")?;
    let params = params.as_slice()?;
    let backend = self::backend(backend)?;
    let logits = py
        .detach(|| {
            network::predict(
                values,
                n_rows,
                n_features,
                &hidden_layers,
                params,
                backend,
                input_batch_norm,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(logits.into_pyarray(py))
}

/// Names of the matmul backends of the FDR network on this platform, the default first.
#[pyfunction]
pub fn fdr_mlp_backends() -> Vec<&'static str> {
    Backend::available_names()
}
