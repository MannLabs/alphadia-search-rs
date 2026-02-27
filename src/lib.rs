use numpy::ndarray::{s, Array1, Axis};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyErr;

mod benchmark;
pub mod candidate;
pub mod constants;
mod convolution;
mod dense_xic_observation;
pub mod dia_data;
pub mod dia_data_builder;
pub mod idf;
mod kernel;
mod mz_index;
pub mod peak_group_quantification;
pub mod peak_group_scoring;
pub mod peak_group_selection;
mod precursor;
mod precursor_quantified;
mod quadrupole_observation;
mod rt_index;
pub mod score;
mod simd;
pub mod speclib_flat;
pub mod speclib_flat_quantified;
mod threadpool;
pub mod traits;
pub mod utils;

use crate::candidate::{CandidateCollection, CandidateFeatureCollection};
use crate::dia_data::DIAData;
pub use crate::kernel::GaussianKernel;
use crate::peak_group_quantification::{PeakGroupQuantification, QuantificationParameters};
use crate::peak_group_scoring::{PeakGroupScoring, ScoringParameters};
use crate::peak_group_selection::{PeakGroupSelection, SelectionParameters};
use crate::speclib_flat::SpecLibFlat;
use crate::speclib_flat_quantified::SpecLibFlatQuantified;

#[pyfunction]
fn benchmark_convolution() -> PyResult<(f64, f64)> {
    // Run the modular benchmark function from the benchmark module
    let results = benchmark::run_convolution_benchmark();

    // Return the original values from the first and second implementations for backward compatibility
    if results.len() >= 2 {
        Ok((results[0].time_seconds, results[1].time_seconds))
    } else {
        Err(PyErr::new::<PyValueError, _>(
            "Benchmark failed to produce enough results",
        ))
    }
}

#[pyfunction]
fn get_optimal_simd_backend() -> PyResult<String> {
    Ok(simd::get_optimal_simd_backend())
}

#[pyfunction]
fn set_simd_backend(backend_name: String) -> PyResult<()> {
    simd::set_backend(&backend_name).map_err(PyErr::new::<PyValueError, _>)
}

#[pyfunction]
fn clear_simd_backend() -> PyResult<()> {
    simd::clear_backend();
    Ok(())
}

#[pyfunction]
fn get_current_simd_backend() -> PyResult<String> {
    Ok(simd::get_current_backend())
}

#[pyfunction]
fn set_num_threads(num_threads: Option<usize>) -> PyResult<()> {
    // Rayon's global thread pool initializes on first use and cannot be changed afterward.
    // If anything triggers a parallel operation before calling set_num_threads(), it will fail
    // Decision: use the global thread pool for now (simplicity!).
    threadpool::set_num_threads(num_threads).map_err(PyErr::new::<PyValueError, _>)
}

#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    Ok(threadpool::get_num_threads())
}

/// Compute z-score filter mask over a feature matrix using batched ndarray linalg.
///
/// The z-score sum `Σ (x_j - μ_j) / σ_j * s_j` is equivalent to `x · w - b`
/// where `w = signs / stds` and `b = Σ(means * signs / stds)`.
/// Processes in batches to limit memory for column extraction.
/// NaN and infinite values are treated as 0.
///
/// Returns a boolean numpy array where True = passes filter (score >= threshold).
#[pyfunction]
fn zscore_filter_mask(
    py: Python,
    features: PyReadonlyArray2<'_, f64>,
    col_indices: Vec<usize>,
    means: Vec<f64>,
    stds: Vec<f64>,
    signs: Vec<f64>,
    threshold: f64,
) -> PyResult<PyObject> {
    let n_zscore = col_indices.len();
    if means.len() != n_zscore || stds.len() != n_zscore || signs.len() != n_zscore {
        return Err(PyErr::new::<PyValueError, _>(
            "means, stds, signs must have the same length as col_indices",
        ));
    }

    let features = features.as_array();
    let n_rows = features.shape()[0];

    // Precompute weight vector w = signs / stds and bias b = Σ(means * w)
    let w = Array1::from_vec(
        signs
            .iter()
            .zip(&stds)
            .map(|(&s, &st)| s / st)
            .collect::<Vec<f64>>(),
    );
    let bias: f64 = means.iter().zip(w.iter()).map(|(m, wi)| m * wi).sum();
    let adjusted_threshold = threshold + bias;

    let mut mask = Array1::<bool>::from_elem(n_rows, false);

    const BATCH_SIZE: usize = 500_000;
    for batch_start in (0..n_rows).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(n_rows);
        let batch = features.slice(s![batch_start..batch_end, ..]);

        // Select z-score columns → contiguous (batch_len, n_zscore) array
        let mut z_batch = batch.select(Axis(1), &col_indices);

        // Replace NaN/inf with 0 in-place
        z_batch.mapv_inplace(|v| {
            if v.is_nan() || v.is_infinite() {
                0.0
            } else {
                v
            }
        });

        // Batched dot product: scores = z_batch · w (shape: batch_len)
        let scores = z_batch.dot(&w);

        // Threshold comparison
        for (i, &s) in scores.iter().enumerate() {
            mask[batch_start + i] = s >= adjusted_threshold;
        }
    }

    Ok(mask.into_pyarray(py).into())
}

#[pymodule]
fn alphadia_search_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DIAData>()?;
    m.add_class::<SpecLibFlat>()?;
    m.add_class::<SpecLibFlatQuantified>()?;
    m.add_class::<PeakGroupScoring>()?;
    m.add_class::<ScoringParameters>()?;
    m.add_class::<PeakGroupSelection>()?;
    m.add_class::<SelectionParameters>()?;
    m.add_class::<PeakGroupQuantification>()?;
    m.add_class::<QuantificationParameters>()?;
    m.add_class::<CandidateCollection>()?;
    m.add_class::<CandidateFeatureCollection>()?;
    m.add_function(wrap_pyfunction!(benchmark_convolution, m)?)?;
    m.add_function(wrap_pyfunction!(get_optimal_simd_backend, m)?)?;
    m.add_function(wrap_pyfunction!(set_simd_backend, m)?)?;
    m.add_function(wrap_pyfunction!(clear_simd_backend, m)?)?;
    m.add_function(wrap_pyfunction!(get_current_simd_backend, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_filter_mask, m)?)?;
    Ok(())
}
