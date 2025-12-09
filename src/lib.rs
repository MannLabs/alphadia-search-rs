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
    Ok(())
}
