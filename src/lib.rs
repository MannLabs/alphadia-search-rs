use pyo3::prelude::*;
use pyo3::{Python, PyErr};
use pyo3::exceptions::PyValueError;

mod xic_slice;
mod quadrupole_observation;
mod rt_index;
mod mz_index;
mod dia_data_builder;
mod dia_data;
mod kernel;
mod benchmark;
mod convolution;
mod precursor;
mod speclib_flat;
pub mod peak_group_scoring;
pub mod candidate;
pub mod score;

use crate::dia_data::DIAData;
pub use crate::kernel::GaussianKernel;
use crate::speclib_flat::SpecLibFlat;
use crate::peak_group_scoring::PeakGroupScoring;
use crate::candidate::CandidateCollection;

#[cfg(test)]
mod convolution_test;

#[pyfunction]
fn benchmark_convolution() -> PyResult<(f64, f64)> {
    // Run the modular benchmark function from the benchmark module
    let results = benchmark::run_convolution_benchmark();
    
    // Return the original values from the first and second implementations for backward compatibility
    if results.len() >= 2 {
        Ok((results[0].time_seconds, results[1].time_seconds))
    } else {
        Err(PyErr::new::<PyValueError, _>("Benchmark failed to produce enough results"))
    }
}


#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DIAData>()?;
    m.add_class::<SpecLibFlat>()?;
    m.add_class::<PeakGroupScoring>()?;
    m.add_class::<CandidateCollection>()?;
    m.add_function(wrap_pyfunction!(benchmark_convolution, m)?)?;
    Ok(())
}