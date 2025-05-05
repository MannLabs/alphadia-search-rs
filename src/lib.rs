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
mod precursor;
mod speclib_flat;
mod peak_group_scoring;

use crate::dia_data::DIAData;
pub use crate::kernel::GaussianKernel;
use crate::speclib_flat::SpecLibFlat;
use crate::peak_group_scoring::PeakGroupScoring;

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

/// A Python module implemented in Rust.
#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DIAData>()?;
    m.add_class::<SpecLibFlat>()?;
    m.add_class::<PeakGroupScoring>()?;
    m.add_function(wrap_pyfunction!(benchmark_convolution, m)?)?;
    Ok(())
}