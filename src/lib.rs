use std::cmp::min;
use std::iter::zip;
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::{Python, PyErr};
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use numpy::ndarray::s;
use numpy::ndarray::{ArrayBase, ViewRepr, Dim, Array1, Array2};
use numpy::PyArray1;
use std::time::Instant;


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


use mz_index::{ppm_index, RESOLUTION_PPM, MZ_START, MZ_END, MZIndex};
use quadrupole_observation::QuadrupoleObservation;
use rt_index::RTIndex;
use dia_data_builder::DIADataBuilder;
use dia_data::{DIAData, PeakGroupScoring};
pub use kernel::GaussianKernel;
use precursor::Precursor;
use speclib_flat::SpecLibFlat;


use ndarray_npy::NpzWriter;
use std::fs::File;
use rand::prelude::*;

pub struct AlphaRawView<'py> {
    pub spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
}

impl<'py> AlphaRawView<'py> {
    fn new(
        spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            spectrum_rt,
            peak_mz,
            peak_intensity,
        }
    }
}

#[pyfunction]
fn test_xic_index<'py>(
    spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
    isolation_lower_mz: PyReadonlyArray1<'py, f32>,
    isolation_upper_mz: PyReadonlyArray1<'py, f32>,
    spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
    spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
    spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
    spectrum_rt: PyReadonlyArray1<'py, f32>,
    peak_mz: PyReadonlyArray1<'py, f32>,
    peak_intensity: PyReadonlyArray1<'py, f32>,
    py: Python<'py>
) -> PyResult<DIAData> {

    let alpha_raw_view = AlphaRawView::new(
        spectrum_delta_scan_idx.as_array(),
        isolation_lower_mz.as_array(),
        isolation_upper_mz.as_array(),
        spectrum_peak_start_idx.as_array(),
        spectrum_peak_stop_idx.as_array(),
        spectrum_cycle_idx.as_array(),
        spectrum_rt.as_array(),
        peak_mz.as_array(),
        peak_intensity.as_array(),
    );

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
    Ok(dia_data)
}

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
    m.add_function(wrap_pyfunction!(test_xic_index, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_convolution, m)?)?;
    Ok(())
}