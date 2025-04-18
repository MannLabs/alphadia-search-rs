use std::iter::zip;

use pyo3::prelude::*;
use pyo3::{Python, PyErr};
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use numpy::PyArray1;
use numpy::ndarray::Array1;
use numpy::ndarray::s;
use numpy::PyUntypedArray;
use numpy::PyArrayMethods;
use std::time::Instant;

mod xic_index;
use xic_index::XICIndex;
use xic_index::XICSlice;
use xic_index::ppm_index;
use xic_index::RESOLUTION_PPM;
use xic_index::MZ_START;
use xic_index::MZ_END;
use xic_index::MZIndex;

// Core functions that don't depend on Python bindings
fn add_numbers(a: usize, b: usize) -> String {
    (a + b).to_string()
}

fn sum_numpy_array(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok(add_numbers(a, b))
}

#[pyfunction]
fn sum_array(arr: PyReadonlyArray1<f64>) -> PyResult<String> {
    let data = arr.as_array();
    let sum = sum_numpy_array(data.as_slice().unwrap());
    Ok(sum.to_string())
}

#[pyclass]
struct Raw {
    data: Array1<f64>,
}

#[pymethods]
impl Raw {
    #[new]
    fn new(arr: &Bound<'_, PyUntypedArray>) -> PyResult<Self> {
        let array = arr.downcast::<PyArray1<f64>>()?.to_owned();
        let data = unsafe { array.as_array().to_owned() };
        Ok(Self { data })
    }

    fn sum(&self) -> f64 {
        if let Some(slice) = self.data.as_slice() {
            sum_numpy_array(slice)
        } else {
            let mut sum:f64 = 0.0;
            for i in 0..self.data.len(){
                sum += self.data[i];
            }
            sum
        }
    }
}

#[pyclass]
struct SumContainer {
    data: Array1<f64>,
}

#[pymethods]
impl SumContainer {
    #[new]
    fn new(arr: &Bound<'_, PyUntypedArray>) -> PyResult<Self> {
        let array = arr.downcast::<PyArray1<f64>>()?.to_owned();
        let data = unsafe { array.as_array().to_owned() };
        Ok(Self { data })
    }

    fn sum_array(&self) -> f64 {
        if let Some(slice) = self.data.as_slice() {
            sum_numpy_array(slice)
        } else {
            let mut sum:f64 = 0.0;
            for i in 0..self.data.len(){
                sum += self.data[i];
            }
            sum
        }
    }
}

#[pyfunction]
fn test_xic_index<'py>(
    peak_start_idx: PyReadonlyArray1<'py, i64>,
    peak_stop_idx: PyReadonlyArray1<'py, i64>,
    cycle_idx: PyReadonlyArray1<'py, i64>,
    mz: PyReadonlyArray1<'py, f32>,
    intensity: PyReadonlyArray1<'py, f32>,
    py: Python<'py>
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // Convert numpy arrays to slices for direct access
    let peak_start = peak_start_idx.as_array();
    let peak_stop = peak_stop_idx.as_array();
    let cycle = cycle_idx.as_array();
    let mz_array = mz.as_array();
    let intensity_array = intensity.as_array();

    // check all arrays have the same length
    if peak_start.len() != peak_stop.len() || peak_start.len() != cycle.len() {
        return Err(PyValueError::new_err("All arrays must have the same length"));
    }

    let mz_index = ppm_index(RESOLUTION_PPM, MZ_START, MZ_END);

    // Pre-allocate XIC slices with estimated capacity
    let estimated_peaks_per_slice = peak_start.len() / mz_index.len();
    let mut xic_slices: Vec<XICSlice> = Vec::with_capacity(mz_index.len());
    for _ in 0..mz_index.len() {
        let mut slice = XICSlice::empty();
        slice.cycle_index.reserve(estimated_peaks_per_slice);
        slice.intensity.reserve(estimated_peaks_per_slice);
        xic_slices.push(slice);
    }

    let mut dia_data = XICIndex::new(
        mz_index,
        xic_slices
    );

    let start_time = Instant::now();

    for i in 0..peak_start.len() {
        let peak_start_idx = peak_start[i] as usize;
        let peak_stop_idx = peak_stop[i] as usize;
        let cycle_val = cycle[i] as usize;

        // Use array views instead of converting to vectors
        let mz_slice = mz_array.slice(s![peak_start_idx..peak_stop_idx]);
        let intensity_slice = intensity_array.slice(s![peak_start_idx..peak_stop_idx]);
    
        for (mz_val, intensity_val) in zip(mz_slice.iter(), intensity_slice.iter()) {
            let target_xic_idx = dia_data.closest_index(*mz_val).unwrap();
            dia_data.xic_slices[target_xic_idx].cycle_index.push(cycle_val as u16);
            dia_data.xic_slices[target_xic_idx].intensity.push(*intensity_val);
        }
    }

    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Time taken: {:?}", duration);

    // calculate total number of peaks
    let mut total_peaks = 0;
    for xic_slice in dia_data.xic_slices.iter() {
        total_peaks += xic_slice.cycle_index.len();
    }

    println!("Total number of peaks: {}", total_peaks);

    let mut hist = Array1::zeros(dia_data.xic_slices.len());
    for i in 0..dia_data.xic_slices.len() {
        hist[i] = dia_data.xic_slices[i].cycle_index.len() as f32;
    }

    let hist_array = PyArray1::from_vec(py, hist.to_vec());
    
    Ok(hist_array)
}

/// A Python module implemented in Rust.
#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_class::<Raw>()?;
    m.add_class::<SumContainer>()?;
    m.add_function(wrap_pyfunction!(test_xic_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    // Helper function that doesn't rely on Python bindings
    fn add_numbers(a: usize, b: usize) -> String {
        (a + b).to_string()
    }
    
    #[test]
    fn test_add_numbers() {
        let result = add_numbers(5, 7);
        assert_eq!(result, "12");
    }
    
    #[test]
    fn test_raw_sum_impl() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let raw = Raw { data };
        let result = raw.sum();
        assert_eq!(result, 15.0);
    }
    
    #[test]
    fn test_sum_container_impl() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let container = SumContainer { data };
        let result = container.sum_array();
        assert_eq!(result, 15.0);
    }
}