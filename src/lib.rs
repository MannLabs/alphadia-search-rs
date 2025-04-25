use std::iter::zip;
use rayon::prelude::*;

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
use numpy::ndarray::{ArrayView1, ArrayViewMut1, ArrayBase, ViewRepr, Dim};

mod xic_index;
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

struct AlphaRawView<'py> {
    spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
}

impl<'py> AlphaRawView<'py> {
    fn new(
        spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            peak_mz,
            peak_intensity,
        }
    }
}


#[pyclass]
struct DIAData {
    mz_index: MZIndex,
    //rt_index: RTIndex,
    quadrupole_observations: Vec<QuadrupoleObservation>,
}

#[pymethods]
impl DIAData {
    #[new]
    fn new() -> Self {
        Self {
            mz_index: MZIndex::new(),
            quadrupole_observations: Vec::new(),
        }
    }

    #[getter]
    fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    #[getter]
    fn num_peaks(&self) -> usize {
        self.quadrupole_observations.iter().map(|q| q.xic_slices.iter().map(|x| x.cycle_index.len()).sum::<usize>()).sum()
    }
}

impl DIAData {
    fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {
        let mz_index = MZIndex::new();
        let num_quadrupole_observations = alpha_raw_view.spectrum_delta_scan_idx.iter().max().unwrap() + 1;
        
        // Parallel iteration over quadrupole observations
        let quadrupole_observations: Vec<QuadrupoleObservation> = (0..num_quadrupole_observations)
            .into_par_iter()
            .map(|i| QuadrupoleObservation::from_alpha_raw(&alpha_raw_view, i, &mz_index))
            .collect();
        
        Self {
            mz_index,
            quadrupole_observations,
        }
    }
}

#[derive(Debug)]
struct QuadrupoleObservation {
    isolation_window: [f32; 2],
    xic_slices: Vec<XICSlice>,
}

impl QuadrupoleObservation {
    fn new(mz_index: &MZIndex) -> Self {
        Self {
            isolation_window: [0.0, 0.0],
            xic_slices: vec![XICSlice::empty(); mz_index.len()],
        }
    }

    fn add_peak(&mut self, mz: f32, intensity: f32, cycle_idx: u16, mz_index: &MZIndex) {
        let closest_idx = mz_index.find_closest_index(mz);
        self.xic_slices[closest_idx].cycle_index.push(cycle_idx);
        self.xic_slices[closest_idx].intensity.push(intensity);
    }

    fn from_alpha_raw(alpha_raw_view: &AlphaRawView, delta_scan_idx: i64, mz_index: &MZIndex) -> Self {
        let mut quad_obs = Self::new(mz_index);
        let mut num_valid_scans = 0;
        let mut num_peaks = 0;
        
        // Find the first valid scan to get isolation window
        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == delta_scan_idx {
                if num_valid_scans == 0 {
                    // Set isolation window from the first valid scan
                    quad_obs.isolation_window = [
                        alpha_raw_view.isolation_lower_mz[i],
                        alpha_raw_view.isolation_upper_mz[i]
                    ];
                }

                // Get the peak indices for this scan
                let peak_start_idx = alpha_raw_view.spectrum_peak_start_idx[i] as usize;
                let peak_stop_idx = alpha_raw_view.spectrum_peak_stop_idx[i] as usize;
                let cycle_idx = alpha_raw_view.spectrum_cycle_idx[i] as u16;

                // Get the mz and intensity slices for this scan
                let mz_slice = alpha_raw_view.peak_mz.slice(s![peak_start_idx..peak_stop_idx]);
                let intensity_slice = alpha_raw_view.peak_intensity.slice(s![peak_start_idx..peak_stop_idx]);

                // Add each peak to the appropriate XIC slice
                for (mz_val, intensity_val) in zip(mz_slice.iter(), intensity_slice.iter()) {
                    quad_obs.add_peak(*mz_val, *intensity_val, cycle_idx, mz_index);
                    num_peaks += 1;
                }

                num_valid_scans += 1;
            }
        }

        println!("Quadrupole observation idx: {}, Cycles: {}, Peaks: {}", delta_scan_idx, num_valid_scans, num_peaks);

        quad_obs
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
    spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
    isolation_lower_mz: PyReadonlyArray1<'py, f32>,
    isolation_upper_mz: PyReadonlyArray1<'py, f32>,
    spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
    spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
    spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
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
        peak_mz.as_array(),
        peak_intensity.as_array(),
    );

    let dia_data = DIAData::from_alpha_raw(&alpha_raw_view);
    Ok(dia_data)
}

/// A Python module implemented in Rust.
#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_class::<Raw>()?;
    m.add_class::<SumContainer>()?;
    m.add_class::<DIAData>()?;
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

    #[test]
    fn test_quadrupole_observation() {
        // Create a new MZIndex
        let mz_index = MZIndex::new();
        
        // Create a new QuadrupoleObservation
        let mut quad_obs = QuadrupoleObservation::new(&mz_index);
        
        // Test initial state
        assert_eq!(quad_obs.isolation_window, [0.0, 0.0]);
        assert_eq!(quad_obs.xic_slices.len(), mz_index.len());
        
        // Add some test peaks
        let test_mz = 500.0; // Example m/z value
        let test_intensity = 1000.0;
        let test_cycle_idx = 1;
        
        quad_obs.add_peak(test_mz, test_intensity, test_cycle_idx, &mz_index);
        
        // Find the closest index for our test m/z
        let closest_idx = mz_index.find_closest_index(test_mz);
        
        // Verify the peak was added to the correct XIC slice
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[0], test_cycle_idx);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity[0], test_intensity);
        
        // Add another peak to the same m/z bin
        quad_obs.add_peak(test_mz, test_intensity * 2.0, test_cycle_idx + 1, &mz_index);
        
        // Verify both peaks are in the same XIC slice
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 2);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 2);
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[1], test_cycle_idx + 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity[1], test_intensity * 2.0);
    }
}