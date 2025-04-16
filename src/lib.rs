use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::PyArray1;
use numpy::ndarray::Array1;
use numpy::PyUntypedArray;
use numpy::PyArrayMethods;

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

/// A Python module implemented in Rust.
#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_class::<Raw>()?;
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
}