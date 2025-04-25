use std::iter::zip;
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::{Python, PyErr};
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use numpy::ndarray::s;
use numpy::ndarray::{ArrayBase, ViewRepr, Dim};

mod xic_index;
mod quadrupole_observation;
use xic_index::XICSlice;
use xic_index::ppm_index;
use xic_index::RESOLUTION_PPM;
use xic_index::MZ_START;
use xic_index::MZ_END;
use xic_index::MZIndex;
use quadrupole_observation::QuadrupoleObservation;

pub struct AlphaRawView<'py> {
    pub spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
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
    m.add_class::<DIAData>()?;
    m.add_function(wrap_pyfunction!(test_xic_index, m)?)?;
    Ok(())
}