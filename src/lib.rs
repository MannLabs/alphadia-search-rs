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
mod xic_index;
mod quadrupole_observation;
use xic_index::XICSlice;
use xic_index::ppm_index;
use xic_index::RESOLUTION_PPM;
use xic_index::MZ_START;
use xic_index::MZ_END;
use xic_index::MZIndex;
use quadrupole_observation::QuadrupoleObservation;


use ndarray_npy::NpzWriter;
use std::fs::File;

pub struct RTIndex {
    pub rt: Array1<f32>,
}

impl RTIndex {
    pub fn new() -> Self {
        Self {
            rt: Array1::from_vec(Vec::new()),
        }
    }

    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {

        let mut rt = Vec::new();

        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == 0 {
                rt.push(alpha_raw_view.spectrum_rt[i]);
            }
        }

        Self {
            rt: Array1::from_vec(rt),
        }
    }
}

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

#[pyclass]
struct DIAData {
    mz_index: MZIndex,
    rt_index: RTIndex,
    quadrupole_observations: Vec<QuadrupoleObservation>,
}

#[pymethods]
impl DIAData {
    #[new]
    fn new() -> Self {
        Self {
            mz_index: MZIndex::new(),
            rt_index: RTIndex::new(),
            quadrupole_observations: Vec::new(),
        }
    }

    #[getter]
    fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        let mut valid_observations = Vec::new();
        for (i, obs) in self.quadrupole_observations.iter().enumerate() {
            if obs.isolation_window[0] <= precursor_mz && obs.isolation_window[1] >= precursor_mz {
                valid_observations.push(i);
            }
        }
        valid_observations
    }

    fn search(&self, lib: &SpecLibFlat, mass_tolerance: f32) -> PyResult<()> {
        let max_precursor_idx = min(1_000_000, lib.num_precursors());

        let start_time = Instant::now();
        // Parallel iteration over precursor indices
        (0..max_precursor_idx).into_par_iter()
            .for_each(|i| {
                let (precursor_mz, fragment_mz, fragment_intensity) = lib.get_precursor(i);
                self.search_precursor(
                    precursor_mz,
                    fragment_mz,
                    fragment_intensity,
                    lib,
                    mass_tolerance
                );
            });
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let precursors_per_second = max_precursor_idx as f32 / duration.as_secs_f32();
        println!("Precursors per second: {:?}", precursors_per_second);

        Ok(())
    }

    fn search_precursor(
        &self,
        precursor_mz: f32,
        fragment_mz: Vec<f32>,
        fragment_intensity: Vec<f32>,
        lib: &SpecLibFlat,
        mass_tolerance: f32
    ) {

        let valid_obs_idxs = self.get_valid_observations(precursor_mz);

        let mut dense_xic: Array2<f32> = Array2::zeros((fragment_mz.len(), self.rt_index.rt.len()));

        for obs_idx in valid_obs_idxs {
            let obs = &self.quadrupole_observations[obs_idx];

            for (f_idx, f_mz) in fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    &self.mz_index, 
                    &mut dense_xic.row_mut(f_idx), 
                    mass_tolerance,
                    *f_mz
                );
            }
            
            /*
            let path = "/Users/georgwallmann/Documents/data/alpha-rs/dense_xic.npz";
            let file = File::create(path).unwrap();
            let mut npz: NpzWriter<File> = NpzWriter::new(file);
            npz.add_array("dense_xic", &dense_xic).unwrap();
            npz.finish().unwrap();
            */



            //let xic = &obs.xic_slices;
            //let xic_slice = xic.get_xic_slice(precursor_mz);
        }
    }
}

impl DIAData {
    fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {
        let mz_index = MZIndex::new();
        let num_quadrupole_observations = alpha_raw_view.spectrum_delta_scan_idx.iter().max().unwrap() + 1;

        let rt_index = RTIndex::from_alpha_raw(&alpha_raw_view);

        // Parallel iteration over quadrupole observations
        let quadrupole_observations: Vec<QuadrupoleObservation> = (0..num_quadrupole_observations)
            .into_par_iter()
            .map(|i| QuadrupoleObservation::from_alpha_raw(&alpha_raw_view, i, &mz_index))
            .collect();
        
        Self {
            mz_index,
            rt_index,
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

    let dia_data = DIAData::from_alpha_raw(&alpha_raw_view);
    Ok(dia_data)
}


#[pyclass]
struct SpecLibFlat {
    precursor_mz: Vec<f32>,
    precursor_start_idx: Vec<i64>,
    precursor_stop_idx: Vec<i64>,
    fragment_mz: Vec<f32>,
    fragment_intensity: Vec<f32>,
}

#[pymethods]
impl SpecLibFlat {
    #[new]
    fn new() -> Self {
        Self {
            precursor_mz: Vec::new(),
            precursor_start_idx: Vec::new(),
            precursor_stop_idx: Vec::new(),
            fragment_mz: Vec::new(),
            fragment_intensity: Vec::new(),
        }
    }

    #[staticmethod]
    fn from_arrays(
        precursor_mz: PyReadonlyArray1<'_, f32>,
        precursor_start_idx: PyReadonlyArray1<'_, i64>,
        precursor_stop_idx: PyReadonlyArray1<'_, i64>,
        fragment_mz: PyReadonlyArray1<'_, f32>,
        fragment_intensity: PyReadonlyArray1<'_, f32>,
    ) -> Self {
        Self {
            precursor_mz: precursor_mz.as_array().to_vec(),
            precursor_start_idx: precursor_start_idx.as_array().to_vec(),
            precursor_stop_idx: precursor_stop_idx.as_array().to_vec(),
            fragment_mz: fragment_mz.as_array().to_vec(),
            fragment_intensity: fragment_intensity.as_array().to_vec(),
        }
    }

    #[getter]
    fn num_precursors(&self) -> usize {
        self.precursor_mz.len()
    }

    #[getter]
    fn num_fragments(&self) -> usize {
        self.fragment_mz.len()
    }

    fn get_precursor(&self, index: usize) -> (f32, Vec<f32>, Vec<f32>) {
        let precursor_mz = self.precursor_mz[index];
        let start_idx = self.precursor_start_idx[index] as usize;
        let stop_idx = self.precursor_stop_idx[index] as usize;

        let fragment_mz = self.fragment_mz[start_idx..stop_idx].to_vec();
        let fragment_intensity = self.fragment_intensity[start_idx..stop_idx].to_vec();

        (precursor_mz, fragment_mz, fragment_intensity)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn alpha_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DIAData>()?;
    m.add_class::<SpecLibFlat>()?;
    m.add_function(wrap_pyfunction!(test_xic_index, m)?)?;
    Ok(())
}