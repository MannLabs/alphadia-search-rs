use pyo3::prelude::*;
use numpy::ndarray::{Array1, Array2};
use std::time::Instant;
use std::cmp::min;
use rayon::prelude::*;

use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::SpecLibFlat;
use crate::kernel::GaussianKernel;
use crate::benchmark::benchmark_nonpadded_symmetric_simd;

use std::fs::File;
use ndarray_npy::NpzWriter;

const TMP_PATH: &str = "/Users/georgwallmann/Documents/data/alpha-rs/";

#[pyclass]
pub struct PeakGroupScoring {
    kernel: GaussianKernel,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(fwhm_rt: f32, kernel_size: usize) -> Self {
        Self {
            kernel: GaussianKernel::new(
                fwhm_rt,
                1.0, // sigma_scale_rt
                kernel_size,
                1.0, // rt_resolution
            ),
        }
    }

    pub fn search(&self, dia_data: &DIAData, lib: &SpecLibFlat, mass_tolerance: f32) -> PyResult<()> {
        let max_precursor_idx = min(10_000_000, lib.num_precursors());

        // store kernel to tmp file as npz
        let kernel_path = format!("{}/kernel.npz", TMP_PATH);
        let file = File::create(kernel_path).unwrap();
        let mut npz: NpzWriter<File> = NpzWriter::new(file);
        npz.add_array("kernel", &self.kernel.kernel_array).unwrap();
        npz.finish().unwrap();

        let start_time = Instant::now();
        // Parallel iteration over precursor indices
        (0..max_precursor_idx).into_par_iter()
            .for_each(|i| {
                let (precursor_mz, fragment_mz, fragment_intensity) = lib.get_precursor(i);
                self.search_precursor(
                    dia_data,
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

    pub fn search_precursor(
        &self,
        dia_data: &DIAData,
        precursor_mz: f32,
        fragment_mz: Vec<f32>,
        fragment_intensity: Vec<f32>,
        lib: &SpecLibFlat,
        mass_tolerance: f32
    ) {
        let valid_obs_idxs = dia_data.get_valid_observations(precursor_mz);

        let mut dense_xic: Array2<f32> = Array2::zeros((fragment_mz.len(), dia_data.rt_index.rt.len()));

        for obs_idx in valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations[obs_idx];

            for (f_idx, f_mz) in fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    &dia_data.mz_index, 
                    &mut dense_xic.row_mut(f_idx), 
                    mass_tolerance,
                    *f_mz
                );
            }

            let convolved_xic = benchmark_nonpadded_symmetric_simd(&self.kernel, &dense_xic);



            // We can now use self.kernel for peak scoring here
            // Example: apply 1D RT kernel to each row of dense_xic
            
            /*
            let path = "/Users/georgwallmann/Documents/data/alpha-rs/dense_xic.npz";
            let file = File::create(path).unwrap();
            let mut npz: NpzWriter<File> = NpzWriter::new(file);
            npz.add_array("dense_xic", &convolved_xic).unwrap();
            npz.finish().unwrap();
            */
            

            //let xic = &obs.xic_slices;
            //let xic_slice = xic.get_xic_slice(precursor_mz);
        }
    }
}

// Regular Rust implementation without PyO3 exposure
impl PeakGroupScoring {
    /// Applies a padded convolution to the input XIC data using the Gaussian kernel
    /// The padding handles edge effects by extending the data at boundaries
    fn padded_convolution(&self, xic: &Array2<f32>) -> Array2<f32> {
        let (n_fragments, n_points) = xic.dim();
        let kernel_size = self.kernel.kernel_array.len();
        let half_kernel = kernel_size / 2;
        
        // Create output array with same dimensions
        let mut convolved = Array2::zeros((n_fragments, n_points));
        
        // Process each fragment
        for f_idx in 0..n_fragments {
            let xic_row = xic.row(f_idx);
            let mut conv_row = convolved.row_mut(f_idx);
            
            // Apply convolution with padding for each point
            for i in 0..n_points {
                let mut sum = 0.0;
                
                for k in 0..kernel_size {
                    let idx = i as isize + (k as isize - half_kernel as isize);
                    let value = if idx < 0 {
                        // Left padding: mirror or use first value
                        xic_row[0]
                    } else if idx >= n_points as isize {
                        // Right padding: mirror or use last value
                        xic_row[n_points - 1]
                    } else {
                        xic_row[idx as usize]
                    };
                    
                    sum += value * self.kernel.kernel_array[k];
                }
                
                conv_row[i] = sum;
            }
        }
        
        convolved
    }
}

#[pyclass]
pub struct DIAData {
    pub mz_index: MZIndex,
    pub rt_index: RTIndex,
    pub quadrupole_observations: Vec<QuadrupoleObservation>,
}

#[pymethods]
impl DIAData {
    #[new]
    pub fn new() -> Self {
        Self {
            mz_index: MZIndex::new(),
            rt_index: RTIndex::new(),
            quadrupole_observations: Vec::new(),
        }
    }

    #[getter]
    pub fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    pub fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        let mut valid_observations = Vec::new();
        for (i, obs) in self.quadrupole_observations.iter().enumerate() {
            if obs.isolation_window[0] <= precursor_mz && obs.isolation_window[1] >= precursor_mz {
                valid_observations.push(i);
            }
        }
        valid_observations
    }
} 