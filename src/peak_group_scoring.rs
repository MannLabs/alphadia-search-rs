use pyo3::prelude::*;
use numpy::ndarray::Array2;
use std::time::Instant;
use std::cmp::min;
use rayon::prelude::*;
use std::fs::File;
use ndarray_npy::NpzWriter;

use crate::kernel::GaussianKernel;
use crate::benchmark::benchmark_nonpadded_symmetric_simd;
use crate::precursor::Precursor;
use crate::SpecLibFlat;
use crate::dia_data::DIAData;

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
                let precursor = lib.get_precursor(i);
                self.search_precursor(
                    dia_data,
                    &precursor,
                    mass_tolerance
                );
            });
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let precursors_per_second = max_precursor_idx as f32 / duration.as_secs_f32();
        println!("Precursors per second: {:?}", precursors_per_second);

        Ok(())
    }
}

impl PeakGroupScoring {
    pub fn search_precursor(
        &self,
        dia_data: &DIAData,
        precursor: &Precursor,
        mass_tolerance: f32
    ) {
        let valid_obs_idxs = dia_data.get_valid_observations(precursor.mz);

        let mut dense_xic: Array2<f32> = Array2::zeros((precursor.fragment_mz.len(), dia_data.rt_index.rt.len()));

        for obs_idx in valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations[obs_idx];

            for (f_idx, f_mz) in precursor.fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    &dia_data.mz_index, 
                    &mut dense_xic.row_mut(f_idx), 
                    mass_tolerance,
                    *f_mz
                );
            }

            let convolved_xic = benchmark_nonpadded_symmetric_simd(&self.kernel, &dense_xic);
        }
    }
} 