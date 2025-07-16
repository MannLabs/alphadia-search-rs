use ndarray_npy::NpzWriter;
use numpy::ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::fs::File;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::convolution::convolution;
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::kernel::GaussianKernel;
use crate::precursor::Precursor;
use crate::score::axis_sqrt_dot_product;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;

pub mod parameters;
pub use parameters::ScoringParameters;

const TMP_PATH: &str = "/Users/georgwallmann/Documents/data/alpha-rs/";

/// Finds local maxima in a 1D array.
/// A local maximum is defined as a point that is higher than the 2 points to its left and right.
/// Returns a tuple of two vectors: (indices, values) sorted by value in descending order.
fn find_local_maxima(array: &Array1<f32>, offset: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let len = array.len();

    // Need at least 5 points to find a local maximum with 2 points on each side
    if len < 5 {
        return (indices, values);
    }

    // Check each point (except the first and last 2) for local maxima
    for i in 2..len - 2 {
        if array[i - 2] < array[i - 1]
            && array[i - 1] < array[i]
            && array[i] > array[i + 1]
            && array[i + 1] > array[i + 2]
        {
            indices.push(i + offset);
            values.push(array[i]);
        }
    }

    // Sort by value in descending order
    if !values.is_empty() {
        // Create index mapping for sorting
        let mut idx_map: Vec<usize> = (0..values.len()).collect();
        idx_map.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reorder both vectors using the sorted mapping
        let sorted_indices: Vec<usize> = idx_map.iter().map(|&i| indices[i]).collect();
        let sorted_values: Vec<f32> = idx_map.iter().map(|&i| values[i]).collect();

        indices = sorted_indices;
        values = sorted_values;
    }

    (indices, values)
}

#[pyclass]
pub struct PeakGroupScoring {
    kernel: GaussianKernel,
    params: ScoringParameters,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self {
            kernel: GaussianKernel::new(
                params.fwhm_rt,
                1.0, // sigma_scale_rt
                params.kernel_size,
                1.0, // rt_resolution
            ),
            params,
        }
    }

    pub fn search(&self, dia_data: &DIAData, lib: &SpecLibFlat) -> CandidateCollection {
        self.search_generic(dia_data, lib)
    }

    pub fn search_next_gen(
        &self,
        dia_data: &DIADataNextGen,
        lib: &SpecLibFlat,
    ) -> CandidateCollection {
        self.search_generic(dia_data, lib)
    }
}

impl PeakGroupScoring {
    /// Generic search function that works with any type implementing DIADataTrait
    fn search_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
    ) -> CandidateCollection {
        let max_precursor_idx = min(10_000_000, lib.num_precursors());

        // store kernel to tmp file as npz
        let kernel_path = format!("{TMP_PATH}/kernel.npz");
        let file = File::create(kernel_path).unwrap();
        let mut npz: NpzWriter<File> = NpzWriter::new(file);
        npz.add_array("kernel", &self.kernel.kernel_array).unwrap();
        npz.finish().unwrap();

        let start_time = Instant::now();
        // Parallel iteration over precursor indices with flat_map to collect candidates
        let candidates: Vec<Candidate> = (0..max_precursor_idx)
            .into_par_iter()
            .flat_map(|i| {
                let precursor = lib.get_precursor(i);
                self.search_precursor_generic(
                    dia_data,
                    &precursor,
                    self.params.mass_tolerance,
                    self.params.rt_tolerance,
                )
            })
            .collect();
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let precursors_per_second = max_precursor_idx as f32 / duration.as_secs_f32();
        println!("Precursors per second: {precursors_per_second:?}");
        println!("Found {} candidates", candidates.len());

        CandidateCollection::from_vec(candidates)
    }

    /// Generic precursor search function that works with any type implementing DIADataTrait
    fn search_precursor_generic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        mass_tolerance: f32,
        rt_tolerance: f32,
    ) -> Vec<Candidate> {
        let valid_obs_idxs = dia_data.get_valid_observations(precursor.mz);

        let (cycle_start_idx, cycle_stop_idx) = dia_data
            .rt_index()
            .get_cycle_idx_limits(precursor.rt, rt_tolerance);

        let mut dense_xic: Array2<f32> = Array2::zeros((
            precursor.fragment_mz.len(),
            cycle_stop_idx - cycle_start_idx,
        ));

        for obs_idx in valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];

            for (f_idx, f_mz) in precursor.fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    dia_data.mz_index(),
                    &mut dense_xic.row_mut(f_idx),
                    cycle_start_idx,
                    cycle_stop_idx,
                    mass_tolerance,
                    *f_mz,
                );
            }
        }

        let convolved_xic = convolution(&self.kernel, &dense_xic);

        let score = axis_sqrt_dot_product(&convolved_xic, &precursor.fragment_intensity);

        let (local_maxima_indices, local_maxima_values) =
            find_local_maxima(&score, cycle_start_idx);

        // Take top 3 maxima (they're already sorted by value in descending order)
        let max_count = std::cmp::min(3, local_maxima_indices.len());

        let mut candidates = Vec::new();

        for i in 0..max_count {
            let cycle_center_idx = local_maxima_indices[i];
            let score = local_maxima_values[i];

            let cycle_start_idx = max(0, cycle_center_idx - self.params.peak_length);
            let cycle_stop_idx = min(
                cycle_center_idx + self.params.peak_length + 1,
                dia_data.rt_index().len(),
            );

            let candidate = Candidate::new(
                precursor.idx,
                i + 1,
                score,
                cycle_start_idx,
                cycle_center_idx,
                cycle_stop_idx,
            );

            candidates.push(candidate);
        }

        candidates
    }

    // Keep the old methods for backwards compatibility, but they now delegate to the generic version
    pub fn search_precursor(
        &self,
        dia_data: &DIAData,
        precursor: &Precursor,
        mass_tolerance: f32,
        rt_tolerance: f32,
    ) -> Vec<Candidate> {
        self.search_precursor_generic(dia_data, precursor, mass_tolerance, rt_tolerance)
    }

    pub fn search_precursor_next_gen(
        &self,
        dia_data: &DIADataNextGen,
        precursor: &Precursor,
        mass_tolerance: f32,
        rt_tolerance: f32,
    ) -> Vec<Candidate> {
        self.search_precursor_generic(dia_data, precursor, mass_tolerance, rt_tolerance)
    }
}

#[cfg(test)]
mod tests;
