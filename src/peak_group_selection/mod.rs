use numpy::ndarray::{s, Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::convolution::convolution;
use crate::dense_xic_observation::DenseXICObservation;
use crate::dia_data::DIAData;
use crate::kernel::GaussianKernel;
use crate::precursor::Precursor;
use crate::score::axis_log_dot_product;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;

pub mod parameters;
pub use parameters::SelectionParameters;

/// The cycle range [start, stop) of a candidate: peak_length cycles on each side of its apex, clipped to the run.
///
/// An apex can lie within peak_length cycles of the first cycle since the convolution smooths up to the run ends,
/// so the start must not underflow.
fn candidate_cycle_limits(
    cycle_center_idx: usize,
    peak_length: usize,
    n_cycles: usize,
) -> (usize, usize) {
    (
        cycle_center_idx.saturating_sub(peak_length),
        min(cycle_center_idx + peak_length + 1, n_cycles),
    )
}

/// Finds local maxima in a 1D array.
/// A local maximum is a point strictly higher than its direct neighbours on both sides.
/// Returns a tuple of two vectors: (indices, values) sorted by value in descending order.
/// Add the log density of a Student-t prior on the retention time to the selection score of each cycle.
///
/// `score[j]` belongs to the cycle `start + j`, whose retention time is `rt[start + j]`. The log density is shifted to
/// zero at `predicted_rt`, so a peak at the predicted retention time keeps its score.
fn add_rt_log_prior(
    score: &mut Array1<f32>,
    rt: &Array1<f32>,
    start: usize,
    predicted_rt: f32,
    scale: f32,
    df: f32,
) {
    let exponent = -0.5 * (df + 1.0);
    for (j, value) in score.iter_mut().enumerate() {
        let deviation = (rt[start + j] - predicted_rt) / scale;
        *value += exponent * (1.0 + deviation * deviation / df).ln();
    }
}

/// Count the fragments with signal within one cycle of the apex, as `num_profiles_filtered` of the scoring does.
///
/// `dense_xic` holds one row per fragment and one column per cycle; `apex` is a column index.
fn fragments_with_signal(dense_xic: &Array2<f32>, apex: usize) -> usize {
    let start = apex.saturating_sub(1);
    let stop = min(apex + 2, dense_xic.ncols());
    dense_xic
        .rows()
        .into_iter()
        .filter(|row| row.slice(s![start..stop]).iter().any(|&value| value > 0.0))
        .count()
}

fn find_local_maxima(array: &Array1<f32>, offset: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let len = array.len();

    if len < 3 {
        return (indices, values);
    }

    // Only the direct neighbours are compared: at short gradients a peak spans about one cycle, so the
    // points two cycles away are background, and requiring them to be monotonic rejects real peaks.
    for i in 1..len - 1 {
        if array[i - 1] < array[i] && array[i] > array[i + 1] {
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
pub struct PeakGroupSelection {
    params: SelectionParameters,
}

#[pymethods]
impl PeakGroupSelection {
    #[new]
    pub fn new(params: SelectionParameters) -> Self {
        Self { params }
    }

    pub fn search(&self, dia_data: &DIAData, lib: &SpecLibFlat) -> CandidateCollection {
        self.search_generic(dia_data, lib)
    }
}

impl PeakGroupSelection {
    /// Generic search function that works with any type implementing DIADataTrait
    fn search_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
    ) -> CandidateCollection {
        let start_time = Instant::now();
        // fwhm_rt is in seconds, the convolution runs on the cycle grid.
        let kernel = GaussianKernel::new(
            self.params.fwhm_rt,
            1.0, // sigma_scale_rt
            self.params.kernel_size,
            dia_data.rt_index().delta_t(),
        );
        // Parallel iteration over precursor indices with filter_map to collect candidates
        let candidates: Vec<Candidate> = (0..lib.num_precursors())
            .into_par_iter()
            .filter_map(|i| {
                let precursor = lib.get_precursor_filtered(
                    i,
                    true, // Always filter non-zero intensities for scoring
                    true, // Filter Y1 ions by default
                    self.params.min_ion_number,
                    self.params.top_k_fragments,
                );
                self.search_precursor_generic(
                    dia_data,
                    &kernel,
                    &precursor,
                    self.params.mass_tolerance,
                    self.params.rt_tolerance,
                    self.params.candidate_count,
                )
            })
            .flatten()
            .collect();
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let precursors_per_second = lib.num_precursors() as f32 / duration.as_secs_f32();
        println!("Precursors per second: {precursors_per_second:?}");
        println!("Found {} candidates", candidates.len());

        CandidateCollection::from_vec(candidates)
    }

    /// Generic precursor search function that works with any type implementing DIADataTrait
    fn search_precursor_generic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        kernel: &GaussianKernel,
        precursor: &Precursor,
        mass_tolerance: f32,
        rt_tolerance: f32,
        candidate_count: usize,
    ) -> Option<Vec<Candidate>> {
        let (cycle_start_idx, cycle_stop_idx) = dia_data.rt_index().get_cycle_idx_limits(
            precursor.rt,
            rt_tolerance,
            self.params.kernel_size,
        );

        // Validate cycle range
        if cycle_stop_idx <= cycle_start_idx {
            return None;
        }

        // Create dense XIC observation using the filtered precursor fragments
        let dense_xic_obs = DenseXICObservation::new(
            dia_data,
            precursor.mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            &precursor.fragment_mz,
        );

        let convolved_xic = convolution(kernel, &dense_xic_obs.dense_xic);

        let mut score = axis_log_dot_product(&convolved_xic, &precursor.fragment_intensity);

        if self.params.rt_prior_scale > 0.0 && self.params.rt_prior_df > 0.0 {
            add_rt_log_prior(
                &mut score,
                &dia_data.rt_index().rt,
                cycle_start_idx,
                precursor.rt,
                self.params.rt_prior_scale,
                self.params.rt_prior_df,
            );
        }

        let (local_maxima_indices, local_maxima_values) =
            find_local_maxima(&score, cycle_start_idx);

        // The maxima are sorted by score in descending order. Take the best ones with enough fragments.
        let mut candidates = Vec::new();

        for (&cycle_center_idx, &score) in local_maxima_indices.iter().zip(&local_maxima_values) {
            // The maxima are sorted, so no later one is above the minimum score either.
            if candidates.len() == candidate_count || score <= self.params.min_score {
                break;
            }
            if self.params.min_fragments > 0
                && fragments_with_signal(
                    &dense_xic_obs.dense_xic,
                    cycle_center_idx - cycle_start_idx,
                ) < self.params.min_fragments
            {
                continue;
            }
            let i = candidates.len();

            let (cycle_start_idx, cycle_stop_idx) = candidate_cycle_limits(
                cycle_center_idx,
                self.params.peak_length,
                dia_data.rt_index().len(),
            );

            let candidate = Candidate::new(
                precursor.precursor_idx,
                i,
                score,
                cycle_start_idx,
                cycle_center_idx,
                cycle_stop_idx,
            );

            candidates.push(candidate);
        }

        Some(candidates)
    }
}

#[cfg(test)]
mod tests;
