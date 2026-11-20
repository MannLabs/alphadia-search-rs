use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;
use std::time::Instant;

use crate::any_dia_data::AnyDIAData;
use crate::candidate::{Candidate, CandidateCollection};
use crate::convolution::convolution;
use crate::dense_xic_observation::DenseXICObservation;
use crate::kernel::GaussianKernel;
use crate::precursor::Precursor;
use crate::score::axis_log_dot_product;
use crate::traits::{DIADataTrait, MobilityFilter};

/// Each mobility sub-window of the selection is this many times the bin spacing wide, so that a peak between two bin
/// centres lies fully inside one of them.
const MOBILITY_BIN_OVERLAP: f32 = 2.0;
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

/// Reorders the maxima by their score plus a gaussian log prior on the distance of their RT to the precursor's
/// expected RT (sd `sigma` seconds). The values stay the raw scores.
///
/// Over a wide RT window, interfering peaks far from the expected RT outscore a faint true peak near it; the prior
/// keeps the near peak among the candidates without changing the score the classifier sees.
fn rank_by_rt_prior(
    indices: &mut Vec<usize>,
    values: &mut Vec<f32>,
    cycle_rt: &Array1<f32>,
    expected_rt: f32,
    sigma: f32,
) {
    let mut order: Vec<(usize, f32, f32)> = indices
        .iter()
        .zip(values.iter())
        .map(|(&idx, &value)| {
            let z = (cycle_rt[idx] - expected_rt) / sigma;
            (idx, value, value - 0.5 * z * z)
        })
        .collect();
    order.sort_by(|a, b| b.2.total_cmp(&a.2));
    *indices = order.iter().map(|o| o.0).collect();
    *values = order.iter().map(|o| o.1).collect();
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

    pub fn search(&self, dia_data: AnyDIAData<'_>, lib: &SpecLibFlat) -> CandidateCollection {
        crate::any_dia_data::with_dia_data!(dia_data, |data| self.search_generic(data, lib))
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

    /// The selection score of every cycle in [cycle_start_idx, cycle_stop_idx).
    ///
    /// With mobility_selection_bins > 1 the mobility window is split into overlapping sub-windows, each scored on its
    /// own, and a cycle keeps its best sub-window's score: a precursor at one mobility then no longer competes with the
    /// summed signal of everything else eluting within the whole window.
    fn selection_score<T: DIADataTrait>(
        &self,
        dia_data: &T,
        kernel: &GaussianKernel,
        precursor: &Precursor,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
    ) -> Array1<f32> {
        let score_of = |filter: MobilityFilter| {
            let dense_xic_obs = DenseXICObservation::new(
                dia_data,
                precursor.mz,
                filter,
                cycle_start_idx,
                cycle_stop_idx,
                mass_tolerance,
                &precursor.fragment_mz,
            );
            let convolved_xic = convolution(kernel, &dense_xic_obs.dense_xic);
            axis_log_dot_product(&convolved_xic, &precursor.fragment_intensity)
        };
        let filter = precursor.mobility_filter(self.params.mobility_tolerance);
        let bins = self.params.mobility_selection_bins;
        match filter {
            MobilityFilter::Window {
                mobility,
                tolerance,
            } if bins > 1 => {
                let step = 2.0 * tolerance / bins as f32;
                let half_width = 0.5 * step * MOBILITY_BIN_OVERLAP;
                let mut best: Option<Array1<f32>> = None;
                for bin in 0..bins {
                    let center = mobility - tolerance + (bin as f32 + 0.5) * step;
                    let score = score_of(MobilityFilter::Window {
                        mobility: center,
                        tolerance: half_width,
                    });
                    best = Some(match best {
                        None => score,
                        Some(b) => numpy::ndarray::Zip::from(&b)
                            .and(&score)
                            .map_collect(|&x, &y| x.max(y)),
                    });
                }
                best.expect("bins > 1")
            }
            _ => score_of(filter),
        }
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

        let score = self.selection_score(
            dia_data,
            kernel,
            precursor,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
        );

        let (mut local_maxima_indices, mut local_maxima_values) =
            find_local_maxima(&score, cycle_start_idx);
        if self.params.rt_prior_sigma > 0.0 {
            rank_by_rt_prior(
                &mut local_maxima_indices,
                &mut local_maxima_values,
                &dia_data.rt_index().rt,
                precursor.rt,
                self.params.rt_prior_sigma,
            );
        }

        // Take top maxima (they're already sorted by value in descending order)
        let max_count = std::cmp::min(candidate_count, local_maxima_indices.len());

        let mut candidates = Vec::new();

        for i in 0..max_count {
            let cycle_center_idx = local_maxima_indices[i];
            let score = local_maxima_values[i];

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
