//! Peak group quantification operates only on candidates scored at 1% FDR.
//!
//! Avoids storing memory-intensive fragment-level properties for all possible candidates
//! (e.g., 3M precursors × 3 candidates × target+decoy = 18M candidates) by quantifying
//! only the pre-filtered high-confidence subset.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::DIAData;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, filter_non_zero, median_axis_0, normalize_profiles,
};
use crate::precursor_quantified::PrecursorQuantified;
use crate::traits::DIADataTrait;
use crate::utils::calculate_fragment_mz_and_errors;
use crate::{SpecLibFlat, SpecLibFlatQuantified};
use numpy::ndarray::{Array2, Axis};

pub mod parameters;
pub mod tests;
pub use parameters::QuantificationParameters;
use parameters::{BACKGROUND_EDGES, BACKGROUND_PROFILE, BACKGROUND_WEIGHTED};

/// Summed fragment intensities over the quantification window with a flat per-fragment background removed.
///
/// The window spans several peak widths, so a plain sum adds the background of every cycle; at low abundance that
/// background dominates and ratios between runs are compressed towards 1.
/// - edges: the background is the mean of the first and last cycle, where the peak has decayed; the sum of the
///   background-corrected cycles, clipped at 0.
/// - profile: least-squares fit x = a * profile + b per fragment against the precursor's median profile; the signal
///   is a * sum(profile), clipped at 0. Uses every cycle for the background estimate, unlike edges. Falls back to the
///   plain sum when the profile is flat.
/// - weighted: least-squares fit x = a * profile per fragment without an intercept (a matched filter); the signal is
///   a * sum(profile). Nothing is subtracted, so no estimated background adds noise; cycles far from the apex get
///   little weight, so of a flat background only (sum p)^2 / sum p^2 cycles' worth is kept instead of the whole
///   window. Falls back to the plain sum when the profile is empty.
///
/// When every fragment clips at 0 the plain sums are returned: the quantified table is the base of the PSM output,
/// so an all-zero precursor would lose its identification, not only its quantity.
pub fn background_corrected_intensities(
    xic: &Array2<f32>,
    profile: &[f32],
    mode: &str,
) -> Vec<f32> {
    let (num_fragments, num_cycles) = xic.dim();
    let sums: Vec<f32> = xic.sum_axis(Axis(1)).to_vec();
    if num_cycles == 0 {
        return sums;
    }
    let corrected: Vec<f32> = match mode {
        BACKGROUND_EDGES => (0..num_fragments)
            .map(|f| {
                let row = xic.row(f);
                let background = 0.5 * (row[0] + row[num_cycles - 1]);
                row.iter().map(|x| x - background).sum::<f32>().max(0.0)
            })
            .collect(),
        BACKGROUND_PROFILE => {
            let n = num_cycles as f32;
            let profile_sum: f32 = profile.iter().sum();
            let profile_mean = profile_sum / n;
            let profile_ss: f32 = profile.iter().map(|p| (p - profile_mean).powi(2)).sum();
            if profile_ss <= f32::EPSILON || profile_sum <= 0.0 {
                return sums;
            }
            (0..num_fragments)
                .map(|f| {
                    let row = xic.row(f);
                    let mean = sums[f] / n;
                    let covariance: f32 = row
                        .iter()
                        .zip(profile)
                        .map(|(x, p)| (x - mean) * (p - profile_mean))
                        .sum();
                    (covariance / profile_ss * profile_sum).max(0.0)
                })
                .collect()
        }
        BACKGROUND_WEIGHTED => {
            let profile_sum: f32 = profile.iter().sum();
            let profile_square_sum: f32 = profile.iter().map(|p| p * p).sum();
            if profile_square_sum <= f32::EPSILON || profile_sum <= 0.0 {
                return sums;
            }
            (0..num_fragments)
                .map(|f| {
                    let projection: f32 = xic.row(f).iter().zip(profile).map(|(x, p)| x * p).sum();
                    (projection / profile_square_sum * profile_sum).max(0.0)
                })
                .collect()
        }
        _ => return sums,
    };
    if corrected.iter().all(|&x| x <= 0.0) {
        return sums;
    }
    corrected
}

#[pyclass]
pub struct PeakGroupQuantification {
    params: QuantificationParameters,
}

#[pymethods]
impl PeakGroupQuantification {
    #[new]
    pub fn new(params: QuantificationParameters) -> Self {
        Self { params }
    }

    pub fn quantify(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        self.quantify_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupQuantification {
    fn quantify_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        let start_time = Instant::now();

        let quantified_precursors: Vec<PrecursorQuantified> = candidates
            .par_iter()
            .filter_map(|candidate| {
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true,
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => self.quantify_precursor(dia_data, &precursor, candidate),
                    None => None,
                }
            })
            .collect();

        let duration = start_time.elapsed();
        println!(
            "Peak group quantification completed in {:.2}s for {} precursors",
            duration.as_secs_f64(),
            quantified_precursors.len()
        );

        SpecLibFlatQuantified::from_precursor_quantified_vec(quantified_precursors)
    }

    fn quantify_precursor<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &crate::precursor::Precursor,
        candidate: &Candidate,
    ) -> Option<PrecursorQuantified> {
        let cycle_start = candidate.cycle_start;
        let cycle_stop = candidate.cycle_stop;
        let cycle_center = candidate.cycle_center;

        let num_cycles = cycle_stop - cycle_start;

        if num_cycles == 0 || cycle_center < cycle_start || cycle_center >= cycle_stop {
            return None;
        }

        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            cycle_start,
            cycle_stop,
            self.params.tolerance_ppm,
            &precursor.fragment_mz,
        );

        let num_fragments = precursor.fragment_mz.len();
        let mut fragment_correlation_observed = vec![0.0f32; num_fragments];

        let _center_cycle_idx = cycle_center - cycle_start;

        // Calculate observed m/z values and mass errors for all fragments
        let (fragment_mz_observed, fragment_mass_error_observed) = calculate_fragment_mz_and_errors(
            &dense_xic_mz_obs.dense_mz,
            &dense_xic_mz_obs.dense_xic,
            &precursor.fragment_mz,
        );

        let normalized_xic = normalize_profiles(&dense_xic_mz_obs.dense_xic, 1);

        // Filter to only non-zero profiles for median calculation (same as in scoring)
        let filtered_xic = filter_non_zero(&normalized_xic);
        let median_profile = median_axis_0(&filtered_xic);

        for fragment_idx in 0..num_fragments {
            let fragment_profile = normalized_xic.row(fragment_idx);
            let correlation = calculate_correlation_safe(
                fragment_profile.as_slice().unwrap_or(&[]),
                &median_profile,
            );
            fragment_correlation_observed[fragment_idx] = correlation;
        }

        let observation_intensities = background_corrected_intensities(
            &dense_xic_mz_obs.dense_xic,
            &median_profile,
            &self.params.background_subtraction,
        );

        let rt_observed = dia_data.rt_index().rt[cycle_center];

        let precursor_quantified = PrecursorQuantified {
            precursor_idx: precursor.precursor_idx,
            mz_library: precursor.mz_library,
            mz: precursor.mz,
            rt_library: precursor.rt_library,
            rt: precursor.rt,
            naa: precursor.naa,
            rank: candidate.rank,
            rt_observed,
            // Clone is necessary because we only have a borrowed reference (&Precursor) to the precursor,
            // but PrecursorQuantified needs to own its Vec<T> data. Since Vec<T> contains heap-allocated
            // data, we must clone to create new owned copies rather than trying to move from a borrowed value.
            fragment_mz_library: precursor.fragment_mz_library.clone(),
            fragment_mz: precursor.fragment_mz.clone(),
            fragment_intensity: observation_intensities,
            fragment_cardinality: precursor.fragment_cardinality.clone(),
            fragment_charge: precursor.fragment_charge.clone(),
            fragment_loss_type: precursor.fragment_loss_type.clone(),
            fragment_number: precursor.fragment_number.clone(),
            fragment_position: precursor.fragment_position.clone(),
            fragment_type: precursor.fragment_type.clone(),
            fragment_mz_observed,
            fragment_correlation_observed,
            fragment_mass_error_observed,
        };

        precursor_quantified.filter_fragments_by_intensity(0.0)
    }
}
