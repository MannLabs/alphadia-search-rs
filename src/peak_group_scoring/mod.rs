use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{
    Candidate, CandidateCollection, CandidateFeature, CandidateFeatureCollection,
};
use crate::constants::FragmentType;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::DIAData;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, calculate_hyperscore, calculate_hyperscore_inverse_mass_error,
    calculate_longest_ion_series, correlation_axis_0, intensity_ion_series, median_axis_0,
    normalize_profiles,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::utils::{calculate_fragment_mz_and_errors, calculate_weighted_mean_absolute_error};
use crate::SpecLibFlat;
use numpy::ndarray::Axis;

pub mod parameters;
pub mod tests;
pub mod utils;
pub use parameters::ScoringParameters;

#[pyclass]
#[allow(dead_code)]
pub struct PeakGroupScoring {
    params: ScoringParameters,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self { params }
    }

    pub fn score(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> CandidateFeatureCollection {
        self.score_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupScoring {
    /// Generic scoring function that works with any type implementing DIADataTrait
    fn score_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> CandidateFeatureCollection {
        let start_time = Instant::now();

        // Parallel iteration over candidates to score each one
        let scored_candidates: Vec<CandidateFeature> = candidates
            .par_iter()
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true, // Always filter non-zero intensities for scoring
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => {
                        self.score_candidate_generic(dia_data, &precursor, candidate)
                    }
                    None => {
                        eprintln!(
                            "Warning: Candidate precursor_idx {} not found in library. Skipping.",
                            candidate.precursor_idx
                        );
                        None
                    }
                }
            })
            .collect();

        // Create collection from Vec
        let feature_collection = CandidateFeatureCollection::from_vec(scored_candidates);

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = candidates.len() as f32 / duration.as_secs_f32();
        println!(
            "Scored {} candidates at {:.2} candidates/second",
            candidates.len(),
            candidates_per_second
        );

        feature_collection
    }

    /// Generic candidate scoring function that works with any type implementing DIADataTrait
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
    ) -> Option<CandidateFeature> {
        // Scoring implementation for individual candidate will be added here
        // For now, return the original score

        let cycle_start_idx = candidate.cycle_start;
        let cycle_stop_idx = candidate.cycle_stop;
        let mass_tolerance = self.params.mass_tolerance;

        // Create dense XIC and m/z observation using the filtered precursor fragments
        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            &precursor.fragment_mz,
        );

        // Normalize the profiles before calculating median
        let normalized_xic = normalize_profiles(&dense_xic_mz_obs.dense_xic, 1);
        let median_profile = median_axis_0(&normalized_xic);

        // Calculate correlations of each profile with the median profile
        let correlations = correlation_axis_0(&median_profile, &normalized_xic);

        let observation_intensities = dense_xic_mz_obs.dense_xic.sum_axis(Axis(1));

        let intensity_correlations = calculate_correlation_safe(
            observation_intensities.as_slice().unwrap(),
            &precursor.fragment_intensity,
        );

        // all of this part is highly experimental and needs to be refined

        // Calculate feature values (using score as proxy for now)
        let mean_correlation = if !correlations.is_empty() {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        } else {
            0.0
        };

        let mut sorted_correlations = correlations.clone();
        sorted_correlations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_correlation = if !sorted_correlations.is_empty() {
            let mid = sorted_correlations.len() / 2;
            if sorted_correlations.len() % 2 == 0 {
                (sorted_correlations[mid - 1] + sorted_correlations[mid]) / 2.0
            } else {
                sorted_correlations[mid]
            }
        } else {
            0.0
        };

        let correlation_std = if correlations.len() > 1 {
            let variance = correlations
                .iter()
                .map(|&x| (x - mean_correlation).powi(2))
                .sum::<f32>()
                / (correlations.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let num_over_95 = correlations.iter().filter(|&x| *x > 0.95).count();
        let num_over_90 = correlations.iter().filter(|&x| *x > 0.90).count();
        let num_over_80 = correlations.iter().filter(|&x| *x > 0.80).count();
        let num_over_50 = correlations.iter().filter(|&x| *x > 0.50).count();

        let intensity_correlation = intensity_correlations;
        let num_fragments = precursor.fragment_mz.len();
        let num_scans = cycle_stop_idx - cycle_start_idx;

        let matched_mask_intensity: Vec<bool> =
            observation_intensities.iter().map(|&x| x > 0.0).collect();
        let observation_intensities_slice = observation_intensities.as_slice().unwrap();

        let hyperscore_intensity_observation = calculate_hyperscore(
            &precursor.fragment_type,
            observation_intensities_slice,
            &matched_mask_intensity,
        );

        let hyperscore_intensity_library = calculate_hyperscore(
            &precursor.fragment_type,
            &precursor.fragment_intensity,
            &matched_mask_intensity,
        );

        // Calculate longest continuous ion series
        let (longest_b_series, longest_y_series) = calculate_longest_ion_series(
            &precursor.fragment_type,
            &precursor.fragment_number,
            &matched_mask_intensity,
        );

        // Calculate fragment m/z and mass errors
        let (_fragment_mz_observed, fragment_mass_errors) = calculate_fragment_mz_and_errors(
            &dense_xic_mz_obs.dense_mz,
            &dense_xic_mz_obs.dense_xic,
            &precursor.fragment_mz,
        );

        // Calculate weighted mean absolute mass error using library intensities
        let weighted_mass_error = calculate_weighted_mean_absolute_error(
            &fragment_mass_errors,
            &precursor.fragment_intensity,
        );

        // Calculate hyperscore with inverse mass error weighting
        // Use observed intensities (sum across cycles) and exclude zero intensity fragments
        let hyperscore_inverse_mass_error = calculate_hyperscore_inverse_mass_error(
            &precursor.fragment_type,
            observation_intensities.as_slice().unwrap(),
            &matched_mask_intensity,
            &fragment_mass_errors,
        );

        // Calculate retention time features
        let rt_observed = dia_data.rt_index().rt[candidate.cycle_center];
        let delta_rt = rt_observed - precursor.rt;

        // Calculate intensity scores for b and y series
        let intensity_b_series = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities.as_slice().unwrap(),
            &matched_mask_intensity,
            FragmentType::B,
        );

        let intensity_y_series = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities.as_slice().unwrap(),
            &matched_mask_intensity,
            FragmentType::Y,
        );

        // Create and return candidate feature
        Some(CandidateFeature::new(
            candidate.precursor_idx,
            candidate.rank,
            candidate.score,
            mean_correlation,
            median_correlation,
            correlation_std,
            intensity_correlation,
            num_fragments as f32,
            num_scans as f32,
            num_over_95 as f32,
            num_over_90 as f32,
            num_over_80 as f32,
            num_over_50 as f32,
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            hyperscore_inverse_mass_error,
            rt_observed,
            delta_rt,
            longest_b_series as f32,
            longest_y_series as f32,
            precursor.naa as f32,
            weighted_mass_error,
            intensity_b_series,
            intensity_y_series,
        ))
    }
}
