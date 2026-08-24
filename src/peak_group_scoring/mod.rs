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
    calculate_correlation_safe, calculate_dot_product, calculate_fwhm_rt, calculate_hyperscore,
    calculate_hyperscore_inverse_mass_error, calculate_longest_ion_series, correlation_axis_0,
    filter_non_zero, intensity_ion_series, median_axis_0, normalize_profiles,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::utils::{
    calculate_fragment_mz_and_errors, calculate_median, calculate_std,
    calculate_weighted_mean_absolute_error, count_values_above, create_ranked_mask,
};
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
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => {
                        self.score_candidate_generic(dia_data, lib, &precursor, candidate)
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
        lib: &SpecLibFlat,
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

        // Filter to only non-zero profiles for median calculation
        let filtered_xic = filter_non_zero(&normalized_xic);

        let median_profile = median_axis_0(&normalized_xic);
        let median_profile_filtered = median_axis_0(&filtered_xic);

        let num_profiles = normalized_xic.shape()[0];
        let num_profiles_filtered = filtered_xic.shape()[0];

        // Calculate sum of median profile
        let median_profile_sum = median_profile.iter().sum::<f32>();
        let median_profile_sum_filtered = median_profile_filtered.iter().sum::<f32>();

        let fwhm_rt = calculate_fwhm_rt(
            &median_profile_filtered,
            cycle_start_idx,
            &dia_data.rt_index().rt,
        );

        // Calculate correlations of each profile with the median profile
        let correlations: Vec<f32> = correlation_axis_0(&median_profile_filtered, &normalized_xic);

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

        let median_correlation = calculate_median(&correlations);

        let correlation_std = calculate_std(&correlations);

        let num_over_95 = count_values_above(&correlations, 0.95, None);
        let num_over_90 = count_values_above(&correlations, 0.90, None);
        let num_over_80 = count_values_above(&correlations, 0.80, None);
        let num_over_50 = count_values_above(&correlations, 0.50, None);
        let num_over_0 = count_values_above(&correlations, 0.0, None);

        // Calculate ranked features using masks based on library intensities
        // Create masks selecting specific rank ranges
        let mask_0_5 = create_ranked_mask(&precursor.fragment_intensity, 0, 6); // ranks 0-5 (top 6)
        let mask_6_11 = create_ranked_mask(&precursor.fragment_intensity, 6, 12); // ranks 6-11 (next 6)
        let mask_12_17 = create_ranked_mask(&precursor.fragment_intensity, 12, 18); // ranks 12-17 (next 6)
        let mask_18_23 = create_ranked_mask(&precursor.fragment_intensity, 18, 24); // ranks 18-23 (next 6)

        // Calculate num_over_0 for each rank range
        let num_over_0_rank_0_5 = count_values_above(&correlations, 0.0, Some(&mask_0_5));
        let num_over_0_rank_6_11 = count_values_above(&correlations, 0.0, Some(&mask_6_11));
        let num_over_0_rank_12_17 = count_values_above(&correlations, 0.0, Some(&mask_12_17));
        let num_over_0_rank_18_23 = count_values_above(&correlations, 0.0, Some(&mask_18_23));

        // Calculate num_over_50 for each rank range
        let num_over_50_rank_0_5 = count_values_above(&correlations, 0.50, Some(&mask_0_5));
        let num_over_50_rank_6_11 = count_values_above(&correlations, 0.50, Some(&mask_6_11));
        let num_over_50_rank_12_17 = count_values_above(&correlations, 0.50, Some(&mask_12_17));
        let num_over_50_rank_18_23 = count_values_above(&correlations, 0.50, Some(&mask_18_23));

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
        let intensity_b_raw = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities.as_slice().unwrap(),
            &matched_mask_intensity,
            FragmentType::B,
        );

        let intensity_y_raw = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities.as_slice().unwrap(),
            &matched_mask_intensity,
            FragmentType::Y,
        );

        // Apply log10 transformation (add epsilon to avoid log(0))
        const EPSILON: f32 = 1e-8;
        let log10_b_ion_intensity = (intensity_b_raw + EPSILON).log10();
        let log10_y_ion_intensity = (intensity_y_raw + EPSILON).log10();

        // Calculate IDF values for this precursor's fragments
        let idf_values = lib.idf.get_idf(&precursor.fragment_mz_library);

        // Calculate IDF-based scores
        let idf_hyperscore = calculate_hyperscore(
            &precursor.fragment_type,
            &idf_values,
            &matched_mask_intensity,
        );

        let idf_xic_dot_product = calculate_dot_product(&idf_values, &correlations);
        let idf_intensity_dot_product =
            calculate_dot_product(&idf_values, observation_intensities.as_slice().unwrap());

        let log_idf_intensity_dot_product = (idf_intensity_dot_product + EPSILON).log10();

        // Create mask for top 6 IDF values
        let mask_top6_idf = create_ranked_mask(&idf_values, 0, 6); // ranks 0-5 (top 6 by IDF)

        // Calculate IDF-based correlation features
        let num_over_0_top6_idf = count_values_above(&correlations, 0.0, Some(&mask_top6_idf));
        let num_over_50_top6_idf = count_values_above(&correlations, 0.50, Some(&mask_top6_idf));

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
            num_over_0 as f32,
            num_over_0_rank_0_5 as f32,
            num_over_0_rank_6_11 as f32,
            num_over_0_rank_12_17 as f32,
            num_over_0_rank_18_23 as f32,
            num_over_50_rank_0_5 as f32,
            num_over_50_rank_6_11 as f32,
            num_over_50_rank_12_17 as f32,
            num_over_50_rank_18_23 as f32,
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            hyperscore_inverse_mass_error,
            rt_observed,
            delta_rt,
            longest_b_series as f32,
            longest_y_series as f32,
            precursor.naa as f32,
            weighted_mass_error,
            log10_b_ion_intensity,
            log10_y_ion_intensity,
            fwhm_rt,
            idf_hyperscore,
            idf_xic_dot_product,
            log_idf_intensity_dot_product,
            median_profile_sum,
            median_profile_sum_filtered,
            num_profiles as f32,
            num_profiles_filtered as f32,
            num_over_0_top6_idf as f32,
            num_over_50_top6_idf as f32,
        ))
    }
}
