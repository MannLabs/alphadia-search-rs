//! The features of every feature set: fragment co-elution, hyperscores, mass and retention time errors and the
//! intensities of the b and y series. They depend on the candidate alone.

use numpy::ndarray::Array1;

use crate::candidate::Candidate;
use crate::constants::FragmentType;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::utils::{
    calculate_fwhm_rt, calculate_hyperscore, calculate_hyperscore_inverse_mass_error,
    calculate_longest_ion_series, intensity_ion_series,
};
use crate::precursor::Precursor;
use crate::utils::{
    calculate_fragment_mz_and_errors, calculate_median, calculate_std,
    calculate_weighted_mean_absolute_error, count_values_above, create_ranked_mask,
};

/// Added to the b and y series intensities before the log10, so that a series without signal stays finite
const LOG_INTENSITY_EPSILON: f32 = 1e-8;

feature_block! {
    pub struct CoreFeatures {
        /// score of the candidate selection
        pub score: f32,
        pub mean_correlation: f32,
        pub median_correlation: f32,
        pub correlation_std: f32,
        /// fragments correlating with the median profile above 0.80
        pub num_over_80: f32,
        /// fragments correlating above 0 among the 6 of highest library intensity
        pub num_over_0_rank_0_5: f32,
        pub hyperscore_intensity_observation: f32,
        pub hyperscore_intensity_library: f32,
        pub hyperscore_inverse_mass_error: f32,
        pub rt_observed: f32,
        pub delta_rt: f32,
        pub longest_b_series: f32,
        pub longest_y_series: f32,
        pub naa: f32,
        pub weighted_mass_error: f32,
        pub log10_b_ion_intensity: f32,
        pub log10_y_ion_intensity: f32,
        /// full width at half maximum of the median profile, in retention time
        pub fwhm_rt: f32,
        /// sum of the median of the normalized XICs
        pub median_profile_sum: f32,
        /// number of fragment XICs with signal
        pub num_profiles_filtered: f32,
    }
}

/// The core features of one candidate. `rt` holds the retention time of every cycle.
pub fn calculate_core_features(
    observation: &PeakGroupObservation,
    precursor: &Precursor,
    candidate: &Candidate,
    rt: &Array1<f32>,
) -> CoreFeatures {
    let correlations = &observation.correlations;
    let observation_intensities = observation.observation_intensities.as_slice();
    let matched_mask = &observation.matched_mask;
    let library_intensities = &precursor.fragment_intensity;

    let mean_correlation = if !correlations.is_empty() {
        correlations.iter().sum::<f32>() / correlations.len() as f32
    } else {
        0.0
    };

    let rank_mask_0_5 = create_ranked_mask(library_intensities, 0, 6);
    let count_over = |threshold: f32, mask: Option<&Vec<bool>>| {
        count_values_above(correlations, threshold, mask.map(|m| m.as_slice())) as f32
    };

    let (longest_b_series, longest_y_series) = calculate_longest_ion_series(
        &precursor.fragment_type,
        &precursor.fragment_number,
        matched_mask,
    );

    let (_fragment_mz_observed, fragment_mass_errors) = calculate_fragment_mz_and_errors(
        &observation.dense_mz,
        &observation.dense_xic,
        &precursor.fragment_mz,
    );

    let rt_observed = rt[candidate.cycle_center];

    let log10_ion_intensity = |fragment_type: u8| {
        let intensity = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities,
            matched_mask,
            fragment_type,
        );
        (intensity + LOG_INTENSITY_EPSILON).log10()
    };

    CoreFeatures {
        score: candidate.score,
        mean_correlation,
        median_correlation: calculate_median(correlations),
        correlation_std: calculate_std(correlations),
        num_over_80: count_over(0.80, None),
        num_over_0_rank_0_5: count_over(0.0, Some(&rank_mask_0_5)),
        hyperscore_intensity_observation: calculate_hyperscore(
            &precursor.fragment_type,
            observation_intensities,
            matched_mask,
        ),
        hyperscore_intensity_library: calculate_hyperscore(
            &precursor.fragment_type,
            library_intensities,
            matched_mask,
        ),
        hyperscore_inverse_mass_error: calculate_hyperscore_inverse_mass_error(
            &precursor.fragment_type,
            observation_intensities,
            matched_mask,
            &fragment_mass_errors,
        ),
        rt_observed,
        delta_rt: rt_observed - precursor.rt,
        longest_b_series: longest_b_series as f32,
        longest_y_series: longest_y_series as f32,
        naa: precursor.naa as f32,
        weighted_mass_error: calculate_weighted_mean_absolute_error(
            &fragment_mass_errors,
            library_intensities,
        ),
        log10_b_ion_intensity: log10_ion_intensity(FragmentType::B),
        log10_y_ion_intensity: log10_ion_intensity(FragmentType::Y),
        fwhm_rt: calculate_fwhm_rt(
            &observation.median_profile_filtered,
            candidate.cycle_start,
            rt,
        ),
        median_profile_sum: observation.median_profile.iter().sum::<f32>(),
        num_profiles_filtered: observation.num_profiles_filtered as f32,
    }
}
