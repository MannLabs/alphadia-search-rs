//! Features of the calibration feature set only: fragment counts by correlation threshold and library intensity
//! rank, and summaries of the fragment XICs.
//!
//! Permuting each of them in offline replays of alphaDIA's final FDR round moved the real identifications by less
//! than 1 %, and a refit without all of them kept the real identifications level (MannLabs/alphadia#908), so the
//! full feature set leaves them out.

use crate::candidate::Candidate;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::utils::calculate_correlation_safe;
use crate::precursor::Precursor;
use crate::utils::{count_values_above, create_ranked_mask};

feature_block! {
    pub struct CalibrationOnlyFeatures {
        /// observed vs library intensity correlation over all fragments
        pub intensity_correlation: f32,
        pub num_fragments: f32,
        pub num_scans: f32,
        /// fragments correlating with the median profile above 0.95, 0.90, 0.50 and 0
        pub num_over_95: f32,
        pub num_over_90: f32,
        pub num_over_50: f32,
        pub num_over_0: f32,
        /// the same above 0 and 0.50, among the fragments of a library intensity rank range
        pub num_over_0_rank_6_11: f32,
        pub num_over_0_rank_12_17: f32,
        pub num_over_0_rank_18_23: f32,
        pub num_over_50_rank_0_5: f32,
        pub num_over_50_rank_6_11: f32,
        pub num_over_50_rank_12_17: f32,
        pub num_over_50_rank_18_23: f32,
        /// sum of the median of the normalized XICs with signal
        pub median_profile_sum_filtered: f32,
        /// number of fragment XICs
        pub num_profiles: f32,
    }
}

/// The calibration-only features of one candidate.
pub fn calculate_calibration_only_features(
    observation: &PeakGroupObservation,
    precursor: &Precursor,
    candidate: &Candidate,
) -> CalibrationOnlyFeatures {
    let correlations = &observation.correlations;
    let library_intensities = &precursor.fragment_intensity;

    let rank_mask_0_5 = create_ranked_mask(library_intensities, 0, 6);
    let rank_mask_6_11 = create_ranked_mask(library_intensities, 6, 12);
    let rank_mask_12_17 = create_ranked_mask(library_intensities, 12, 18);
    let rank_mask_18_23 = create_ranked_mask(library_intensities, 18, 24);
    let count_over = |threshold: f32, mask: Option<&Vec<bool>>| {
        count_values_above(correlations, threshold, mask.map(|m| m.as_slice())) as f32
    };

    CalibrationOnlyFeatures {
        intensity_correlation: calculate_correlation_safe(
            &observation.observation_intensities,
            library_intensities,
        ),
        num_fragments: precursor.fragment_mz.len() as f32,
        num_scans: (candidate.cycle_stop - candidate.cycle_start) as f32,
        num_over_95: count_over(0.95, None),
        num_over_90: count_over(0.90, None),
        num_over_50: count_over(0.50, None),
        num_over_0: count_over(0.0, None),
        num_over_0_rank_6_11: count_over(0.0, Some(&rank_mask_6_11)),
        num_over_0_rank_12_17: count_over(0.0, Some(&rank_mask_12_17)),
        num_over_0_rank_18_23: count_over(0.0, Some(&rank_mask_18_23)),
        num_over_50_rank_0_5: count_over(0.50, Some(&rank_mask_0_5)),
        num_over_50_rank_6_11: count_over(0.50, Some(&rank_mask_6_11)),
        num_over_50_rank_12_17: count_over(0.50, Some(&rank_mask_12_17)),
        num_over_50_rank_18_23: count_over(0.50, Some(&rank_mask_18_23)),
        median_profile_sum_filtered: observation.median_profile_filtered.iter().sum::<f32>(),
        num_profiles: observation.dense_xic.nrows() as f32,
    }
}
