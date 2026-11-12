//! What the feature blocks read of one candidate: its fragment XICs and their agreement with the median profile.

use numpy::ndarray::{Array2, Axis};

use crate::candidate::Candidate;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::peak_group_scoring::utils::{
    correlation_axis_0, filter_non_zero, median_axis_0, normalize_profiles,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, MobilityFilter};

pub struct PeakGroupObservation {
    /// fragment XICs: fragments x cycles
    pub dense_xic: Array2<f32>,
    /// observed m/z of the fragment XICs: fragments x cycles
    pub dense_mz: Array2<f32>,
    /// median of the normalized XICs
    pub median_profile: Vec<f32>,
    /// median of the normalized XICs with signal
    pub median_profile_filtered: Vec<f32>,
    /// number of fragment XICs with signal
    pub num_profiles_filtered: usize,
    /// correlation of each normalized XIC with `median_profile_filtered`
    pub correlations: Vec<f32>,
    /// observed intensity per fragment, summed over the candidate's cycles
    pub observation_intensities: Vec<f32>,
    /// observed intensity > 0 per fragment
    pub matched_mask: Vec<bool>,
}

impl PeakGroupObservation {
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        mass_tolerance: f32,
        mobility: MobilityFilter,
    ) -> Self {
        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            mobility,
            candidate.cycle_start,
            candidate.cycle_stop,
            mass_tolerance,
            &precursor.fragment_mz,
        );

        let normalized_xic = normalize_profiles(&dense_xic_mz_obs.dense_xic, 1);
        let filtered_xic = filter_non_zero(&normalized_xic);

        let median_profile = median_axis_0(&normalized_xic);
        let median_profile_filtered = median_axis_0(&filtered_xic);
        let correlations = correlation_axis_0(&median_profile_filtered, &normalized_xic);

        let observation_intensities = dense_xic_mz_obs.dense_xic.sum_axis(Axis(1)).to_vec();
        let matched_mask = observation_intensities.iter().map(|&x| x > 0.0).collect();

        Self {
            dense_xic: dense_xic_mz_obs.dense_xic,
            dense_mz: dense_xic_mz_obs.dense_mz,
            median_profile,
            median_profile_filtered,
            num_profiles_filtered: filtered_xic.shape()[0],
            correlations,
            observation_intensities,
            matched_mask,
        }
    }
}
