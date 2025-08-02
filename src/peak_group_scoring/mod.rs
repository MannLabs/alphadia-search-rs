use pyo3::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, correlation_axis_0, median_axis_0, normalize_profiles,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;
use numpy::ndarray::{Array2, Axis};

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

    pub fn score(&self, dia_data: &DIAData, lib: &SpecLibFlat, candidates: &CandidateCollection) {
        self.score_generic(dia_data, lib, candidates)
    }

    pub fn score_next_gen(
        &self,
        dia_data: &DIADataNextGen,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) {
        self.score_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupScoring {
    /// Generic scoring function that works with any type implementing DIADataTrait
    fn score_generic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) {
        let start_time = Instant::now();

        // Sequential iteration over candidates to score each one
        let n = std::cmp::min(100, candidates.len());
        let _scored_candidates: Vec<_> = candidates
            .iter()
            .take(n)
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx(candidate.precursor_idx) {
                    Some(precursor) => {
                        Some(self.score_candidate_generic(dia_data, &precursor, candidate))
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

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = n as f32 / duration.as_secs_f32();
        println!("Scored {n} candidates at {candidates_per_second:.2} candidates/second");
    }

    /// Generic candidate scoring function that works with any type implementing DIADataTrait
    fn score_candidate_generic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
    ) -> f32 {
        // Scoring implementation for individual candidate will be added here
        // For now, return the original score
        println!("Scoring candidate: {candidate:?}");

        // Apply fragment filtering based on scoring parameters
        let (filtered_fragment_mz, filtered_fragment_intensity) = precursor.get_fragments_filtered(
            true, // Always filter non-zero intensities for scoring
            self.params.top_k_fragments,
        );

        let cycle_start_idx = candidate.cycle_start;
        let cycle_stop_idx = candidate.cycle_stop;
        let mass_tolerance = self.params.mass_tolerance;

        let mut dense_xic: Array2<f32> =
            Array2::zeros((filtered_fragment_mz.len(), cycle_stop_idx - cycle_start_idx));

        // For now, use all observations - this will be refined later
        let valid_obs_idxs = dia_data.get_valid_observations(precursor.mz);

        for &obs_idx in &valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];

            for (f_idx, f_mz) in filtered_fragment_mz.iter().enumerate() {
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

        println!("Number of fragments: {:?}", filtered_fragment_mz.len());
        println!("Number of scans: {:?}", cycle_stop_idx - cycle_start_idx);

        // Normalize the profiles before calculating median
        let normalized_xic = normalize_profiles(&dense_xic, 1);
        let median_profile = median_axis_0(&normalized_xic);

        // Calculate correlations of each profile with the median profile
        let correlations = correlation_axis_0(&median_profile, &normalized_xic);

        let observation_intensities = dense_xic.sum_axis(Axis(1));

        let intensity_correlations = calculate_correlation_safe(
            observation_intensities.as_slice().unwrap(),
            &filtered_fragment_intensity,
        );

        println!("Correlations: {correlations:?}");
        println!("Observation intensities: {observation_intensities:?}");
        println!("Library intensities: {filtered_fragment_intensity:?}");
        println!("Intensity correlations: {intensity_correlations:?}");

        // For now, return the original score
        candidate.score
    }
}
