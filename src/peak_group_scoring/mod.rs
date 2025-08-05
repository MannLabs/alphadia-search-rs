use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{
    Candidate, CandidateCollection, CandidateFeature, CandidateFeatureCollection,
};
use crate::dense_xic_observation::DenseXICObservation;
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, correlation_axis_0, median_axis_0, normalize_profiles,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
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

    pub fn score_next_gen(
        &self,
        dia_data: &DIADataNextGen,
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
        let n = std::cmp::min(100_000_000, candidates.len());

        let scored_candidates: Vec<CandidateFeature> = candidates
            .par_iter()
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

        // Create collection from Vec
        let feature_collection = CandidateFeatureCollection::from_vec(scored_candidates);

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = n as f32 / duration.as_secs_f32();
        println!("Scored {n} candidates at {candidates_per_second:.2} candidates/second");

        feature_collection
    }

    /// Generic candidate scoring function that works with any type implementing DIADataTrait
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
    ) -> CandidateFeature {
        // Scoring implementation for individual candidate will be added here
        // For now, return the original score

        // Apply fragment filtering based on scoring parameters
        let (filtered_fragment_mz, filtered_fragment_intensity) = precursor.get_fragments_filtered(
            true, // Always filter non-zero intensities for scoring
            self.params.top_k_fragments,
        );

        let cycle_start_idx = candidate.cycle_start;
        let cycle_stop_idx = candidate.cycle_stop;
        let mass_tolerance = self.params.mass_tolerance;

        // Create dense XIC observation using the new struct
        let dense_xic_obs = DenseXICObservation::new(
            dia_data,
            precursor.mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            &filtered_fragment_mz,
        );

        // Normalize the profiles before calculating median
        let normalized_xic = normalize_profiles(&dense_xic_obs.dense_xic, 1);
        let median_profile = median_axis_0(&normalized_xic);

        // Calculate correlations of each profile with the median profile
        let correlations = correlation_axis_0(&median_profile, &normalized_xic);

        let observation_intensities = dense_xic_obs.dense_xic.sum_axis(Axis(1));

        let intensity_correlations = calculate_correlation_safe(
            observation_intensities.as_slice().unwrap(),
            &filtered_fragment_intensity,
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
        let num_fragments = filtered_fragment_mz.len();
        let num_scans = cycle_stop_idx - cycle_start_idx;

        // Create and return candidate feature
        CandidateFeature::new(
            candidate.precursor_idx,
            candidate.rank,
            candidate.score,
            mean_correlation,
            median_correlation,
            correlation_std,
            intensity_correlation,
            num_fragments,
            num_scans,
            num_over_95,
            num_over_90,
            num_over_80,
            num_over_50,
        )
    }
}
