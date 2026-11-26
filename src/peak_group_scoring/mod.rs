use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::cross_candidate::{
    apply_cross_candidate_features, CandidateContext,
};
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::features::{CandidateFeature, CandidateFeatureCollection};
use crate::peak_group_scoring::ion_coverage::calculate_ion_coverage_features;
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::window_share::{cycle_intensity_sums, window_share_entry};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::window_groups::WindowGroups;
use crate::SpecLibFlat;

pub mod core;
pub mod cross_candidate;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod ion_coverage;
pub mod library_agreement;
pub mod observation;
pub mod parameters;
pub mod tests;
pub mod utils;
pub mod window_share;
pub use feature_set::FeatureSet;
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
        // whole-run input of the window-share block: MS2 intensity per quadrupole observation and cycle, and the
        // window group of every precursor m/z
        let (cycle_intensities, window_groups) = if self
            .params
            .feature_set
            .has_block(FeatureBlockId::WindowShare)
        {
            let observations = dia_data.quadrupole_observations();
            let windows: Vec<[f32; 2]> = observations.iter().map(|o| o.isolation_window).collect();
            (
                observations.iter().map(cycle_intensity_sums).collect(),
                WindowGroups::new(&windows),
            )
        } else {
            (Vec::new(), WindowGroups::new(&[]))
        };
        self.score_generic(
            dia_data,
            lib,
            candidates,
            &cycle_intensities,
            &window_groups,
        )
    }
}

impl PeakGroupScoring {
    /// Generic scoring function that works with any type implementing DIADataTrait
    fn score_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
        cycle_intensities: &[Vec<f32>],
        window_groups: &WindowGroups,
    ) -> CandidateFeatureCollection {
        let start_time = Instant::now();

        // Parallel iteration over candidates to score each one
        let (mut scored_candidates, contexts): (Vec<_>, Vec<_>) = candidates
            .par_iter()
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true,  // Always filter non-zero intensities for scoring
                    true,  // Filter Y1 ions by default
                    false, // Scoring uses the short ions
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => self.score_candidate_generic(
                        dia_data,
                        &precursor,
                        candidate,
                        cycle_intensities,
                        window_groups,
                    ),
                    None => {
                        eprintln!(
                            "Warning: Candidate precursor_idx {} not found in library. Skipping.",
                            candidate.precursor_idx
                        );
                        None
                    }
                }
            })
            .unzip();

        apply_cross_candidate_features(
            self.params.feature_set,
            self.params.top_k_fragments,
            &mut scored_candidates,
            contexts,
        );

        let feature_collection =
            CandidateFeatureCollection::from_vec(self.params.feature_set, scored_candidates);

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = candidates.len() as f32 / duration.as_secs_f32();
        println!(
            "Scored {} candidates with the {:?} feature set at {:.2} candidates/second",
            candidates.len(),
            self.params.feature_set,
            candidates_per_second
        );

        feature_collection
    }

    /// Scores one candidate with the blocks of the feature set. The cross-candidate fields stay 0; the returned
    /// context holds what `apply_cross_candidate_features` needs to fill them. `cycle_intensities` (summed MS2 intensity
    /// per quadrupole observation and cycle) and `window_groups` are empty without the window-share block.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        cycle_intensities: &[Vec<f32>],
        window_groups: &WindowGroups,
    ) -> Option<(CandidateFeature, CandidateContext)> {
        let observation =
            PeakGroupObservation::new(dia_data, precursor, candidate, self.params.mass_tolerance);

        let core =
            calculate_core_features(&observation, precursor, candidate, &dia_data.rt_index().rt);

        let feature_set = self.params.feature_set;

        let mut context = CandidateContext::default();

        let ion_coverage = if feature_set.has_block(FeatureBlockId::IonCoverage) {
            let (ion_coverage, ion_evidence) = calculate_ion_coverage_features(
                &precursor.fragment_intensity,
                &observation.observation_intensities,
                &observation.correlations,
                &precursor.fragment_type,
                &precursor.fragment_number,
                &precursor.fragment_charge,
                precursor.naa,
                self.params.top_k_fragments,
            );
            context.ion_evidence = ion_evidence;
            Some(ion_coverage)
        } else {
            None
        };

        let library_agreement = feature_set
            .has_block(FeatureBlockId::LibraryAgreement)
            .then(|| {
                calculate_library_agreement_features(
                    &observation.dense_xic,
                    &observation.median_profile_filtered,
                    &precursor.fragment_intensity,
                    &observation.observation_intensities,
                    &observation.correlations,
                    &precursor.fragment_type,
                )
            });

        if feature_set.has_block(FeatureBlockId::WindowShare) {
            context.window_share = Some(window_share_entry(
                &observation.dense_xic,
                &observation.median_profile_filtered,
                candidate.cycle_start,
                candidate.cycle_center,
                &observation.quadrupole_observation_indices,
                cycle_intensities,
                window_groups.group_of(precursor.mz),
            ));
        }

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            ion_coverage,
            library_agreement,
            // set with the whole block by apply_cross_candidate_features
            window_share: None,
        };
        Some((feature, context))
    }
}
