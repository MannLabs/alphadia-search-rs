use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::features::{CandidateFeature, CandidateFeatureCollection};
use crate::peak_group_scoring::ion_coverage::{
    calculate_ion_coverage_features, num_ion_classes, IonEvidence, IonNoiseModel,
};
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::window_share::{
    calculate_window_share_features, window_share_context, WindowShareEntry,
    WINDOW_SHARE_CYCLE_RADIUS,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;

pub mod core;
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

/// One scored candidate with what the blocks that need all candidates use later: the ion evidence of the
/// ion-coverage block (empty without it) and the window-share entry of the window-share block (`None` without it).
struct ScoredCandidate {
    feature: CandidateFeature,
    ion_evidence: Vec<IonEvidence>,
    window_share: Option<WindowShareEntry>,
}

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

        // MS2 intensity of every isolation window per cycle, for the window share
        let window_intensities: Vec<Vec<f32>> = if self
            .params
            .feature_set
            .has_block(FeatureBlockId::WindowShare)
        {
            dia_data
                .quadrupole_observations()
                .iter()
                .map(|obs| obs.cycle_intensity_sums())
                .collect()
        } else {
            Vec::new()
        };

        // Parallel iteration over candidates to score each one
        let scored: Vec<ScoredCandidate> = candidates
            .par_iter()
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true, // Always filter non-zero intensities for scoring
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => self.score_candidate_generic(
                        dia_data,
                        &precursor,
                        candidate,
                        &window_intensities,
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
            .collect();
        drop(window_intensities);

        let mut scored_candidates = Vec::with_capacity(scored.len());
        let mut ion_evidence = Vec::with_capacity(scored.len());
        let mut window_shares = Vec::with_capacity(scored.len());
        for candidate in scored {
            scored_candidates.push(candidate.feature);
            ion_evidence.push(candidate.ion_evidence);
            window_shares.push(candidate.window_share);
        }

        if self
            .params
            .feature_set
            .has_block(FeatureBlockId::IonCoverage)
        {
            self.set_ion_fisher_lib(&mut scored_candidates, &ion_evidence);
        }
        drop(ion_evidence);

        if self
            .params
            .feature_set
            .has_block(FeatureBlockId::WindowShare)
        {
            self.set_window_share_context(&mut scored_candidates, &window_shares);
        }
        drop(window_shares);

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

    /// The ion-class noise model needs all candidates: it is fitted on those that are not the best of their
    /// precursor, targets and decoys alike, and then scores every candidate's fragments.
    fn set_ion_fisher_lib(
        &self,
        scored_candidates: &mut [CandidateFeature],
        ion_evidence: &[Vec<IonEvidence>],
    ) {
        let noise_model = IonNoiseModel::fit(
            scored_candidates
                .par_iter()
                .zip(ion_evidence.par_iter())
                .filter(|(feature, _)| feature.rank >= 1)
                .map(|(_, evidence)| evidence.as_slice()),
            num_ion_classes(self.params.top_k_fragments),
        );
        scored_candidates
            .par_iter_mut()
            .zip(ion_evidence.par_iter())
            .for_each(|(feature, evidence)| {
                if let Some(ion_coverage) = &mut feature.ion_coverage {
                    ion_coverage.ion_fisher_lib = noise_model.fisher(evidence);
                }
            });
    }

    /// The rank and z-score of the window share compare each candidate with the candidates of its isolation window
    /// whose apex lies within WINDOW_SHARE_CYCLE_RADIUS cycles, so they need all candidates.
    fn set_window_share_context(
        &self,
        scored_candidates: &mut [CandidateFeature],
        window_shares: &[Option<WindowShareEntry>],
    ) {
        let (positions, entries): (Vec<usize>, Vec<WindowShareEntry>) = window_shares
            .iter()
            .enumerate()
            .filter_map(|(position, entry)| entry.map(|entry| (position, entry)))
            .unzip();
        let context = window_share_context(&entries, WINDOW_SHARE_CYCLE_RADIUS);
        for (position, (rank, zscore)) in positions.into_iter().zip(context) {
            if let Some(window_share) = &mut scored_candidates[position].window_share {
                window_share.apex_window_signal_frac_rank = rank;
                window_share.apex_window_signal_frac_zscore = zscore;
            }
        }
    }

    /// Scores one candidate with the blocks of the feature set. The ion evidence feeds the noise model of
    /// `ion_fisher_lib` and is empty if the feature set has no ion-coverage block. The window-share entry feeds the
    /// window context and is `None` if the feature set has no window-share block. `window_intensities` holds the
    /// summed MS2 intensity per isolation window and cycle.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        window_intensities: &[Vec<f32>],
    ) -> Option<ScoredCandidate> {
        let observation =
            PeakGroupObservation::new(dia_data, precursor, candidate, self.params.mass_tolerance);

        let core =
            calculate_core_features(&observation, precursor, candidate, &dia_data.rt_index().rt);

        let feature_set = self.params.feature_set;

        let (ion_coverage, ion_evidence) = if feature_set.has_block(FeatureBlockId::IonCoverage) {
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
            (Some(ion_coverage), ion_evidence)
        } else {
            (None, Vec::new())
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

        let (window_share, window_share_entry) =
            if feature_set.has_block(FeatureBlockId::WindowShare) {
                let (window_share, entry) = calculate_window_share_features(
                    &observation.dense_xic,
                    &observation.median_profile_filtered,
                    candidate.cycle_start,
                    candidate.cycle_center,
                    &observation.contributing_obs_indices,
                    window_intensities,
                );
                (Some(window_share), Some(entry))
            } else {
                (None, None)
            };

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            ion_coverage,
            library_agreement,
            window_share,
        };
        Some(ScoredCandidate {
            feature,
            ion_evidence,
            window_share: window_share_entry,
        })
    }
}
