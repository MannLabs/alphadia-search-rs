use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::features::{
    CandidateFeature, CandidateFeatureCollection, SetSpecificFeatures,
};
use crate::peak_group_scoring::ion_coverage::{
    calculate_ion_coverage_features, num_ion_classes, IonEvidence, IonNoiseModel,
};
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
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
    ) -> PyResult<CandidateFeatureCollection> {
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
    ) -> PyResult<CandidateFeatureCollection> {
        let start_time = Instant::now();

        // Parallel iteration over candidates to score each one
        let (mut scored_candidates, ion_evidence): (Vec<_>, Vec<Vec<IonEvidence>>) = candidates
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
            .unzip();

        if self.params.feature_set == FeatureSet::Full {
            self.add_ion_fisher(&mut scored_candidates, &ion_evidence);
        }
        drop(ion_evidence);

        let feature_collection =
            CandidateFeatureCollection::from_vec(self.params.feature_set, scored_candidates)?;

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = candidates.len() as f32 / duration.as_secs_f32();
        println!(
            "Scored {} candidates with the {:?} feature set at {:.2} candidates/second",
            candidates.len(),
            self.params.feature_set,
            candidates_per_second
        );

        Ok(feature_collection)
    }

    /// The ion-class noise model needs all candidates: it is fitted on those that are not the best of their
    /// precursor, targets and decoys alike, and then scores every candidate's fragments.
    fn add_ion_fisher(
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
                if let SetSpecificFeatures::Full { ion_coverage, .. } = &mut feature.set_specific {
                    ion_coverage.ion_fisher_lib = noise_model.fisher(evidence);
                }
            });
    }

    /// Scores one candidate with the blocks of the feature set. The ion evidence feeds the noise model of
    /// `ion_fisher_lib` and is empty if the feature set has no ion-coverage block.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
    ) -> Option<(CandidateFeature, Vec<IonEvidence>)> {
        let observation =
            PeakGroupObservation::new(dia_data, precursor, candidate, self.params.mass_tolerance);

        let core =
            calculate_core_features(&observation, precursor, candidate, &dia_data.rt_index().rt);

        let (set_specific, ion_evidence) = match self.params.feature_set {
            FeatureSet::Calibration => (SetSpecificFeatures::Calibration, Vec::new()),
            FeatureSet::Full => {
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
                let library_agreement = calculate_library_agreement_features(
                    &observation.dense_xic,
                    &observation.median_profile_filtered,
                    &precursor.fragment_intensity,
                    &observation.observation_intensities,
                    &observation.correlations,
                    &precursor.fragment_type,
                );
                let full = SetSpecificFeatures::Full {
                    ion_coverage,
                    library_agreement,
                };
                (full, ion_evidence)
            }
        };

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            set_specific,
        };
        Some((feature, ion_evidence))
    }
}
