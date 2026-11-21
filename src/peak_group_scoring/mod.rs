use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::features::{CandidateFeature, CandidateFeatureCollection};
use crate::peak_group_scoring::ion_coverage::{
    calculate_ion_coverage_features, num_ion_classes, IonEvidence, IonNoiseModel,
};
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::symbolic::{
    calculate_symbolic_features_from_dense, SYMBOLIC_HALF_WINDOW,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;
use numpy::ndarray::s;

pub mod core;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod ion_coverage;
pub mod library_agreement;
pub mod observation;
pub mod parameters;
pub mod symbolic;
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
    ) -> CandidateFeatureCollection
    where
        T::QuadrupoleObservation: Sync,
    {
        let start_time = Instant::now();

        let crowding_tables: Vec<Option<Vec<u8>>> =
            if self.params.feature_set.has_block(FeatureBlockId::Symbolic) {
                dia_data
                    .quadrupole_observations()
                    .par_iter()
                    .map(|obs| obs.build_1da_crowding_table(dia_data.mz_index()))
                    .collect()
            } else {
                Vec::new()
            };

        // Parallel iteration over candidates to score each one
        let (mut scored_candidates, ion_evidence): (Vec<_>, Vec<Vec<IonEvidence>>) = candidates
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
                        &crowding_tables,
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

        if self
            .params
            .feature_set
            .has_block(FeatureBlockId::IonCoverage)
        {
            self.set_ion_fisher_lib(&mut scored_candidates, &ion_evidence);
        }
        drop(ion_evidence);

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

    /// Scores one candidate with the blocks of the feature set. The ion evidence feeds the noise model of
    /// `ion_fisher_lib` and is empty if the feature set has no ion-coverage block.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        crowding_tables: &[Option<Vec<u8>>],
    ) -> Option<(CandidateFeature, Vec<IonEvidence>)> {
        let feature_set = self.params.feature_set;
        let has_symbolic = feature_set.has_block(FeatureBlockId::Symbolic);

        let (observation, symbolic) = if has_symbolic {
            let n_cycles = dia_data.rt_index().rt.len();
            let sym_lo = candidate
                .cycle_center
                .saturating_sub(SYMBOLIC_HALF_WINDOW)
                .min(n_cycles);
            let sym_hi = (candidate.cycle_center + SYMBOLIC_HALF_WINDOW + 1).min(n_cycles);
            let cand_lo = candidate.cycle_start;
            let cand_hi = candidate.cycle_stop.max(cand_lo);
            let union_lo = cand_lo.min(sym_lo);
            let union_hi = cand_hi.max(sym_hi);

            let union_dense = DenseXICMZObservation::new(
                dia_data,
                precursor.mz,
                union_lo,
                union_hi,
                self.params.mass_tolerance,
                &precursor.fragment_mz,
            );
            let rel_lo = cand_lo - union_lo;
            let rel_hi = cand_hi - union_lo;
            let obs = PeakGroupObservation::from_dense(
                union_dense
                    .dense_xic
                    .slice(s![.., rel_lo..rel_hi])
                    .to_owned(),
                union_dense
                    .dense_mz
                    .slice(s![.., rel_lo..rel_hi])
                    .to_owned(),
            );
            let sym = calculate_symbolic_features_from_dense(
                dia_data,
                precursor,
                candidate,
                self.params.mass_tolerance,
                &union_dense.dense_xic,
                &union_dense.dense_mz,
                union_lo,
                &union_dense.contributing_obs_indices,
                crowding_tables,
            );
            (obs, Some(sym))
        } else {
            (
                PeakGroupObservation::new(
                    dia_data,
                    precursor,
                    candidate,
                    self.params.mass_tolerance,
                ),
                None,
            )
        };

        let core =
            calculate_core_features(&observation, precursor, candidate, &dia_data.rt_index().rt);

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

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            ion_coverage,
            library_agreement,
            symbolic,
        };
        Some((feature, ion_evidence))
    }
}
