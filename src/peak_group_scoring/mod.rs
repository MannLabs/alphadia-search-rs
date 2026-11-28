use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::cross_candidate::{
    apply_cross_candidate_features, CandidateContext,
};
use crate::peak_group_scoring::detection::{
    calculate_detection_features, scores_detection, DetectionFeatures, DetectionModel,
};
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::features::{
    CandidateFeature, CandidateFeatureCollection, FeatureColumns,
};
use crate::peak_group_scoring::fragment_isotope::{
    calculate_fragment_isotope_features, isotope_eligible_fragments, isotope_shifted_ions,
};
use crate::peak_group_scoring::ion_coverage::calculate_ion_coverage_features;
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::ms1::{
    calculate_ms1_apex_calibration, calculate_ms1_features, Ms1ApexFeatures,
};
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::window_share::{cycle_intensity_sums, window_share_entry};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::window_groups::WindowGroups;
use crate::SpecLibFlat;

pub mod core;
pub mod cross_candidate;
pub mod detection;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod fragment_isotope;
pub mod ion_coverage;
pub mod library_agreement;
pub mod ms1;
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

    /// The apex isotope statistics of each candidate for an MS1 mass calibration: the columns of
    /// `ms1::Ms1ApexFeatures` (`ms1_iso_cosine`, `ms1_n_isotopes`, `ms1_log_m0_apex`, `ms1_ppm_signed`), with
    /// `precursor_idx` and `rank`, extracted at `ms1_mass_tolerance` around the MS1 m/z that `SpecLibFlat::set_isotopes`
    /// set. All 0 for a candidate without isotopes or a run without MS1 scans.
    pub fn score_ms1_calibration(
        &self,
        py: Python,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> PyResult<Py<PyAny>> {
        let ms1_observations = if lib.has_isotopes() {
            dia_data.ms1_observations()
        } else {
            Vec::new()
        };
        let scored: Vec<(usize, usize, Ms1ApexFeatures)> = candidates
            .par_iter()
            .filter_map(|candidate| {
                let precursor = lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true,
                    true,
                    false, // Scoring uses the short ions
                    self.params.top_k_fragments,
                )?;
                // the MS1 mass calibration reads the monoisotopic fragment XICs only, as the calibration rounds do
                let observation = PeakGroupObservation::new(
                    dia_data,
                    &precursor,
                    candidate,
                    self.params.mass_tolerance,
                    None,
                );
                let apex = calculate_ms1_apex_calibration(
                    dia_data,
                    lib,
                    candidate,
                    &observation,
                    &ms1_observations,
                    self.params.ms1_mass_tolerance,
                );
                Some((candidate.precursor_idx, candidate.rank, apex))
            })
            .collect();

        let mut columns = FeatureColumns::new(Ms1ApexFeatures::NAMES.to_vec(), scored.len());
        for (i, (precursor_idx, rank, apex)) in scored.iter().enumerate() {
            columns.set_row(i, *precursor_idx, *rank, apex.values());
        }
        columns.into_dict(py)
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

        // The MS1 survey scans of the MS1 block; none if the library has no isotopes, so that the block costs nothing
        // then and every candidate gets the missing values
        let ms1_observations: Vec<usize> =
            if self.params.feature_set.has_block(FeatureBlockId::Ms1) && lib.has_isotopes() {
                dia_data.ms1_observations()
            } else {
                Vec::new()
            };

        // The run-wide inputs of the detection block, from all candidates alike (no labels)
        let detection_model = scores_detection(&self.params)
            .then(|| DetectionModel::fit(dia_data, lib, candidates, &self.params));

        let sums_isotopes = self.params.feature_set.sums_fragment_isotopes();

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
                    Some(precursor) => {
                        let isotope_shifted = if sums_isotopes {
                            lib.precursor_array_index(candidate.precursor_idx)
                                .map(|index| {
                                    isotope_shifted_ions(lib, index, self.params.mass_tolerance)
                                })
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        self.score_candidate_generic(
                            dia_data,
                            lib,
                            &precursor,
                            candidate,
                            cycle_intensities,
                            window_groups,
                            &ms1_observations,
                            detection_model.as_ref(),
                            &isotope_shifted,
                        )
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
    /// `ms1_observations` are the MS1 survey scans the MS1 block reads; without them, or without library isotopes for
    /// the precursor, the MS1 block takes its missing values. `detection_model` holds the run-wide inputs of the
    /// detection block; without it (`detection_features` off) the detection block takes its missing values.
    #[allow(clippy::too_many_arguments)]
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        precursor: &Precursor,
        candidate: &Candidate,
        cycle_intensities: &[Vec<f32>],
        window_groups: &WindowGroups,
        ms1_observations: &[usize],
        detection_model: Option<&DetectionModel>,
        isotope_shifted: &[u64],
    ) -> Option<(CandidateFeature, CandidateContext)> {
        let feature_set = self.params.feature_set;
        let isotope_eligible = feature_set.sums_fragment_isotopes().then(|| {
            isotope_eligible_fragments(precursor, self.params.mass_tolerance, isotope_shifted)
        });
        let observation = PeakGroupObservation::new(
            dia_data,
            precursor,
            candidate,
            self.params.mass_tolerance,
            isotope_eligible.as_deref(),
        );

        let core =
            calculate_core_features(&observation, precursor, candidate, &dia_data.rt_index().rt);

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

        let fragment_isotope = isotope_eligible
            .as_deref()
            .filter(|_| feature_set.has_block(FeatureBlockId::FragmentIsotope))
            .and_then(|eligible| {
                observation.isotope_xics.as_ref().map(|isotopes| {
                    calculate_fragment_isotope_features(
                        isotopes.monoisotopic.view(),
                        isotopes.mplus1.view(),
                        isotopes.mminus1.view(),
                        &precursor.fragment_mz,
                        &precursor.fragment_charge,
                        &precursor.fragment_intensity,
                        &isotopes.monoisotopic_intensities,
                        &observation.correlations,
                        eligible,
                    )
                })
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

        let ms1 = feature_set.has_block(FeatureBlockId::Ms1).then(|| {
            calculate_ms1_features(
                dia_data,
                lib,
                candidate,
                &observation,
                core.fwhm_rt,
                ms1_observations,
                self.params.ms1_mass_tolerance,
            )
        });

        let detection =
            feature_set
                .has_block(FeatureBlockId::Detection)
                .then(|| match detection_model {
                    Some(model) => calculate_detection_features(
                        dia_data,
                        precursor,
                        candidate,
                        &observation,
                        model,
                        &self.params,
                    ),
                    None => DetectionFeatures::missing(),
                });

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            ion_coverage,
            library_agreement,
            // set with the whole block by apply_cross_candidate_features
            window_share: None,
            ms1,
            detection,
            fragment_isotope,
        };
        Some((feature, context))
    }
}
