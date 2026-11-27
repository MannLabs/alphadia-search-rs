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
use crate::peak_group_scoring::symbolic::{
    calculate_symbolic_features, RunScales, SymbolicInputs, CORE_SIGMAS, MAX_FRAGMENTS,
    NUM_CONTEXT_TRACES, WINDOW_LEN,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;
use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::IntoPyArray;
use pyo3::types::PyDict;

pub mod core;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod ion_coverage;
pub mod library_agreement;
pub mod observation;
pub mod parameters;
pub mod symbolic;
#[cfg(test)]
mod symbolic_fixture;
pub mod symbolic_weights;
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

    /// The raw inputs of the symbolic features for every candidate and the run scales estimated from all of them,
    /// as a dict of arrays. Rows of candidates whose precursor is not in the library are zero.
    pub fn export_symbolic_inputs(
        &self,
        py: Python,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> PyResult<Py<PyAny>> {
        let scales = self.estimate_run_scales(dia_data, lib, candidates);
        let tolerance = CORE_SIGMAS * scales.sigma_mz;
        let inputs: Vec<Option<SymbolicInputs>> = candidates
            .par_iter()
            .map(|candidate| {
                self.scoring_precursor(lib, candidate).map(|precursor| {
                    SymbolicInputs::extract(dia_data, &precursor, candidate, tolerance)
                })
            })
            .collect();

        let n = inputs.len();
        let mut xic = Array3::<f32>::zeros((n, MAX_FRAGMENTS, WINDOW_LEN));
        let mut ppm_error = Array3::<f32>::zeros((n, MAX_FRAGMENTS, WINDOW_LEN));
        let mut context = Array4::<f32>::zeros((n, MAX_FRAGMENTS, NUM_CONTEXT_TRACES, WINDOW_LEN));
        let mut fragment_meta = Array3::<f32>::zeros((n, MAX_FRAGMENTS, 4));
        let mut candidate_meta = Array2::<f32>::zeros((n, 4));
        let mut num_fragments = Array1::<u64>::zeros(n);
        for (i, inputs) in inputs.iter().enumerate() {
            let Some(inputs) = inputs else { continue };
            num_fragments[i] = inputs.num_fragments as u64;
            for f in 0..inputs.num_fragments {
                for t in 0..WINDOW_LEN {
                    xic[[i, f, t]] = inputs.xic[f][t];
                    ppm_error[[i, f, t]] = inputs.ppm_error[f][t];
                    for c in 0..NUM_CONTEXT_TRACES {
                        context[[i, f, c, t]] = inputs.context[f][c][t];
                    }
                }
                fragment_meta[[i, f, 0]] = inputs.library_intensity[f];
                fragment_meta[[i, f, 1]] = inputs.is_y[f];
                fragment_meta[[i, f, 2]] = inputs.charge[f];
                fragment_meta[[i, f, 3]] = inputs.number[f];
            }
            candidate_meta[[i, 0]] = inputs.delta_rt;
            candidate_meta[[i, 1]] = inputs.naa;
            candidate_meta[[i, 2]] = inputs.cycle_start_offset;
            candidate_meta[[i, 3]] = inputs.cycle_stop_offset;
        }

        let dict = PyDict::new(py);
        dict.set_item("xic", xic.into_pyarray(py))?;
        dict.set_item("ppm_error", ppm_error.into_pyarray(py))?;
        dict.set_item("context", context.into_pyarray(py))?;
        dict.set_item("fragment_meta", fragment_meta.into_pyarray(py))?;
        dict.set_item("candidate_meta", candidate_meta.into_pyarray(py))?;
        dict.set_item("num_fragments", num_fragments.into_pyarray(py))?;
        dict.set_item("median_peak", scales.median_peak)?;
        dict.set_item("sigma_mz", scales.sigma_mz)?;
        dict.set_item("sigma_rt", scales.sigma_rt)?;
        Ok(dict.into())
    }

    /// `(median_peak, sigma_mz, sigma_rt)` of the symbolic features, estimated from all candidates of a run.
    pub fn symbolic_run_scales(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> (f32, f32, f32) {
        let scales = self.estimate_run_scales(dia_data, lib, candidates);
        (scales.median_peak, scales.sigma_mz, scales.sigma_rt)
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

        let run_scales = self
            .params
            .feature_set
            .has_block(FeatureBlockId::Symbolic)
            .then(|| self.estimate_run_scales(dia_data, lib, candidates));

        // Parallel iteration over candidates to score each one
        let (mut scored_candidates, ion_evidence): (Vec<_>, Vec<Vec<IonEvidence>>) = candidates
            .par_iter()
            .filter_map(|candidate| match self.scoring_precursor(lib, candidate) {
                Some(precursor) => self.score_candidate_generic(
                    dia_data,
                    &precursor,
                    candidate,
                    run_scales.as_ref(),
                ),
                None => {
                    eprintln!(
                        "Warning: Candidate precursor_idx {} not found in library. Skipping.",
                        candidate.precursor_idx
                    );
                    None
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

    /// The precursor of a candidate with the fragments that scoring uses, found by idx (not array position).
    fn scoring_precursor(&self, lib: &SpecLibFlat, candidate: &Candidate) -> Option<Precursor> {
        lib.get_precursor_by_idx_filtered(
            candidate.precursor_idx,
            true,  // Always filter non-zero intensities for scoring
            true,  // Filter Y1 ions by default
            false, // Scoring uses the short ions
            self.params.top_k_fragments,
        )
    }

    fn estimate_run_scales<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> RunScales
    where
        T::QuadrupoleObservation: Sync,
    {
        RunScales::estimate(
            dia_data,
            lib,
            candidates,
            self.params.mass_tolerance,
            self.params.top_k_fragments,
        )
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
    /// `ion_fisher_lib` and is empty if the feature set has no ion-coverage block. `run_scales` is given if the
    /// feature set has the symbolic block, which extracts its own window around the candidate.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        run_scales: Option<&RunScales>,
    ) -> Option<(CandidateFeature, Vec<IonEvidence>)> {
        let feature_set = self.params.feature_set;

        let observation =
            PeakGroupObservation::new(dia_data, precursor, candidate, self.params.mass_tolerance);
        // The network reads its own window, in units of the run's mass accuracy.
        let symbolic = run_scales.map(|scales| {
            let inputs = SymbolicInputs::extract(
                dia_data,
                precursor,
                candidate,
                CORE_SIGMAS * scales.sigma_mz,
            );
            calculate_symbolic_features(&inputs, scales)
        });

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
