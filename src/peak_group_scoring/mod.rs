use numpy::ndarray::{s, Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::core::calculate_core_features;
use crate::peak_group_scoring::detection::{
    background_rates, flank_counts, flank_hit_intensities, DetectionFeatures,
    DETECTION_PRIOR_SAMPLE,
};
use crate::peak_group_scoring::faint::{
    apex_fragments, faint_features, fit_curve, ApexFragment, BackgroundDensity, FaintFeatures,
    FaintModel, FIT_CANDIDATES,
};
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::features::{CandidateFeature, CandidateFeatureCollection};
use crate::peak_group_scoring::ion_coverage::{
    calculate_ion_coverage_features, num_ion_classes, IonEvidence, IonNoiseModel,
};
use crate::peak_group_scoring::library_agreement::calculate_library_agreement_features;
use crate::peak_group_scoring::ms1::{
    calculate_ms1_apex_features, isotope_mz, Ms1ApexFeatures, Ms1Features, C13_MASS_DELTA,
    ISOTOPES_BELOW,
};
use crate::peak_group_scoring::ms1_accuracy::{
    calculate_ms1_accuracy_features, Ms1AccuracyFeatures, ACCURATE_PPM,
};
use crate::peak_group_scoring::ms1_fine::ms1_fine_features;
use crate::peak_group_scoring::ms1_profile::calculate_ms1_profile_features;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::shadow::{shadow_lib_share, shadow_mz};
use crate::peak_group_scoring::window_share::{
    apex_column, calculate_window_share_features, window_share_context, WindowShareEntry,
    WINDOW_SHARE_CYCLE_RADIUS,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;

pub mod core;
pub mod detection;
pub mod faint;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod ion_coverage;
pub mod library_agreement;
pub mod ms1;
pub mod ms1_accuracy;
pub mod ms1_fine;
pub mod ms1_profile;
pub mod observation;
pub mod parameters;
pub mod shadow;
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

    /// The apex isotope statistics of each candidate for an MS1 mass calibration: `ms1_iso_cosine`, `ms1_n_isotopes`,
    /// `ms1_log_m0_apex` and `ms1_ppm_signed` (see `ms1::Ms1ApexFeatures`), with `precursor_idx` and `rank`, extracted
    /// at `ms1_mass_tolerance` around the MS1 m/z that `SpecLibFlat::set_isotopes` set. All 0 for a candidate without
    /// isotopes or a run without MS1 scans.
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
                    self.params.top_k_fragments,
                )?;
                let observation = PeakGroupObservation::new(
                    dia_data,
                    &precursor,
                    candidate,
                    self.params.mass_tolerance,
                );
                let apex = self
                    .ms1_isotope_xic(dia_data, lib, candidate, &ms1_observations)
                    .map_or_else(Ms1ApexFeatures::default, |isotopes| {
                        calculate_ms1_apex_features(
                            &isotopes.xic.dense_xic,
                            &isotopes.xic.dense_mz,
                            isotopes.ms1_mz,
                            isotopes.abundance,
                            &observation.median_profile_filtered,
                        )
                    });
                Some((candidate.precursor_idx, candidate.rank, apex))
            })
            .collect();

        let n = scored.len();
        let mut precursor_idx = Array1::<u64>::zeros(n);
        let mut rank = Array1::<u64>::zeros(n);
        let mut columns = [
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
        ];
        for (i, (idx, candidate_rank, apex)) in scored.iter().enumerate() {
            precursor_idx[i] = *idx as u64;
            rank[i] = *candidate_rank as u64;
            for (column, value) in columns.iter_mut().zip([
                apex.iso_cosine,
                apex.n_isotopes,
                apex.log_m0_apex,
                apex.ppm_signed,
            ]) {
                column[i] = value;
            }
        }
        let dict = PyDict::new(py);
        dict.set_item("precursor_idx", precursor_idx.into_pyarray(py))?;
        dict.set_item("rank", rank.into_pyarray(py))?;
        for (name, column) in [
            "ms1_iso_cosine",
            "ms1_n_isotopes",
            "ms1_log_m0_apex",
            "ms1_ppm_signed",
        ]
        .into_iter()
        .zip(columns)
        {
            dict.set_item(name, column.into_pyarray(py))?;
        }
        Ok(dict.into())
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

        // The MS1 survey scans of the MS1 block; none if the library has no isotopes, so that the block costs nothing
        // then and every candidate gets the missing values
        let ms1_observations: Vec<usize> =
            if self.params.feature_set.has_block(FeatureBlockId::Ms1) && lib.has_isotopes() {
                dia_data.ms1_observations()
            } else {
                Vec::new()
            };

        // The run-wide inputs of the detection block, from all candidates alike (no labels)
        let detection_model =
            scores_detection(&self.params).then(|| self.detection_model(dia_data, lib, candidates));

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
                        lib,
                        &precursor,
                        candidate,
                        &window_intensities,
                        &ms1_observations,
                        detection_model.as_ref(),
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
    /// summed MS2 intensity per isolation window and cycle. `ms1_observations` are the MS1 survey scans the MS1 block
    /// reads; without them, or without library isotopes for the precursor, the MS1 block takes its missing values.
    /// `detection_model` holds the run-wide inputs of the detection block; without it (`detection_features` off) the
    /// detection block takes its missing values.
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        precursor: &Precursor,
        candidate: &Candidate,
        window_intensities: &[Vec<f32>],
        ms1_observations: &[usize],
        detection_model: Option<&DetectionModel>,
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

        let ms1 = feature_set.has_block(FeatureBlockId::Ms1).then(|| {
            self.calculate_ms1(
                dia_data,
                lib,
                candidate,
                &observation,
                core.fwhm_rt,
                ms1_observations,
            )
        });

        let detection =
            feature_set
                .has_block(FeatureBlockId::Detection)
                .then(|| match detection_model {
                    Some(model) => self.calculate_detection(
                        dia_data,
                        precursor,
                        candidate,
                        &observation,
                        model,
                    ),
                    None => DetectionFeatures::missing(),
                });

        let feature = CandidateFeature {
            precursor_idx: candidate.precursor_idx,
            rank: candidate.rank,
            core,
            ion_coverage,
            library_agreement,
            window_share,
            ms1,
            detection,
        };
        Some(ScoredCandidate {
            feature,
            ion_evidence,
            window_share: window_share_entry,
        })
    }

    /// The MS1 block of one candidate, all at the precursor's MS1 m/z over the candidate's cycles: the isotope XICs
    /// summed over the MS1 scans of a cycle, against the whole fragment profile (`ms1_profile`) and at its apex
    /// (`ms1_iso_m1_err`); the M0 / M+1 traces of each MS1 scan apart at ACCURATE_PPM (`ms1_accuracy`) and at the MS1
    /// tolerance, at the scans' own times (`ms1_fine`).
    fn calculate_ms1<T: DIADataTrait>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidate: &Candidate,
        observation: &PeakGroupObservation,
        fwhm_rt: f32,
        ms1_observations: &[usize],
    ) -> Ms1Features {
        let Some(Ms1IsotopeXic {
            charge,
            ms1_mz,
            abundance: isotope_intensity,
            traces,
            xic: ms1_xic,
        }) = self.ms1_isotope_xic(dia_data, lib, candidate, ms1_observations)
        else {
            return Ms1Features::missing();
        };
        let fragment_profile = &observation.median_profile_filtered;
        let apex = apex_column(
            fragment_profile,
            candidate.cycle_center.saturating_sub(candidate.cycle_start),
        );
        let profile = calculate_ms1_profile_features(
            &ms1_xic.dense_xic,
            isotope_intensity,
            fragment_profile,
            apex,
        );
        let ms1_iso_m1_err = calculate_ms1_apex_features(
            &ms1_xic.dense_xic,
            &ms1_xic.dense_mz,
            ms1_mz,
            isotope_intensity,
            fragment_profile,
        )
        .iso_m1_err;

        // M0 and M+1 of each MS1 scan apart
        let scan_xic = |obs: usize, tolerance: f32, mz: &[f32]| {
            DenseXICMZObservation::from_observations(
                dia_data,
                vec![obs],
                candidate.cycle_start,
                candidate.cycle_stop,
                tolerance,
                mz,
            )
            .dense_xic
        };
        let accurate_mz = &traces[ISOTOPES_BELOW..(ISOTOPES_BELOW + 2).min(traces.len())];
        let accuracy = if accurate_mz.len() == 2 {
            let scan_xics: Vec<_> = ms1_observations
                .iter()
                .map(|&obs| scan_xic(obs, ACCURATE_PPM, accurate_mz))
                .collect();
            calculate_ms1_accuracy_features(&scan_xics, fragment_profile)
        } else {
            Ms1AccuracyFeatures::default()
        };

        let fine_mz = [ms1_mz, ms1_mz + C13_MASS_DELTA / charge.max(1) as f32];
        let n_cycles = candidate.cycle_stop - candidate.cycle_start;
        let mut m0 = Array2::<f32>::zeros((ms1_observations.len(), n_cycles));
        let mut m1 = m0.clone();
        for (row, &obs) in ms1_observations.iter().enumerate() {
            let xic = scan_xic(obs, self.params.ms1_mass_tolerance, &fine_mz);
            m0.row_mut(row).assign(&xic.row(0));
            m1.row_mut(row).assign(&xic.row(1));
        }
        let ms1_position: Vec<f32> = ms1_observations
            .iter()
            .map(|&obs| dia_data.cycle_position(obs))
            .collect();
        let windows = &observation.contributing_obs_indices;
        let fragment_position = if windows.is_empty() {
            0.0
        } else {
            windows
                .iter()
                .map(|&obs| dia_data.cycle_position(obs))
                .sum::<f32>()
                / windows.len() as f32
        };
        let cycle_rt = dia_data
            .rt_index()
            .rt
            .slice(s![candidate.cycle_start..candidate.cycle_stop])
            .to_vec();
        let fine = ms1_fine_features(
            &m0,
            &m1,
            &ms1_position,
            &cycle_rt,
            fragment_position,
            fragment_profile,
            fwhm_rt,
        );

        Ms1Features {
            ms1_signal_at_apex: profile.ms1_signal_at_apex,
            ms1_isotope_correlation: profile.ms1_isotope_correlation,
            ms1_isotope_cosine: profile.ms1_isotope_cosine,
            ms1_m_minus_1_frac: profile.ms1_m_minus_1_frac,
            ms1_log_intensity: profile.ms1_log_intensity,
            ms1_iso_m1_err,
            ms1_m0_accurate_share: accuracy.ms1_m0_accurate_share,
            ms1_m1_accurate_share: accuracy.ms1_m1_accurate_share,
            ms1_fine_shape: fine.ms1_fine_shape,
            ms1_fine_iso_corr: fine.ms1_fine_iso_corr,
        }
    }

    /// The isotope XICs M-1 .. M+(n-1) of a candidate's precursor over the MS1 scans and the candidate's cycles, at
    /// the MS1 m/z and the MS1 tolerance; None without MS1 scans or without library isotopes for the precursor.
    fn ms1_isotope_xic<'a, T: DIADataTrait>(
        &self,
        dia_data: &T,
        lib: &'a SpecLibFlat,
        candidate: &Candidate,
        ms1_observations: &[usize],
    ) -> Option<Ms1IsotopeXic<'a>> {
        if ms1_observations.is_empty() {
            return None;
        }
        let (charge, ms1_mz, abundance) = lib.get_isotopes_by_idx(candidate.precursor_idx)?;
        let traces = isotope_mz(ms1_mz, charge, abundance.len());
        let xic = DenseXICMZObservation::from_observations(
            dia_data,
            ms1_observations.to_vec(),
            candidate.cycle_start,
            candidate.cycle_stop,
            self.params.ms1_mass_tolerance,
            &traces,
        );
        Some(Ms1IsotopeXic {
            charge,
            ms1_mz,
            abundance,
            traces,
            xic,
        })
    }

    /// The detection XIC of a candidate (all its fragments over `detection::window` around its selection apex) and the
    /// apex column in it.
    fn detection_xic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
    ) -> (Array2<f32>, usize) {
        let num_cycles = dia_data.rt_index().rt.len();
        let (start, stop) = detection::window(candidate.cycle_center, num_cycles);
        let xic = DenseXICObservation::new(
            dia_data,
            precursor.mz,
            start,
            stop,
            self.params.mass_tolerance,
            &precursor.fragment_mz,
        );
        (xic.dense_xic, candidate.cycle_center - start)
    }

    /// The detection XIC of a candidate's precursor, filtered as for scoring; None if it is not in the library.
    fn library_detection_xic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidate: &Candidate,
    ) -> Option<(Precursor, Array2<f32>, usize)> {
        let precursor = lib.get_precursor_by_idx_filtered(
            candidate.precursor_idx,
            true,
            true,
            self.params.top_k_fragments,
        )?;
        let (xic, apex_col) = self.detection_xic(dia_data, &precursor, candidate);
        Some((precursor, xic, apex_col))
    }

    /// Run-wide inputs of the detection block, from the candidates alone (targets and decoys alike): the mean flank
    /// hit rate of a fragment channel and the log2-intensity density of flank hits over a sample of the candidates, and
    /// the detection curve fitted on the candidates with the highest selection scores.
    fn detection_model<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> DetectionModel {
        let step = (candidates.len() / DETECTION_PRIOR_SAMPLE).max(1);
        let sample: Vec<&Candidate> = candidates.iter().step_by(step).collect();
        let flank_columns: Vec<(Vec<u32>, u32, Vec<f64>)> = sample
            .par_iter()
            .filter_map(|candidate| {
                let (_, xic, apex_col) = self.library_detection_xic(dia_data, lib, candidate)?;
                let (flank_hits, flank_scans) = flank_counts(&xic, apex_col);
                let log2_hits = flank_hit_intensities(&xic, apex_col)
                    .into_iter()
                    .map(|v| (v as f64).log2())
                    .collect();
                Some((flank_hits, flank_scans, log2_hits))
            })
            .collect();
        let (hits, scans) =
            flank_columns
                .iter()
                .fold((0u64, 0u64), |acc, (flank_hits, flank_scans, _)| {
                    (
                        acc.0 + flank_hits.iter().map(|&h| h as u64).sum::<u64>(),
                        acc.1 + *flank_scans as u64 * flank_hits.len() as u64,
                    )
                });
        let flank_hit_rate = if scans == 0 {
            0.0
        } else {
            hits as f32 / scans as f32
        };
        let log2_hits: Vec<f64> = flank_columns.into_iter().flat_map(|(_, _, v)| v).collect();
        let background = BackgroundDensity::from_log2(&log2_hits);

        let mut by_score: Vec<&Candidate> = candidates.iter().collect();
        let n_fit = FIT_CANDIDATES.min(by_score.len());
        if n_fit > 0 && n_fit < by_score.len() {
            by_score.select_nth_unstable_by(n_fit - 1, |a, b| b.score.total_cmp(&a.score));
        }
        let fit_set: Vec<Vec<ApexFragment>> = by_score[..n_fit]
            .par_iter()
            .filter_map(|candidate| {
                let (precursor, xic, apex_col) =
                    self.library_detection_xic(dia_data, lib, candidate)?;
                let rates = background_rates(&xic, apex_col, flank_hit_rate);
                apex_fragments(&xic, apex_col, &precursor.fragment_intensity, &rates)
            })
            .collect();
        let curve = fit_curve(&fit_set, &background);
        println!(
            "Detection model: flank hit rate {:.3}, detection limit 2^{:.2}, width {:.2}, scatter {:.2} (log2)",
            flank_hit_rate, curve.limit, curve.width, curve.scatter
        );
        DetectionModel {
            flank_hit_rate,
            faint: FaintModel { curve, background },
        }
    }

    /// The detection block of one candidate: the fragment M-1 shadow share over the candidate window, and the
    /// faint-detection terms from its detection XIC.
    fn calculate_detection<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        observation: &PeakGroupObservation,
        model: &DetectionModel,
    ) -> DetectionFeatures {
        // The XICs one 1+ and one 2+ isotope spacing below each fragment, over the same isolation windows, cycles
        // and tolerance as the fragment XICs
        let shadow_xic = DenseXICObservation::new(
            dia_data,
            precursor.mz,
            candidate.cycle_start,
            candidate.cycle_stop,
            self.params.mass_tolerance,
            &shadow_mz(&precursor.fragment_mz),
        );
        let shadow_lib_share = shadow_lib_share(
            &observation.dense_xic,
            &shadow_xic.dense_xic,
            &observation.median_profile_filtered,
            &observation.correlations,
            &precursor.fragment_intensity,
            &precursor.fragment_mz,
        );

        let (xic, apex_col) = self.detection_xic(dia_data, precursor, candidate);
        let faint = apex_fragments(
            &xic,
            apex_col,
            &precursor.fragment_intensity,
            &background_rates(&xic, apex_col, model.flank_hit_rate),
        )
        .map_or_else(FaintFeatures::missing, |fragments| {
            faint_features(&fragments, &model.faint)
        });

        DetectionFeatures {
            shadow_lib_share,
            det_faint_hits: faint.det_faint_hits,
            det_faint_miss: faint.det_faint_miss,
        }
    }
}

/// The MS1 isotope traces of a candidate's precursor and their XICs.
struct Ms1IsotopeXic<'a> {
    charge: u8,
    ms1_mz: f32,
    /// library isotope abundances M0..
    abundance: &'a [f32],
    /// m/z of the traces M-1 .. M+(n-1)
    traces: Vec<f32>,
    xic: DenseXICMZObservation,
}

/// Whether the detection block is scored: it needs the run-wide detection model.
fn scores_detection(params: &ScoringParameters) -> bool {
    params.detection_features && params.feature_set.has_block(FeatureBlockId::Detection)
}

/// The run-wide inputs of the detection block: the mean flank hit rate of a fragment channel (the prior of each
/// channel's background rate) and the faint-detection model.
struct DetectionModel {
    flank_hit_rate: f32,
    faint: FaintModel,
}
