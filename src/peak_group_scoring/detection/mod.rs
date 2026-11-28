//! Fragment detection block for sparsely sampled, ion-counting MS2 data (final feature set only).
//!
//! At low input on Astral a peptide's peak is about one MS2 scan wide and a fragment is either seen (a few ions) or
//! not. Correlations over the candidate window then compare one spike with noise and collapse for faint peptides as
//! for decoys. The block holds three features that do not need several points across the peak:
//! - `shadow_lib_share` (`shadow`): the share of coherent fragments that are the M+1 isotope of another ion,
//! - `det_faint_hits`, `det_faint_miss` (`faint`): the hit and miss terms of an abundance-aware likelihood ratio of
//!   the fragments seen and missed at the apex scan, against each channel's background hit rate.
//!
//! The background hit rate of a channel is measured on flank scans of the same channel, away from the peak, so a hit
//! in a quiet channel counts more than a hit in a channel that common fragments of co-isolated peptides light up all
//! the time. The detection XIC is taken over `FLANK_MAX` cycles on both sides of the candidate apex; hits are
//! intensity > 0. A channel's background rate is (flank hits + PRIOR_WEIGHT * b0) / (flank scans + PRIOR_WEIGHT) with
//! b0 the run's mean flank hit rate, estimated over all candidates alike (no labels).
//!
//! The block needs a run-wide model (`DetectionModel`) and is scored only with
//! `ScoringParameters::detection_features`; without it every value is 0. All values are finite.

use numpy::ndarray::Array2;
use rayon::prelude::*;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::DenseXICObservation;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::feature_set::FeatureBlockId;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::ScoringParameters;
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;

pub mod faint;
pub mod shadow;

#[cfg(test)]
mod tests;

use faint::{
    apex_fragments, faint_features, fit_curve, ApexFragment, BackgroundDensity, FaintFeatures,
    FaintModel, FIT_CANDIDATES,
};
use shadow::{shadow_lib_share, shadow_mz};

/// Candidates sampled to estimate the run-wide flank hit rate and the background-hit intensity density.
pub const DETECTION_PRIOR_SAMPLE: usize = 20_000;
/// Flank scans start this many cycles from the apex: a peak is ~1 scan wide, so these hold background only.
pub const FLANK_MIN: usize = 4;
/// Flank scans end this many cycles from the apex.
pub const FLANK_MAX: usize = 9;
/// Pseudo-scans of the run-wide rate b0 added to each channel's flank count.
const PRIOR_WEIGHT: f32 = 2.0;
/// Bounds that keep the log-likelihoods finite.
const MIN_PROBABILITY: f32 = 1e-6;

feature_block! {
    /// The fragment detection features of one candidate.
    pub struct DetectionFeatures {
        /// library-weighted share of the coherent fragments that are the M+1 isotope of another ion (`shadow`)
        pub shadow_lib_share: f32,
        /// hit terms of the faint-detection likelihood ratio at its best abundance (`faint`)
        pub det_faint_hits: f32,
        /// miss terms of the faint-detection likelihood ratio at its best abundance, <= 0 (`faint`)
        pub det_faint_miss: f32,
    }
}

impl DetectionFeatures {
    /// Feature values without `ScoringParameters::detection_features`, or for a candidate whose traces do not fit.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// Whether the detection block is scored: it needs the run-wide detection model.
pub fn scores_detection(params: &ScoringParameters) -> bool {
    params.detection_features && params.feature_set.has_block(FeatureBlockId::Detection)
}

/// The run-wide inputs of the detection block: the mean flank hit rate of a fragment channel (the prior of each
/// channel's background rate) and the faint-detection model.
pub struct DetectionModel {
    flank_hit_rate: f32,
    faint: FaintModel,
}

impl DetectionModel {
    /// Run-wide inputs of the detection block, from the candidates alone (targets and decoys alike): the mean flank
    /// hit rate of a fragment channel and the log2-intensity density of flank hits over a sample of the candidates,
    /// and the detection curve fitted on the candidates with the highest selection scores.
    pub fn fit<T: DIADataTrait + Sync>(
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
        params: &ScoringParameters,
    ) -> Self {
        let step = (candidates.len() / DETECTION_PRIOR_SAMPLE).max(1);
        let sample: Vec<&Candidate> = candidates.iter().step_by(step).collect();
        let flank_columns: Vec<(Vec<u32>, u32, Vec<f64>)> = sample
            .par_iter()
            .filter_map(|candidate| {
                let (_, xic, apex_col) = library_detection_xic(dia_data, lib, candidate, params)?;
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
                    library_detection_xic(dia_data, lib, candidate, params)?;
                let rates = background_rates(&xic, apex_col, flank_hit_rate);
                apex_fragments(&xic, apex_col, &precursor.fragment_intensity, &rates)
            })
            .collect();
        let curve = fit_curve(&fit_set, &background);
        println!(
            "Detection model: flank hit rate {:.3}, detection limit 2^{:.2}, width {:.2}, scatter {:.2} (log2)",
            flank_hit_rate, curve.limit, curve.width, curve.scatter
        );
        Self {
            flank_hit_rate,
            faint: FaintModel { curve, background },
        }
    }
}

/// The detection block of one candidate: the fragment M-1 shadow share over the candidate window, and the
/// faint-detection terms from its detection XIC.
pub fn calculate_detection_features<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    observation: &PeakGroupObservation,
    model: &DetectionModel,
    params: &ScoringParameters,
) -> DetectionFeatures {
    // The XICs one 1+ and one 2+ isotope spacing below each fragment, over the same isolation windows, cycles and
    // tolerance as the fragment XICs
    let shadow_xic = DenseXICObservation::new(
        dia_data,
        precursor.mz,
        candidate.cycle_start,
        candidate.cycle_stop,
        params.mass_tolerance,
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

    let (xic, apex_col) = detection_xic(dia_data, precursor, candidate, params);
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

/// The detection XIC of a candidate (all its fragments over `window` around its selection apex) and the apex column
/// in it.
fn detection_xic<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    params: &ScoringParameters,
) -> (Array2<f32>, usize) {
    let num_cycles = dia_data.rt_index().rt.len();
    let (start, stop) = window(candidate.cycle_center, num_cycles);
    let xic = DenseXICObservation::new(
        dia_data,
        precursor.mz,
        start,
        stop,
        params.mass_tolerance,
        &precursor.fragment_mz,
    );
    (xic.dense_xic, candidate.cycle_center - start)
}

/// The detection XIC of a candidate's precursor, filtered as for scoring; None if it is not in the library.
fn library_detection_xic<T: DIADataTrait>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidate: &Candidate,
    params: &ScoringParameters,
) -> Option<(Precursor, Array2<f32>, usize)> {
    let precursor = lib.get_precursor_by_idx_filtered(
        candidate.precursor_idx,
        true,
        true,
        false, // Scoring uses the short ions
        params.top_k_fragments,
    )?;
    let (xic, apex_col) = detection_xic(dia_data, &precursor, candidate, params);
    Some((precursor, xic, apex_col))
}

/// The cycle range [start, stop) of the detection XIC around `apex` in a run of `num_cycles` cycles.
pub fn window(apex: usize, num_cycles: usize) -> (usize, usize) {
    (
        apex.saturating_sub(FLANK_MAX),
        (apex + FLANK_MAX + 1).min(num_cycles),
    )
}

/// Flank hits and flank scans per fragment of a detection XIC whose apex is column `apex_col`.
pub fn flank_counts(xic: &Array2<f32>, apex_col: usize) -> (Vec<u32>, u32) {
    let flank: Vec<usize> = (0..xic.ncols())
        .filter(|&c| {
            let distance = c.abs_diff(apex_col);
            (FLANK_MIN..=FLANK_MAX).contains(&distance)
        })
        .collect();
    let hits = (0..xic.nrows())
        .map(|f| flank.iter().filter(|&&c| xic[[f, c]] > 0.0).count() as u32)
        .collect();
    (hits, flank.len() as u32)
}

/// Intensities of all flank hits of a detection XIC whose apex is column `apex_col`.
pub fn flank_hit_intensities(xic: &Array2<f32>, apex_col: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for c in 0..xic.ncols() {
        if (FLANK_MIN..=FLANK_MAX).contains(&c.abs_diff(apex_col)) {
            out.extend(xic.column(c).iter().filter(|&&v| v > 0.0));
        }
    }
    out
}

/// Each fragment channel's background hit rate: its flank hit share, shrunk towards the run's rate `b0`.
pub fn background_rates(xic: &Array2<f32>, apex_col: usize, b0: f32) -> Vec<f32> {
    let (flank_hits, flank_scans) = flank_counts(xic, apex_col);
    flank_hits
        .iter()
        .map(|&k| clamp((k as f32 + PRIOR_WEIGHT * b0) / (flank_scans as f32 + PRIOR_WEIGHT)))
        .collect()
}

fn clamp(p: f32) -> f32 {
    p.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}
