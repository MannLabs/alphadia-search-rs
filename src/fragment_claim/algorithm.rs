//! Extracts the matched fragment traces of every candidate, indexes them by window, apex
//! cycle and observed m/z, and derives for every candidate how much of its matched evidence a
//! higher-scoring candidate claims. `DOCS.md` gives the definitions and an example.

use numpy::ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::candidate_context::{log_mz_bin, log_mz_bin_width, pack_key};
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::peak_group_scoring::deconvolution::compute_shares;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, calculate_hyperscore, correlation_axis_0, filter_non_zero,
    median_axis_0, normalize_profiles,
};
use crate::speclib_flat::SpecLibFlat;
use crate::traits::DIADataTrait;
use crate::utils::{calculate_fragment_mz_and_errors, create_ranked_mask};

/// Apex cycles of two co-eluting candidates differ by at most one cycle in practice.
pub const DEFAULT_CYCLE_RADIUS: u32 = 1;
const MAX_WINDOWS: usize = 1 << 16;
const MAX_CYCLES: usize = 1 << 16;
const EPSILON: f32 = 1e-8;
/// Fragments that make up the top-3 features.
const TOP_N_FRAGMENTS: usize = 3;
/// Library fragments without signal are evidence against a precursor in the mixture model.
const ABSENT_EVIDENCE: bool = true;

#[derive(Debug, Clone)]
pub struct ClaimParameters {
    /// ppm tolerance of the trace extraction (the scoring tolerance)
    pub mass_tolerance: f32,
    pub top_k_fragments: usize,
    /// ppm tolerance within which two observed fragment m/z are the same ion
    pub claim_tolerance: f32,
    pub cycle_radius: u32,
    /// soft claims: the share of every matched peak that the mixture model of the co-isolated
    /// precursors attributes to the candidate (`peak_group_scoring::deconvolution`)
    pub soft_claims: bool,
}

impl ClaimParameters {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("mass_tolerance", self.mass_tolerance),
            ("claim_tolerance", self.claim_tolerance),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be a positive number, got {value}"));
            }
        }
        if self.top_k_fragments == 0 {
            return Err("top_k_fragments must be at least 1".to_string());
        }
        Ok(())
    }
}

/// The claim features of one candidate. Every field is zero for a candidate without a window,
/// without library fragments or without signal; the soft fields are zero without soft claims.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClaimFeatures {
    pub precursor_idx: usize,
    pub rank: usize,
    pub n_matched: f32,
    pub n_claimed: f32,
    pub frac_claimed: f32,
    pub frac_shared_any: f32,
    pub intensity_frac_claimed: f32,
    pub lib_frac_claimed: f32,
    pub top3_claimed: f32,
    pub n_claimants: f32,
    pub claimant_log_ratio: f32,
    pub mean_corr_free: f32,
    pub num_over_50_free: f32,
    pub num_over_0_free: f32,
    pub hyperscore_free: f32,
    pub log10_intensity_free: f32,
    pub soft_intensity_frac_claimed: f32,
    pub soft_top3_frac_claimed: f32,
    pub soft_log10_intensity_own: f32,
    pub soft_lib_corr_own: f32,
}

pub type FeatureColumn = (&'static str, fn(&ClaimFeatures) -> f32);

/// The hard-claim columns, name and accessor side by side so that neither can drift from the
/// other.
pub const HARD_FEATURES: &[FeatureColumn] = &[
    ("clm_n_matched", |f| f.n_matched),
    ("clm_n_claimed", |f| f.n_claimed),
    ("clm_frac_claimed", |f| f.frac_claimed),
    ("clm_frac_shared_any", |f| f.frac_shared_any),
    ("clm_intensity_frac_claimed", |f| f.intensity_frac_claimed),
    ("clm_lib_frac_claimed", |f| f.lib_frac_claimed),
    ("clm_top3_claimed", |f| f.top3_claimed),
    ("clm_n_claimants", |f| f.n_claimants),
    ("clm_claimant_log_ratio", |f| f.claimant_log_ratio),
    ("clm_mean_corr_free", |f| f.mean_corr_free),
    ("clm_num_over_50_free", |f| f.num_over_50_free),
    ("clm_num_over_0_free", |f| f.num_over_0_free),
    ("clm_hyperscore_free", |f| f.hyperscore_free),
    ("clm_log10_intensity_free", |f| f.log10_intensity_free),
];

/// The soft-claim columns, computed only with `ClaimParameters::soft_claims`.
pub const SOFT_FEATURES: &[FeatureColumn] = &[
    ("clm_soft_intensity_frac_claimed", |f| {
        f.soft_intensity_frac_claimed
    }),
    ("clm_soft_top3_frac_claimed", |f| f.soft_top3_frac_claimed),
    ("clm_soft_log10_intensity_own", |f| {
        f.soft_log10_intensity_own
    }),
    ("clm_soft_lib_corr_own", |f| f.soft_lib_corr_own),
];

/// The columns `compute_claim_features` fills for the given switch, hard claims first.
pub fn enabled_features(soft_claims: bool) -> impl Iterator<Item = &'static FeatureColumn> {
    let soft: &'static [FeatureColumn] = if soft_claims { SOFT_FEATURES } else { &[] };
    HARD_FEATURES.iter().chain(soft.iter())
}

/// One matched fragment of one candidate: the evidence the scorer saw for it.
#[derive(Debug, Clone)]
struct MatchedFragment {
    /// Position in the scorer's fragment list of the precursor
    fragment: usize,
    mz_observed: f32,
    intensity: f32,
    correlation: f32,
    lib_intensity: f32,
    fragment_type: u8,
}

/// What the query phase needs to know about every candidate.
struct CandidateSlot {
    precursor_idx: usize,
    /// The first valid isolation window of the precursor; the scorer sums over all of them.
    window: Option<u32>,
    cycle: u32,
    score: f32,
    /// Library intensity of every scorer fragment, for the top-3 mask
    lib_intensity: Vec<f32>,
    matched: Vec<MatchedFragment>,
    /// soft-claim features (intensity share claimed, top-3 share claimed, own log10 intensity,
    /// library correlation of the own intensities), zero without soft claims
    soft: [f32; 4],
}

impl CandidateSlot {
    fn empty(candidate: &Candidate) -> Self {
        Self {
            precursor_idx: candidate.precursor_idx,
            window: None,
            cycle: candidate.cycle_center as u32,
            score: candidate.score,
            lib_intensity: Vec::new(),
            matched: Vec::new(),
            soft: [0.0; 4],
        }
    }
}

/// One matched fragment in the index. 16 bytes.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    key: u64,
    candidate: u32,
    mz: f32,
}

struct RunIndex {
    slots: Vec<CandidateSlot>,
    /// Sorted by `key`.
    entries: Vec<IndexEntry>,
}

impl RunIndex {
    fn range(&self, lo: u64, hi: u64) -> &[IndexEntry] {
        let start = self.entries.partition_point(|entry| entry.key < lo);
        let end = self.entries.partition_point(|entry| entry.key <= hi);
        &self.entries[start..end.max(start)]
    }
}

fn validate_sizes<T: DIADataTrait>(
    dia_data: &T,
    candidates: &CandidateCollection,
) -> Result<(), String> {
    if dia_data.num_observations() >= MAX_WINDOWS {
        return Err(format!(
            "the index supports at most {MAX_WINDOWS} isolation windows, got {}",
            dia_data.num_observations()
        ));
    }
    if candidates.len() > u32::MAX as usize {
        return Err(format!(
            "the index supports at most {} candidates, got {}",
            u32::MAX,
            candidates.len()
        ));
    }
    if let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.cycle_center >= MAX_CYCLES)
    {
        return Err(format!(
            "the index supports at most {MAX_CYCLES} cycles, candidate of precursor {} has cycle {}",
            candidate.precursor_idx, candidate.cycle_center
        ));
    }
    Ok(())
}

/// Extracts the traces of one candidate exactly as the scorer does and keeps the fragments
/// with signal.
fn extract_slot<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidate: &Candidate,
    share: Option<&Array2<f32>>,
    params: &ClaimParameters,
) -> CandidateSlot {
    let mut slot = CandidateSlot::empty(candidate);
    let Some(precursor) = lib.get_precursor_by_idx_filtered(
        candidate.precursor_idx,
        true,
        true,
        params.top_k_fragments,
    ) else {
        return slot;
    };
    slot.window = dia_data
        .get_valid_observations(precursor.mz)
        .first()
        .map(|&window| window as u32);
    if slot.window.is_none()
        || precursor.fragment_mz.is_empty()
        || candidate.cycle_stop <= candidate.cycle_start
    {
        return slot;
    }
    let dense = DenseXICMZObservation::new(
        dia_data,
        precursor.mz,
        candidate.cycle_start,
        candidate.cycle_stop,
        params.mass_tolerance,
        &precursor.fragment_mz,
    );
    let normalized = normalize_profiles(&dense.dense_xic, 1);
    let median_profile_filtered = median_axis_0(&filter_non_zero(&normalized));
    let correlations = correlation_axis_0(&median_profile_filtered, &normalized);
    let intensities = dense.dense_xic.sum_axis(Axis(1));
    let (mz_observed, _) =
        calculate_fragment_mz_and_errors(&dense.dense_mz, &dense.dense_xic, &precursor.fragment_mz);
    slot.matched = (0..precursor.fragment_mz.len())
        .filter(|&f| intensities[f] > 0.0 && mz_observed[f] > 0.0)
        .map(|f| MatchedFragment {
            fragment: f,
            mz_observed: mz_observed[f],
            intensity: intensities[f],
            correlation: correlations[f],
            lib_intensity: precursor.fragment_intensity[f],
            fragment_type: precursor.fragment_type[f],
        })
        .collect();
    if params.soft_claims && !slot.matched.is_empty() {
        slot.soft = soft_claims(
            &dense.dense_xic,
            share,
            &intensities.to_vec(),
            &precursor.fragment_intensity,
        );
    }
    slot.lib_intensity = precursor.fragment_intensity;
    slot
}

/// The soft-claim features of one candidate from its raw fragment block and its share matrix
/// (none: nothing shared, every peak is its own).
fn soft_claims(
    xic: &Array2<f32>,
    share: Option<&Array2<f32>>,
    raw: &[f32],
    lib_intensity: &[f32],
) -> [f32; 4] {
    let own: Vec<f32> = match share {
        Some(s) if s.dim() == xic.dim() => (xic * s).sum_axis(Axis(1)).to_vec(),
        _ => raw.to_vec(),
    };
    let claimed_frac = |mask: &dyn Fn(usize) -> bool| {
        let raw_sum: f32 = (0..raw.len()).filter(|&f| mask(f)).map(|f| raw[f]).sum();
        let own_sum: f32 = (0..raw.len()).filter(|&f| mask(f)).map(|f| own[f]).sum();
        if raw_sum > 0.0 {
            (1.0 - own_sum / raw_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let top3 = create_ranked_mask(lib_intensity, 0, TOP_N_FRAGMENTS);
    [
        claimed_frac(&|_| true),
        claimed_frac(&|f| top3[f]),
        (own.iter().sum::<f32>() + EPSILON).log10(),
        calculate_correlation_safe(&own, lib_intensity),
    ]
}

fn build_index<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidates: &CandidateCollection,
    params: &ClaimParameters,
) -> RunIndex {
    let shares = if params.soft_claims {
        compute_shares(
            dia_data,
            lib,
            candidates,
            params.top_k_fragments,
            params.mass_tolerance,
            ABSENT_EVIDENCE,
        )
    } else {
        Vec::new()
    };
    let slots: Vec<CandidateSlot> = candidates
        .par_iter()
        .enumerate()
        .map(|(i, candidate)| {
            let share = shares.get(i).and_then(|s| s.as_ref());
            extract_slot(dia_data, lib, candidate, share, params)
        })
        .collect();
    let bin_width = log_mz_bin_width(params.claim_tolerance);
    let mut entries: Vec<IndexEntry> = slots
        .par_iter()
        .enumerate()
        .flat_map_iter(|(candidate_idx, slot)| {
            slot.matched.iter().filter_map(move |m| {
                Some(IndexEntry {
                    key: pack_key(
                        slot.window?,
                        slot.cycle,
                        log_mz_bin(m.mz_observed, bin_width),
                    ),
                    candidate: candidate_idx as u32,
                    mz: m.mz_observed,
                })
            })
        })
        .collect();
    entries.par_sort_unstable_by_key(|entry| entry.key);
    RunIndex { slots, entries }
}

/// Who shares and who claims the matched fragments of one candidate.
struct Ownership {
    shared_any: Vec<bool>,
    claimed: Vec<bool>,
    /// distinct candidates that claim at least one fragment
    claimants: Vec<u32>,
    best_claimant_score: f32,
}

fn ownership(
    slot: &CandidateSlot,
    window: u32,
    index: &RunIndex,
    params: &ClaimParameters,
) -> Ownership {
    let cycle_lo = slot.cycle.saturating_sub(params.cycle_radius);
    let cycle_hi = slot
        .cycle
        .saturating_add(params.cycle_radius)
        .min(MAX_CYCLES as u32 - 1);
    let bin_width = log_mz_bin_width(params.claim_tolerance);

    // pass 1: the other candidates sharing every matched fragment, once each
    let sharers: Vec<Vec<(u32, f32)>> = slot
        .matched
        .iter()
        .map(|m| {
            let mut sharers: Vec<(u32, f32)> = Vec::new();
            let tolerance = m.mz_observed * params.claim_tolerance * 1e-6;
            let bin = log_mz_bin(m.mz_observed, bin_width);
            for cycle in cycle_lo..=cycle_hi {
                let lo = pack_key(window, cycle, bin.saturating_sub(1));
                let hi = pack_key(window, cycle, bin.saturating_add(1));
                for entry in index.range(lo, hi) {
                    let other = &index.slots[entry.candidate as usize];
                    if other.precursor_idx == slot.precursor_idx
                        || (entry.mz - m.mz_observed).abs() > tolerance
                    {
                        continue;
                    }
                    sharers.push((entry.candidate, other.score));
                }
            }
            sharers.sort_unstable_by_key(|s| s.0);
            sharers.dedup_by_key(|s| s.0);
            sharers
        })
        .collect();

    // pass 2: a sharer with a strictly higher selection score claims the fragment
    let mut claimed = vec![false; sharers.len()];
    let mut claimants: Vec<u32> = Vec::new();
    let mut best_claimant_score = f32::NEG_INFINITY;
    for (i, fragment_sharers) in sharers.iter().enumerate() {
        for &(other, score) in fragment_sharers {
            if score > slot.score {
                claimed[i] = true;
                claimants.push(other);
                best_claimant_score = best_claimant_score.max(score);
            }
        }
    }
    claimants.sort_unstable();
    claimants.dedup();

    Ownership {
        shared_any: sharers.iter().map(|s| !s.is_empty()).collect(),
        claimed,
        claimants,
        best_claimant_score,
    }
}

fn candidate_features(
    candidate_idx: usize,
    candidate: &Candidate,
    index: &RunIndex,
    params: &ClaimParameters,
) -> ClaimFeatures {
    let mut features = ClaimFeatures {
        precursor_idx: candidate.precursor_idx,
        rank: candidate.rank,
        ..Default::default()
    };
    let slot = &index.slots[candidate_idx];
    let Some(window) = slot.window else {
        return features;
    };
    if slot.matched.is_empty() {
        return features;
    }
    let Ownership {
        shared_any,
        claimed,
        claimants,
        best_claimant_score,
    } = ownership(slot, window, index, params);

    let n = slot.matched.len();
    let claimed_sum = |value: fn(&MatchedFragment) -> f32| -> f32 {
        slot.matched
            .iter()
            .zip(&claimed)
            .filter(|(_, &c)| c)
            .map(|(m, _)| value(m))
            .sum()
    };
    let intensity_total: f32 = slot.matched.iter().map(|m| m.intensity).sum();
    let lib_total: f32 = slot.matched.iter().map(|m| m.lib_intensity).sum();
    let intensity_claimed = claimed_sum(|m| m.intensity);
    let lib_claimed = claimed_sum(|m| m.lib_intensity);
    let top3 = create_ranked_mask(&slot.lib_intensity, 0, TOP_N_FRAGMENTS);
    let top3_claimed = slot
        .matched
        .iter()
        .zip(&claimed)
        .filter(|(m, &c)| c && top3[m.fragment])
        .count();
    let free: Vec<&MatchedFragment> = slot
        .matched
        .iter()
        .zip(&claimed)
        .filter(|(_, &c)| !c)
        .map(|(m, _)| m)
        .collect();
    let n_claimed = claimed.iter().filter(|&&c| c).count();
    let n_shared = shared_any.iter().filter(|&&s| s).count();

    // hyperscore over the free fragments, with the library intensities and types of the
    // scorer's fragment list
    let free_types: Vec<u8> = free.iter().map(|m| m.fragment_type).collect();
    let free_lib: Vec<f32> = free.iter().map(|m| m.lib_intensity).collect();
    let free_mask = vec![true; free.len()];

    features.n_matched = n as f32;
    features.n_claimed = n_claimed as f32;
    features.frac_claimed = n_claimed as f32 / n as f32;
    features.frac_shared_any = n_shared as f32 / n as f32;
    features.intensity_frac_claimed = if intensity_total > 0.0 {
        intensity_claimed / intensity_total
    } else {
        0.0
    };
    features.lib_frac_claimed = if lib_total > 0.0 {
        lib_claimed / lib_total
    } else {
        0.0
    };
    features.top3_claimed = top3_claimed as f32;
    features.n_claimants = claimants.len() as f32;
    features.claimant_log_ratio =
        if !claimants.is_empty() && best_claimant_score > 0.0 && slot.score > 0.0 {
            (best_claimant_score / slot.score).ln()
        } else {
            0.0
        };
    features.mean_corr_free = if free.is_empty() {
        0.0
    } else {
        free.iter().map(|m| m.correlation).sum::<f32>() / free.len() as f32
    };
    features.num_over_50_free = free.iter().filter(|m| m.correlation > 0.5).count() as f32;
    features.num_over_0_free = free.iter().filter(|m| m.correlation > 0.0).count() as f32;
    features.hyperscore_free = calculate_hyperscore(&free_types, &free_lib, &free_mask);
    features.log10_intensity_free =
        (free.iter().map(|m| m.intensity).sum::<f32>() + EPSILON).log10();
    [
        features.soft_intensity_frac_claimed,
        features.soft_top3_frac_claimed,
        features.soft_log10_intensity_own,
        features.soft_lib_corr_own,
    ] = slot.soft;
    features
}

/// Computes the claim features of every candidate, in the candidate order of the caller.
pub fn compute_claim_features<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidates: &CandidateCollection,
    params: &ClaimParameters,
) -> Result<Vec<ClaimFeatures>, String> {
    params.validate()?;
    validate_sizes(dia_data, candidates)?;
    let start_time = Instant::now();
    let index = build_index(dia_data, lib, candidates, params);
    let features: Vec<ClaimFeatures> = candidates
        .par_iter()
        .enumerate()
        .map(|(candidate_idx, candidate)| {
            candidate_features(candidate_idx, candidate, &index, params)
        })
        .collect();
    println!(
        "Computed claim features for {} candidates ({} indexed matched fragments) in {:.2}s",
        candidates.len(),
        index.entries.len(),
        start_time.elapsed().as_secs_f64()
    );
    Ok(features)
}
