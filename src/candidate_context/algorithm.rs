//! Builds a sorted fragment index over all candidates of a run and derives the competition and
//! context features of every candidate from it. `DOCS.md` gives the data layout, the feature
//! definitions and an example.

use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::speclib_flat::SpecLibFlat;
use crate::traits::DIADataTrait;

/// Single source of truth for the feature names, in the order of [`ContextFeatures::values`].
pub const CONTEXT_FEATURE_NAMES: &[&str] = &[
    "ctx_candidate_density",
    "ctx_n_competitors",
    "ctx_n_competitors_higher",
    "ctx_claimant_rank",
    "ctx_shared_frac_any",
    "ctx_shared_frac_higher",
    "ctx_shared_lib_intensity_frac_higher",
    "ctx_competitor_log_ratio",
];

/// Same fragment set as `ScoringParameters` uses on the alphaDIA side.
pub const DEFAULT_TOP_K_FRAGMENTS: usize = 12;
/// Fewer shared ions are a coincidence, see `fragment_competition`.
pub const DEFAULT_MIN_SHARED: usize = 3;
/// Apex cycles of two co-eluting candidates differ by at most one cycle in practice.
pub const DEFAULT_CYCLE_RADIUS: usize = 1;

/// Layout of the packed index key: 16 bits window, 16 bits cycle, 32 bits log-m/z bin.
const WINDOW_SHIFT: u32 = 48;
const CYCLE_SHIFT: u32 = 32;
const MAX_WINDOWS: usize = 1 << 16;
const MAX_CYCLES: usize = 1 << 16;
/// `ln` is undefined at zero; no real fragment is lighter than this.
const MIN_MZ: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct ContextParameters {
    /// Fragment m/z tolerance in ppm.
    pub mass_tolerance: f32,
    pub top_k_fragments: usize,
    pub min_shared: usize,
    pub cycle_radius: usize,
}

impl ContextParameters {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.mass_tolerance.is_finite() && self.mass_tolerance > 0.0) {
            return Err(format!(
                "mass_tolerance must be a positive number, got {}",
                self.mass_tolerance
            ));
        }
        if self.top_k_fragments == 0 {
            return Err("top_k_fragments must be at least 1".to_string());
        }
        if self.min_shared == 0 {
            return Err("min_shared must be at least 1".to_string());
        }
        Ok(())
    }
}

/// The context features of one candidate. A candidate without a window, without library
/// fragments or without any other candidate nearby has `claimant_rank` 1 and zero elsewhere.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContextFeatures {
    pub precursor_idx: usize,
    pub rank: usize,
    pub candidate_density: f32,
    pub n_competitors: f32,
    pub n_competitors_higher: f32,
    pub claimant_rank: f32,
    pub shared_frac_any: f32,
    pub shared_frac_higher: f32,
    pub shared_lib_intensity_frac_higher: f32,
    pub competitor_log_ratio: f32,
}

impl ContextFeatures {
    /// The feature values in the order of [`CONTEXT_FEATURE_NAMES`].
    pub fn values(&self) -> [f32; CONTEXT_FEATURE_NAMES.len()] {
        [
            self.candidate_density,
            self.n_competitors,
            self.n_competitors_higher,
            self.claimant_rank,
            self.shared_frac_any,
            self.shared_frac_higher,
            self.shared_lib_intensity_frac_higher,
            self.competitor_log_ratio,
        ]
    }
}

/// Width of one log-m/z bin. It is twice the tolerance, so two m/z values within the tolerance
/// differ in `ln(m/z)` by less than one bin and therefore fall into the same or an adjacent bin.
pub fn log_mz_bin_width(mass_tolerance_ppm: f32) -> f64 {
    (1.0 + 2.0 * mass_tolerance_ppm as f64 * 1e-6).ln()
}

pub fn log_mz_bin(mz: f32, bin_width: f64) -> u32 {
    ((mz as f64).max(MIN_MZ).ln() / bin_width).floor() as u32
}

/// Packs window, cycle and bin into one key whose order is the lexicographic order of
/// `(window, cycle, bin)`. Consecutive bins of one window and cycle have consecutive keys, so
/// a bin range is one contiguous key range.
pub fn pack_key(window: u32, cycle: u32, bin: u32) -> u64 {
    debug_assert!((window as usize) < MAX_WINDOWS);
    debug_assert!((cycle as usize) < MAX_CYCLES);
    ((window as u64) << WINDOW_SHIFT) | ((cycle as u64) << CYCLE_SHIFT) | bin as u64
}

/// Number of values in `sorted` that lie in the inclusive range `[lo, hi]`.
fn count_in_range(sorted: &[u64], lo: u64, hi: u64) -> usize {
    let start = sorted.partition_point(|&key| key < lo);
    let end = sorted.partition_point(|&key| key <= hi);
    end.saturating_sub(start)
}

/// One fragment of one candidate. 16 bytes, so the index of a large run fits in memory.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    key: u64,
    candidate: u32,
    mz: f32,
}

struct FragmentIndex {
    /// Sorted by `key`.
    entries: Vec<IndexEntry>,
}

impl FragmentIndex {
    /// The entries whose key lies in the inclusive range `[lo, hi]`.
    fn range(&self, lo: u64, hi: u64) -> &[IndexEntry] {
        let start = self.entries.partition_point(|entry| entry.key < lo);
        let end = self.entries.partition_point(|entry| entry.key <= hi);
        &self.entries[start..end.max(start)]
    }
}

/// What the query phase needs to know about every other candidate.
struct CandidateSlot {
    precursor_idx: usize,
    /// The first isolation window that contains the precursor m/z, if any.
    window: Option<u32>,
    cycle: u32,
    score: f32,
}

struct RunIndex {
    slots: Vec<CandidateSlot>,
    fragments: FragmentIndex,
    /// One key `(window, cycle, 0)` per candidate with a window, sorted. Serves the density.
    positions: Vec<u64>,
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

fn build_index<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidates: &CandidateCollection,
    params: &ContextParameters,
) -> RunIndex {
    let bin_width = log_mz_bin_width(params.mass_tolerance);

    let (slots, entries_per_candidate): (Vec<CandidateSlot>, Vec<Vec<IndexEntry>>) = candidates
        .par_iter()
        .enumerate()
        .map(|(candidate_idx, candidate)| {
            let precursor = lib.get_precursor_by_idx_filtered(
                candidate.precursor_idx,
                true,
                true,
                params.top_k_fragments,
            );
            let window = precursor.as_ref().and_then(|precursor| {
                dia_data
                    .get_valid_observations(precursor.mz)
                    .first()
                    .map(|&window| window as u32)
            });
            let cycle = candidate.cycle_center as u32;
            let entries = match (&precursor, window) {
                (Some(precursor), Some(window)) => precursor
                    .fragment_mz
                    .iter()
                    .map(|&mz| IndexEntry {
                        key: pack_key(window, cycle, log_mz_bin(mz, bin_width)),
                        candidate: candidate_idx as u32,
                        mz,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let slot = CandidateSlot {
                precursor_idx: candidate.precursor_idx,
                window,
                cycle,
                score: candidate.score,
            };
            (slot, entries)
        })
        .unzip();

    let mut entries: Vec<IndexEntry> = entries_per_candidate.into_par_iter().flatten().collect();
    entries.par_sort_unstable_by_key(|entry| entry.key);

    let mut positions: Vec<u64> = slots
        .iter()
        .filter_map(|slot| slot.window.map(|window| pack_key(window, slot.cycle, 0)))
        .collect();
    positions.par_sort_unstable();

    RunIndex {
        slots,
        fragments: FragmentIndex { entries },
        positions,
    }
}

/// Counts the fragments this candidate shares with every other candidate.
fn increment(shared_counts: &mut Vec<(u32, usize)>, other: u32) {
    match shared_counts.iter_mut().find(|(idx, _)| *idx == other) {
        Some((_, count)) => *count += 1,
        None => shared_counts.push((other, 1)),
    }
}

fn candidate_features(
    candidate_idx: usize,
    candidate: &Candidate,
    index: &RunIndex,
    lib: &SpecLibFlat,
    params: &ContextParameters,
) -> ContextFeatures {
    // Rank 1 means that no higher-scoring competitor claims this candidate's fragments. This
    // holds for a candidate without a window or without fragments as well.
    let mut features = ContextFeatures {
        precursor_idx: candidate.precursor_idx,
        rank: candidate.rank,
        claimant_rank: 1.0,
        ..Default::default()
    };
    let slot = &index.slots[candidate_idx];
    let Some(window) = slot.window else {
        return features;
    };

    let radius = params.cycle_radius as u32;
    let cycle_lo = slot.cycle.saturating_sub(radius);
    let cycle_hi = (slot.cycle + radius).min(MAX_CYCLES as u32 - 1);

    let neighbours = count_in_range(
        &index.positions,
        pack_key(window, cycle_lo, 0),
        pack_key(window, cycle_hi, 0),
    );
    // The candidate itself is always one of the neighbours.
    features.candidate_density = neighbours.saturating_sub(1) as f32;

    let Some(precursor) = lib.get_precursor_by_idx_filtered(
        candidate.precursor_idx,
        true,
        true,
        params.top_k_fragments,
    ) else {
        return features;
    };
    let num_fragments = precursor.fragment_mz.len();
    if num_fragments == 0 {
        return features;
    }

    let bin_width = log_mz_bin_width(params.mass_tolerance);
    let mut matched_per_fragment: Vec<Vec<u32>> = Vec::with_capacity(num_fragments);
    let mut shared_counts: Vec<(u32, usize)> = Vec::new();

    for &mz in &precursor.fragment_mz {
        let tolerance = mz * params.mass_tolerance * 1e-6;
        let bin = log_mz_bin(mz, bin_width);
        let mut matched: Vec<u32> = Vec::new();
        for cycle in cycle_lo..=cycle_hi {
            let lo = pack_key(window, cycle, bin.saturating_sub(1));
            let hi = pack_key(window, cycle, bin.saturating_add(1));
            for entry in index.fragments.range(lo, hi) {
                let other = &index.slots[entry.candidate as usize];
                if other.precursor_idx == slot.precursor_idx {
                    continue;
                }
                if (entry.mz - mz).abs() <= tolerance {
                    matched.push(entry.candidate);
                }
            }
        }
        // Two ions of one other candidate can both lie within the tolerance; they are one match.
        matched.sort_unstable();
        matched.dedup();
        for &other in &matched {
            increment(&mut shared_counts, other);
        }
        matched_per_fragment.push(matched);
    }

    let mut n_competitors = 0usize;
    let mut higher: Vec<u32> = Vec::new();
    let mut best_competitor_score = f32::NEG_INFINITY;
    for &(other, count) in &shared_counts {
        if count < params.min_shared {
            continue;
        }
        n_competitors += 1;
        let other_score = index.slots[other as usize].score;
        best_competitor_score = best_competitor_score.max(other_score);
        if other_score > slot.score {
            higher.push(other);
        }
    }
    higher.sort_unstable();

    let mut n_matched_any = 0usize;
    let mut n_claimed_higher = 0usize;
    let mut intensity_claimed_higher = 0.0f32;
    let mut intensity_total = 0.0f32;
    for (matched, &intensity) in matched_per_fragment
        .iter()
        .zip(&precursor.fragment_intensity)
    {
        intensity_total += intensity;
        if !matched.is_empty() {
            n_matched_any += 1;
        }
        if matched
            .iter()
            .any(|other| higher.binary_search(other).is_ok())
        {
            n_claimed_higher += 1;
            intensity_claimed_higher += intensity;
        }
    }

    features.n_competitors = n_competitors as f32;
    features.n_competitors_higher = higher.len() as f32;
    features.claimant_rank = 1.0 + higher.len() as f32;
    features.shared_frac_any = n_matched_any as f32 / num_fragments as f32;
    features.shared_frac_higher = n_claimed_higher as f32 / num_fragments as f32;
    features.shared_lib_intensity_frac_higher = if intensity_total > 0.0 {
        intensity_claimed_higher / intensity_total
    } else {
        0.0
    };
    features.competitor_log_ratio =
        if n_competitors > 0 && best_competitor_score > 0.0 && slot.score > 0.0 {
            (best_competitor_score / slot.score).ln()
        } else {
            0.0
        };
    features
}

/// Computes the context features of every candidate, in the candidate order of the caller.
pub fn compute_context_features<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidates: &CandidateCollection,
    params: &ContextParameters,
) -> Result<Vec<ContextFeatures>, String> {
    params.validate()?;
    validate_sizes(dia_data, candidates)?;
    let start_time = Instant::now();

    let index = build_index(dia_data, lib, candidates, params);
    let features: Vec<ContextFeatures> = candidates
        .par_iter()
        .enumerate()
        .map(|(candidate_idx, candidate)| {
            candidate_features(candidate_idx, candidate, &index, lib, params)
        })
        .collect();

    println!(
        "Computed context features for {} candidates ({} indexed fragments) in {:.2}s",
        candidates.len(),
        index.fragments.entries.len(),
        start_time.elapsed().as_secs_f64()
    );
    Ok(features)
}
