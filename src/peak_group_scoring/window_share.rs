//! How much of its isolation window's MS2 signal a candidate carries at its apex, and how that compares with the
//! other candidates of the same window.
//!
//! In DIA every MS2 spectrum is chimeric, so the share alone mostly measures how crowded the window is. A real
//! precursor still carries a large part of what its window records at its apex, while a wrong candidate that sits on
//! another precursor's peak claims only a small part of it. The rank and z-score compare the share with the
//! candidates of the same window whose apex lies within WINDOW_SHARE_CYCLE_RADIUS cycles, which corrects for the
//! crowding. They need all candidates, so they are set after the per-candidate scoring.
//!
//! All values are finite, never NaN, because NaN rows are dropped before the classifier.

use numpy::ndarray::Array2;
use std::collections::HashMap;

use crate::peak_group_scoring::feature_block::feature_block;

/// Candidates of the same isolation window whose apex lies within this many cycles are compared for the rank and
/// z-score of the window share.
pub const WINDOW_SHARE_CYCLE_RADIUS: usize = 2;

feature_block! {
    /// The window-share features of one candidate.
    pub struct WindowShareFeatures {
        /// share of the isolation window's MS2 intensity at the apex cycle carried by the candidate's fragment XICs
        pub apex_window_signal_frac: f32,
        /// rank of the share among the candidates of the window within WINDOW_SHARE_CYCLE_RADIUS apex cycles (ties
        /// count half, 0.5 if alone)
        pub apex_window_signal_frac_rank: f32,
        /// z-score of the share among the same candidates, 0 if they all have the same share
        pub apex_window_signal_frac_zscore: f32,
    }
}

/// Where a candidate's window share is compared: its isolation window, apex cycle and share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowShareEntry {
    pub window: usize,
    pub apex_cycle: usize,
    pub frac: f32,
}

/// The window-share features of one candidate without the window context (rank and z-score are 0; they need all
/// candidates and are set by `PeakGroupScoring::set_window_share_context`), and the entry to compare its share by.
///
/// * `dense_xic` - fragment XICs, fragments x cycles, starting at cycle `cycle_start`
/// * `median_profile` - the candidate's median elution profile over the XICs with signal
/// * `cycle_center` - the candidate's cycle center, the apex if the median profile has no signal
/// * `contributing_obs_indices` - the isolation windows the XICs were extracted from; the first identifies the
///   window for the comparison
/// * `window_intensities` - summed MS2 intensity per isolation window and cycle
pub fn calculate_window_share_features(
    dense_xic: &Array2<f32>,
    median_profile: &[f32],
    cycle_start: usize,
    cycle_center: usize,
    contributing_obs_indices: &[usize],
    window_intensities: &[Vec<f32>],
) -> (WindowShareFeatures, WindowShareEntry) {
    let apex = apex_column(median_profile, cycle_center.saturating_sub(cycle_start));
    let apex_cycle = cycle_start + apex;
    let window_intensity: f32 = contributing_obs_indices
        .iter()
        .filter_map(|&obs| window_intensities.get(obs)?.get(apex_cycle))
        .sum();
    let frac = finite_or_zero(apex_window_signal_frac(dense_xic, apex, window_intensity));
    let entry = WindowShareEntry {
        window: contributing_obs_indices
            .first()
            .copied()
            .unwrap_or(usize::MAX),
        apex_cycle,
        frac,
    };
    let features = WindowShareFeatures {
        apex_window_signal_frac: frac,
        ..Default::default()
    };
    (features, entry)
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Apex of the candidate (index into the XIC columns): the maximum of the median profile, or `fallback` if the
/// profile has no signal.
pub fn apex_column(median_profile: &[f32], fallback: usize) -> usize {
    median_profile
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(fallback, |(i, _)| i)
}

/// Share of the isolation window's MS2 intensity at the apex that the candidate's fragments carry: the summed
/// fragment XICs in column `apex` divided by `window_intensity`, 0 without signal.
pub fn apex_window_signal_frac(xic: &Array2<f32>, apex: usize, window_intensity: f32) -> f32 {
    if apex >= xic.shape()[1] || window_intensity <= 0.0 {
        return 0.0;
    }
    xic.column(apex).sum() / window_intensity
}

/// The window shares of the candidates of one isolation window and apex cycle, with their sum and sum of squares.
struct WindowShareBucket {
    sorted: Vec<f32>,
    sum: f64,
    sum_sq: f64,
}

/// Where each candidate's window share stands among the candidates of the same window and nearby apex.
///
/// Each candidate is compared to all candidates (itself included) with the same window and
/// `|apex_cycle - own apex_cycle| <= radius`. Returns per candidate:
/// - rank: share of the other candidates with a lower fraction (ties count half), 0.5 if alone
/// - z-score of its fraction among them, 0 if they all have the same fraction
pub fn window_share_context(entries: &[WindowShareEntry], radius: usize) -> Vec<(f32, f32)> {
    let mut buckets: HashMap<(usize, usize), WindowShareBucket> = HashMap::new();
    for entry in entries {
        let bucket = buckets
            .entry((entry.window, entry.apex_cycle))
            .or_insert(WindowShareBucket {
                sorted: Vec::new(),
                sum: 0.0,
                sum_sq: 0.0,
            });
        bucket.sorted.push(entry.frac);
        bucket.sum += entry.frac as f64;
        bucket.sum_sq += (entry.frac as f64) * (entry.frac as f64);
    }
    for bucket in buckets.values_mut() {
        bucket.sorted.sort_by(|a, b| a.total_cmp(b));
    }

    entries
        .iter()
        .map(|entry| {
            let frac = entry.frac;
            let (mut n, mut lower, mut equal) = (0usize, 0usize, 0usize);
            let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
            for c in entry.apex_cycle.saturating_sub(radius)..=entry.apex_cycle + radius {
                if let Some(bucket) = buckets.get(&(entry.window, c)) {
                    let below = bucket.sorted.partition_point(|&v| v < frac);
                    let not_above = bucket.sorted.partition_point(|&v| v <= frac);
                    n += bucket.sorted.len();
                    lower += below;
                    equal += not_above - below;
                    sum += bucket.sum;
                    sum_sq += bucket.sum_sq;
                }
            }
            // the candidate itself is one of the equal values
            let others = n.saturating_sub(1);
            let rank = if others == 0 {
                0.5
            } else {
                (lower as f32 + 0.5 * equal.saturating_sub(1) as f32) / others as f32
            };
            let mean = sum / n.max(1) as f64;
            let std = (sum_sq / n.max(1) as f64 - mean * mean).max(0.0).sqrt();
            let zscore = if std > 1e-12 {
                ((frac as f64 - mean) / std) as f32
            } else {
                0.0
            };
            (finite_or_zero(rank), finite_or_zero(zscore))
        })
        .collect()
}
