//! How much of its isolation window's MS2 signal a candidate carries at its apex, and how that compares with the
//! other candidates of the same window.
//!
//! "Window" is the candidate's window group (`crate::window_groups`): the quadrupole observations its XICs are read
//! from. The share divides the XIC signal by the MS2 signal of exactly these quadrupole observations, and the
//! candidates it is compared with have the same window group, so all shares of a comparison have the same
//! denominator.
//!
//! In DIA every MS2 spectrum is chimeric, so the share alone mostly measures how crowded the window is. A real
//! precursor still carries a large part of what its window records at its apex, while a wrong candidate that sits on
//! another precursor's peak claims only a small part of it. The rank and z-score compare the share with the
//! candidates of the same window group whose apex lies within WINDOW_SHARE_CYCLE_RADIUS cycles, which corrects for
//! the crowding. They need all candidates, so they are set after the per-candidate scoring.
//!
//! All values are finite, never NaN, because NaN rows are dropped before the classifier.

use numpy::ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::peak_group_scoring::feature_block::feature_block;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::window_groups::WindowGroups;

/// Candidates of the same isolation window whose apex lies within this many cycles are compared for the rank and
/// z-score of the window share.
pub const WINDOW_SHARE_CYCLE_RADIUS: u32 = 2;

feature_block! {
    /// The window-share features of one candidate.
    pub struct WindowShareFeatures {
        /// share of the isolation window's MS2 intensity at the apex cycle carried by the candidate's fragment XICs
        pub apex_window_signal_frac: f32,
        /// rank of the share among the candidates of the window group within WINDOW_SHARE_CYCLE_RADIUS apex cycles
        /// (ties count half, 0.5 if alone)
        pub apex_window_signal_frac_rank: f32,
        /// z-score of the share among the same candidates, 0 if they all have the same share
        pub apex_window_signal_frac_zscore: f32,
    }
}

/// What the cross-candidate step needs of one candidate: the window group and apex cycle it is compared in, and its
/// share. u32 keeps it at 12 bytes, since it is held for every candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowShareEntry {
    pub window_group: u32,
    pub apex_cycle: u32,
    pub frac: f32,
}

/// The window-share entry of one candidate.
///
/// * `dense_xic` - fragment XICs, fragments x cycles, starting at cycle `cycle_start`
/// * `median_profile` - the candidate's median elution profile over the XICs with signal
/// * `cycle_center` - the candidate's cycle center, the apex if the median profile has no signal
/// * `window_group` - the window group the XICs were read from
/// * `window_intensities` - summed MS2 intensity of that window group per cycle (`window_group_cycle_intensities`)
pub fn window_share_entry(
    dense_xic: &Array2<f32>,
    median_profile: &[f32],
    cycle_start: usize,
    cycle_center: usize,
    window_group: u32,
    window_intensities: &[f32],
) -> WindowShareEntry {
    let apex = apex_column(median_profile, cycle_center.saturating_sub(cycle_start));
    let apex_cycle = cycle_start + apex;
    let window_intensity = window_intensities.get(apex_cycle).copied().unwrap_or(0.0);
    WindowShareEntry {
        window_group,
        apex_cycle: apex_cycle as u32,
        frac: finite_or_zero(apex_window_signal_frac(dense_xic, apex, window_intensity)),
    }
}

/// Summed MS2 intensity per window group and cycle: the sum of `cycle_intensity_sums` over the quadrupole
/// observations of each group. Empty for the group outside all isolation windows.
pub fn window_group_cycle_intensities(
    quadrupole_observations: &[QuadrupoleObservation],
    window_groups: &WindowGroups,
) -> Vec<Vec<f32>> {
    let per_observation: Vec<Vec<f32>> = quadrupole_observations
        .par_iter()
        .map(cycle_intensity_sums)
        .collect();
    (0..window_groups.num_groups() as u32)
        .map(|group| {
            let mut sums: Vec<f32> = Vec::new();
            for &idx in window_groups.quadrupole_observations(group) {
                let observation_sums = &per_observation[idx];
                if sums.len() < observation_sums.len() {
                    sums.resize(observation_sums.len(), 0.0);
                }
                for (sum, &intensity) in sums.iter_mut().zip(observation_sums) {
                    *sum += intensity;
                }
            }
            sums
        })
        .collect()
}

/// Summed intensity of all peaks of one quadrupole observation per cycle (length = number of cycles).
pub fn cycle_intensity_sums(quadrupole_observation: &QuadrupoleObservation) -> Vec<f32> {
    let mut sums = vec![0.0; quadrupole_observation.num_cycles];
    for (&cycle, &intensity) in quadrupole_observation
        .cycle_indices
        .iter()
        .zip(&quadrupole_observation.intensities)
    {
        if let Some(sum) = sums.get_mut(cycle as usize) {
            *sum += intensity;
        }
    }
    sums
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

/// The window shares of the candidates of one window group and apex cycle, with their sum and sum of squares.
struct WindowShareBucket {
    sorted: Vec<f32>,
    sum: f64,
    sum_sq: f64,
}

/// The window-share features of every candidate, in the order of `entries`.
///
/// Each candidate is compared to all candidates (itself included) with the same window group and
/// `|apex_cycle - own apex_cycle| <= radius`:
/// - rank: share of the other candidates with a lower fraction (ties count half), 0.5 if alone
/// - z-score of its fraction among them, 0 if they all have the same fraction
pub fn window_share_features(
    entries: &[WindowShareEntry],
    radius: u32,
) -> Vec<WindowShareFeatures> {
    let mut buckets: HashMap<(u32, u32), WindowShareBucket> = HashMap::new();
    for entry in entries {
        let bucket = buckets
            .entry((entry.window_group, entry.apex_cycle))
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
        .par_iter()
        .map(|entry| {
            let frac = entry.frac;
            let (mut n, mut lower, mut equal) = (0usize, 0usize, 0usize);
            let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
            for c in
                entry.apex_cycle.saturating_sub(radius)..=entry.apex_cycle.saturating_add(radius)
            {
                if let Some(bucket) = buckets.get(&(entry.window_group, c)) {
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
            WindowShareFeatures {
                apex_window_signal_frac: frac,
                apex_window_signal_frac_rank: finite_or_zero(rank),
                apex_window_signal_frac_zscore: finite_or_zero(zscore),
            }
        })
        .collect()
}
