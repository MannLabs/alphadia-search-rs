//! Per-scan part of the MS1 block: MS1 at the MS1 scans' own times.
//!
//! The cycle-binned MS1 XIC sums every MS1 scan of a cycle into one point. When a cycle holds several MS1 scans (Astral:
//! ~3 per 1.5 s cycle of 300 windows), that throws away the time resolution in which a low-input peak is sampled best:
//! the precursor's MS2 window holds only 2-4 points of the peak, its MS1 trace 6-12. Here every MS1 observation keeps
//! its own point, at the cycle's RT plus its position within the cycle, and the fragment profile is placed at the time
//! of the precursor's own MS2 window, so MS1 and MS2 are compared at the times they were measured.
//!
//! The positions within the cycle come from `DIADataTrait::cycle_position`: the observation's index over the cycle
//! length, or the positions alphaDIA sets for MS1 scans it aligned to the MS2 cycle (their acquisition times).
//! The MS1 block keeps the two per-scan features that the feature selection kept: the shape of the M0 points against
//! the fragment peak and the M0 / M+1 point correlation.
//! A candidate without library isotopes or a run without MS1 scans gets 0; all values are finite, never NaN.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::ms1::Ms1Features;
use crate::peak_group_scoring::utils::{calculate_correlation_safe, finite_or_zero};

/// FWHM to Gaussian standard deviation.
const FWHM_TO_SIGMA: f32 = 1.0 / 2.354_82;
/// The expected MS1 peak is never narrower than this many cycles: the fragment FWHM of a 2-4 point peak is coarse.
const MIN_SIGMA_CYCLES: f32 = 0.5;

/// Apex time of `values` sampled at the evenly spaced `times` (spacing `step`), refined by a parabola through the
/// maximum and its neighbours.
fn refined_apex(times: &[f32], values: &[f32], step: f32) -> Option<f32> {
    let (apex, &top) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if top <= 0.0 {
        return None;
    }
    if apex == 0 || apex + 1 == values.len() {
        return Some(times[apex]);
    }
    let (left, right) = (values[apex - 1], values[apex + 1]);
    let curvature = left - 2.0 * top + right;
    let offset = if curvature < 0.0 {
        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some(times[apex] + offset * step)
}

/// Sets `ms1_fine_shape` and `ms1_fine_iso_corr` from per-observation M0 / M1 XICs; leaves them unchanged if the
/// inputs do not fit or the fragment profile has no signal.
///
/// `m0`, `m1`: [MS1 observation, cycle] over the candidate's cycles; `ms1_position`: each MS1 observation's position
/// within the cycle as a fraction of the cycle (0..1); `cycle_rt`: RT of the candidate's cycles; `fragment_position`:
/// the precursor's MS2 window position within the cycle (fraction); `fragment_profile`: the median fragment profile
/// over the same cycles; `fwhm`: its FWHM in RT units.
#[allow(clippy::too_many_arguments)]
pub fn set_fine_features(
    features: &mut Ms1Features,
    m0: &Array2<f32>,
    m1: &Array2<f32>,
    ms1_position: &[f32],
    cycle_rt: &[f32],
    fragment_position: f32,
    fragment_profile: &[f32],
    fwhm: f32,
) {
    let n_cycles = cycle_rt.len();
    let n_obs = ms1_position.len();
    if n_cycles < 2
        || n_obs == 0
        || m0.dim() != (n_obs, n_cycles)
        || m1.dim() != (n_obs, n_cycles)
        || fragment_profile.len() != n_cycles
    {
        return;
    }
    let cycle_time = (cycle_rt[n_cycles - 1] - cycle_rt[0]) / (n_cycles - 1) as f32;
    if cycle_time <= 0.0 {
        return;
    }

    let fragment_times: Vec<f32> = cycle_rt
        .iter()
        .map(|&t| t + fragment_position * cycle_time)
        .collect();
    let Some(fragment_apex) = refined_apex(&fragment_times, fragment_profile, cycle_time) else {
        return;
    };

    let mut order: Vec<usize> = (0..n_obs).collect();
    order.sort_by(|&a, &b| ms1_position[a].total_cmp(&ms1_position[b]));
    let mut times = Vec::with_capacity(n_obs * n_cycles);
    let mut m0_points = Vec::with_capacity(n_obs * n_cycles);
    let mut m1_points = Vec::with_capacity(n_obs * n_cycles);
    for c in 0..n_cycles {
        for &j in &order {
            times.push(cycle_rt[c] + ms1_position[j] * cycle_time);
            m0_points.push(m0[[j, c]]);
            m1_points.push(m1[[j, c]]);
        }
    }

    let sigma = (fwhm * FWHM_TO_SIGMA).max(MIN_SIGMA_CYCLES * cycle_time);
    let expected: Vec<f32> = times
        .iter()
        .map(|&t| (-0.5 * ((t - fragment_apex) / sigma).powi(2)).exp())
        .collect();

    features.ms1_fine_shape = finite_or_zero(calculate_correlation_safe(&m0_points, &expected));
    features.ms1_fine_iso_corr = finite_or_zero(calculate_correlation_safe(&m0_points, &m1_points));
}
