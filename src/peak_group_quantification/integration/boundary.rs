//! Consensus peak boundaries plus a base-to-base baseline, integrated with the
//! trapezoidal rule.
//!
//! The candidate window is a fixed number of cycles around the apex, so its flanks are
//! usually not peak: chemical background, detector offset and whatever co-eluting species
//! happen to share a fragment m/z. Summing the full window therefore adds a
//! precursor-dependent pedestal to every fragment area — and because that pedestal varies
//! between runs, it shows up as both bias at low abundance and run-to-run inconsistency.
//!
//! Boundaries are found on the consensus profile in two stages, following the scheme
//! Pioneer uses:
//!
//! 1. **Curvature.** Walk outwards from the apex to the first local maximum of the second
//!    derivative. That is the shoulder where the peak stops falling steeply and flattens
//!    into its surroundings — a much better boundary than any fixed fraction of the apex,
//!    because it adapts to the width of the peak instead of assuming one.
//! 2. **Valley.** Continue outwards while the profile keeps descending, tracking the
//!    running minimum, and stop once it climbs back above that minimum by more than
//!    `boundary_valley_tolerance`. The boundary snaps to the running minimum, so it lands
//!    in the trough between this peak and its neighbour rather than partway up the
//!    neighbour.
//!
//! Between those bounds each fragment is integrated after removing the straight line
//! through its own two boundary intensities:
//!
//! ```text
//! area_i = ∫[t_l, t_r] max(0, I_i(t) - b_i(t)) dt
//! b_i(t) = I_i(t_l) + (I_i(t_r) - I_i(t_l)) · (t - t_l) / (t_r - t_l)
//! ```
//!
//! Boundaries are shared by every fragment of the precursor. Per-fragment boundaries would
//! let a noisy fragment be integrated over a different retention time range than its
//! siblings, which distorts exactly the fragment ratios that label-free quantification
//! consumes.
//!
//! # Withholding rather than guessing
//!
//! A window centred on the shoulder of a much larger neighbouring peak has a baseline that
//! is nearly the whole signal, so almost nothing survives subtraction and whatever does is
//! mostly the neighbour. Pioneer withholds the quantity in that case rather than reporting
//! a number that is confidently wrong, and `min_area_surviving_ratio` enables the same
//! test here: it is evaluated once per precursor on the summed trace — never per fragment,
//! whose signal-to-noise is far too low for the ratio to mean anything — and zeroes the
//! whole peak group when the unsubtracted area exceeds the subtracted area by more than the
//! given factor.
//!
//! It ships **disabled**, because withholding is only safe once the consumer reads a zero
//! area as "not quantified in this run". Today a peak group with no positive fragment is
//! dropped by `filter_fragments_by_intensity`, so enabling the gate would discard the
//! identification along with the quantity. Pioneer's calibrated value is `5.0`.

use super::{trapezoid, IntegrationContext};
use crate::peak_group_quantification::parameters::QuantificationParameters;

/// The boundary search starts this many cycles out from the apex, which guarantees an
/// integration range of at least `2 * this + 1` cycles.
const BOUNDARY_SEARCH_OFFSET: usize = 2;

/// A peak is never integrated over fewer than this many cycles.
const MIN_POINTS_INTEGRATED: usize = 3;

pub fn integrate(ctx: &IntegrationContext, params: &QuantificationParameters) -> Vec<f32> {
    let apex = climb_to_apex(ctx.template, ctx.apex);
    let (left, right) =
        find_boundaries(ctx.template, ctx.rt, apex, params.boundary_valley_tolerance);

    let rt_window = &ctx.rt[left..=right];
    let n_window = rt_window.len();

    if params.subtract_baseline && withhold_peak_group(ctx, left, right, params) {
        return vec![0.0; ctx.n_fragments()];
    }

    let mut corrected = vec![0.0f32; n_window];

    (0..ctx.n_fragments())
        .map(|fragment_idx| {
            let profile = ctx.xic.row(fragment_idx);
            let raw = &profile.as_slice().expect("XIC rows are contiguous")[left..=right];

            if !params.subtract_baseline {
                return trapezoid(rt_window, raw);
            }

            subtract_base_to_base(raw, rt_window, &mut corrected);
            trapezoid(rt_window, &corrected)
        })
        .collect()
}

/// Hill-climb from the reported apex to the nearest local maximum in each direction and
/// keep the higher one.
///
/// The apex reported by selection is the maximum of a smoothed score over a kernel, which
/// can sit a cycle or two away from the maximum of the extracted profile. Every boundary
/// decision downstream is measured from the apex, so it is worth getting right.
fn climb_to_apex(template: &[f32], apex: usize) -> usize {
    let last = template.len() - 1;

    let mut right = apex.min(last);
    while right < last && template[right + 1] > template[right] {
        right += 1;
    }

    let mut left = apex.min(last);
    while left > 0 && template[left - 1] > template[left] {
        left -= 1;
    }

    if template[right] >= template[left] {
        right
    } else {
        left
    }
}

/// Inclusive `(left, right)` integration bounds, from curvature then valley walking.
fn find_boundaries(
    template: &[f32],
    rt: &[f32],
    apex: usize,
    valley_tolerance: f32,
) -> (usize, usize) {
    let last = template.len() - 1;
    let curvature = second_derivative(template, rt);

    let mut right = (apex + BOUNDARY_SEARCH_OFFSET).min(last);
    if let Some(shoulder) = first_local_maximum(&curvature, right, last, 1) {
        right = shoulder;
    }
    right = walk_to_valley(template, right, last, 1, valley_tolerance);

    let mut left = apex.saturating_sub(BOUNDARY_SEARCH_OFFSET);
    if let Some(shoulder) = first_local_maximum(&curvature, left, 0, -1) {
        left = shoulder;
    }
    left = walk_to_valley(template, left, 0, -1, valley_tolerance);

    widen_to_minimum_width(left, right, apex, last)
}

/// Second derivative on an unevenly spaced grid:
/// `u''ᵢ = 2/(tᵢ₊₁ - tᵢ₋₁) · [ (uᵢ₊₁ - uᵢ)/(tᵢ₊₁ - tᵢ) - (uᵢ - uᵢ₋₁)/(tᵢ - tᵢ₋₁) ]`.
///
/// The two edge cycles copy their neighbour, which is the value a boundary search would
/// have used there anyway.
fn second_derivative(values: &[f32], rt: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut curvature = vec![0.0f32; n];

    for idx in 1..n - 1 {
        let left_step = rt[idx] - rt[idx - 1];
        let right_step = rt[idx + 1] - rt[idx];
        let total = rt[idx + 1] - rt[idx - 1];

        if left_step <= 0.0 || right_step <= 0.0 || total <= 0.0 {
            continue;
        }

        let right_slope = (values[idx + 1] - values[idx]) / right_step;
        let left_slope = (values[idx] - values[idx - 1]) / left_step;
        curvature[idx] = 2.0 * (right_slope - left_slope) / total;
    }

    if n >= 3 {
        curvature[0] = curvature[1];
        curvature[n - 1] = curvature[n - 2];
    }
    curvature
}

/// First index strictly beyond `from` towards `limit` whose value exceeds both neighbours.
///
/// `step` is `+1` when walking right and `-1` when walking left.
fn first_local_maximum(values: &[f32], from: usize, limit: usize, step: isize) -> Option<usize> {
    let last = values.len() - 1;
    let mut idx = from;

    loop {
        if idx == limit || idx == 0 || idx == last {
            return None;
        }
        let next = (idx as isize + step) as usize;
        if values[idx] > values[idx - 1] && values[idx] > values[idx + 1] {
            return Some(idx);
        }
        idx = next;
    }
}

/// Continue outwards from `from`, tracking the running minimum of `values`, and stop once
/// the profile climbs more than `tolerance` times above it. Returns the index of the
/// running minimum — the bottom of the valley.
fn walk_to_valley(values: &[f32], from: usize, limit: usize, step: isize, tolerance: f32) -> usize {
    let mut minimum = values[from];
    let mut minimum_idx = from;
    let mut idx = from;

    loop {
        // Ties keep the earlier index, so a boundary never wanders out along a flat tail.
        if values[idx] < minimum {
            minimum = values[idx];
            minimum_idx = idx;
        }
        if idx == limit {
            break;
        }

        let next = (idx as isize + step) as usize;
        if values[next] > minimum * tolerance {
            break;
        }
        idx = next;
    }

    minimum_idx
}

/// Widen the bounds until at least [`MIN_POINTS_INTEGRATED`] cycles are covered and the
/// apex lies inside them, preferring whichever side has more room.
fn widen_to_minimum_width(
    mut left: usize,
    mut right: usize,
    apex: usize,
    last: usize,
) -> (usize, usize) {
    left = left.min(apex);
    right = right.max(apex);

    while right - left + 1 < MIN_POINTS_INTEGRATED {
        let can_extend_left = left > 0;
        let can_extend_right = right < last;

        if !can_extend_left && !can_extend_right {
            break;
        }
        if can_extend_right && (!can_extend_left || right - apex <= apex - left) {
            right += 1;
        } else {
            left -= 1;
        }
    }

    (left, right)
}

/// Write `max(0, intensity - baseline)` into `out`, where the baseline is the straight line
/// through the first and last point of the window.
fn subtract_base_to_base(intensity: &[f32], rt: &[f32], out: &mut [f32]) {
    let first = intensity[0];
    let last = intensity[intensity.len() - 1];
    let rt_span = rt[rt.len() - 1] - rt[0];

    // A degenerate retention time span leaves no line to subtract.
    let slope = if rt_span > 0.0 {
        (last - first) / rt_span
    } else {
        0.0
    };

    for (idx, target) in out.iter_mut().enumerate() {
        let baseline = first + slope * (rt[idx] - rt[0]);
        *target = (intensity[idx] - baseline).max(0.0);
    }
}

/// True when so little of the summed trace survives baseline subtraction that the area is
/// more likely to describe a neighbouring peak than this one.
///
/// Evaluated on the sum over fragments, which is the highest signal-to-noise view of the
/// peak group available here — the same ratio computed on an individual fragment would fire
/// on any genuinely low-abundance fragment sitting on chemical background.
fn withhold_peak_group(
    ctx: &IntegrationContext,
    left: usize,
    right: usize,
    params: &QuantificationParameters,
) -> bool {
    let ratio = params.min_area_surviving_ratio;
    if ratio <= 0.0 {
        return false;
    }

    let rt_window = &ctx.rt[left..=right];
    let summed: Vec<f32> = (left..=right)
        .map(|cycle_idx| ctx.xic.column(cycle_idx).sum())
        .collect();

    let mut corrected = vec![0.0f32; summed.len()];
    subtract_base_to_base(&summed, rt_window, &mut corrected);

    let subtracted = trapezoid(rt_window, &corrected);
    let unsubtracted = trapezoid(rt_window, &summed);

    subtracted > 0.0 && unsubtracted >= ratio * subtracted
}
