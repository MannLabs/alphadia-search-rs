//! Least-squares projection of every fragment onto the consensus elution profile.
//!
//! All fragments of a precursor elute with the same chromatographic profile, so a fragment
//! trace carries only two unknowns: how much of the peptide it accounts for, and what
//! background it sits on. Writing the consensus profile as `p(t)` and the extracted trace
//! as `I_i(t)`, the model is
//!
//! ```text
//! I_i(t) = a_i · p(t) + b_i + noise
//! ```
//!
//! and the reported area is the peak alone:
//!
//! ```text
//! a_i    = Σ_c w_c p_c (I_ic - b_i) / Σ_c w_c p_c²
//! area_i = a_i · ∫ p(t) dt
//! ```
//!
//! The amplitude is the matched filter for the problem and, for independent noise of equal
//! variance, the minimum-variance unbiased estimator — strictly better than integrating the
//! trace directly, which spends variance on cycles the elution profile says carry no signal.
//! A trace that follows the profile exactly reproduces the trapezoidal area of that trace,
//! so the numbers stay on the same scale as the other strategies while being far less
//! sensitive to what sits in the flanks of the window.
//!
//! # Background
//!
//! `b_i` is estimated as the median of `I_ic - a_i p_c` over the flank cycles — those where
//! the elution profile has decayed below [`TEMPLATE_SUPPORT_FRACTION`] of its apex.
//!
//! Dropping the background term — set `projection_fit_baseline` to false — leaves the
//! amplitude inflated by `b · Σp / Σp²`, so the area is inflated by that times `∫p` rather
//! than by `b` times the full window width. That is already a substantial improvement over
//! integrating the raw trace, and it shrinks further the less of the window the peak
//! occupies, but it is not zero.
//!
//! Fitting `b_i` jointly with `a_i` by least squares, which would be the obvious thing to
//! do, fails badly here: a single co-eluting species in one flank cycle drags both
//! parameters at once, and the resulting residuals are so large everywhere that the robust
//! reweighting below cannot tell which cycle was the culprit. A median over the flanks has
//! no such failure mode — it ignores a minority of contaminated flank cycles outright, and
//! it still recovers a genuine flat background exactly. Where the peak fills the whole
//! window there are no flank cycles and no background is subtracted, which is the safe
//! answer rather than a guess.
//!
//! # Interference
//!
//! On-peak interference cannot be ignored, only bounded. A co-eluting species that leaks
//! into a fragment m/z inflates a few cycles, and plain least squares would spread that
//! inflation into the amplitude. Iteratively reweighted least squares with Huber weights
//! caps the influence of any single cycle at `huber_k` robust scale units:
//!
//! ```text
//! w_c = min(1, huber_k · s / |I_ic - a_i p_c - b_i|),   s = 1.4826 · median(|residual|)
//! ```
//!
//! The scale is taken over the peak itself, not the flanks — flank residuals do not
//! influence the amplitude, so letting them inflate `s` would only mask interference where
//! it matters.
//!
//! Note that DIA noise is closer to Poisson than to constant variance, so the weighting is
//! an approximation; the shape prior, not the noise model, does most of the work here.

use super::{integrate_full_window, trapezoid, IntegrationContext};
use crate::peak_group_quantification::parameters::QuantificationParameters;

/// Consistency factor that turns a median absolute deviation into a standard-deviation
/// equivalent for normally distributed residuals.
const MAD_TO_SIGMA: f32 = 1.4826;

/// Cycles where the consensus profile falls below this fraction of its apex are treated as
/// flank: they carry no weight in the amplitude, they are where the background is measured,
/// and they are excluded from the robust scale estimate.
const TEMPLATE_SUPPORT_FRACTION: f32 = 0.05;

/// A background estimated from one flank cycle is that cycle's noise, not a background.
const MIN_FLANK_CYCLES: usize = 2;

/// Iteration stops once the robust scale drops to this fraction of the amplitude: the fit
/// is then exact to within numerical noise and further reweighting is meaningless.
const MIN_RELATIVE_SCALE: f32 = 1e-6;

pub fn integrate(ctx: &IntegrationContext, params: &QuantificationParameters) -> Vec<f32> {
    let template = normalize_to_apex(ctx.template);
    let template_area = trapezoid(ctx.rt, &template);

    // Without a positive template area there is no shape to project onto.
    if template_area <= 0.0 {
        return integrate_full_window(ctx);
    }

    let support = support_mask(&template);
    let mut workspace = Workspace::new(ctx.n_cycles());

    (0..ctx.n_fragments())
        .map(|fragment_idx| {
            let profile = ctx.xic.row(fragment_idx);
            let amplitude = robust_amplitude(
                &template,
                profile.as_slice().expect("XIC rows are contiguous"),
                &support,
                params,
                &mut workspace,
            );
            (amplitude * template_area).max(0.0)
        })
        .collect()
}

/// Scratch buffers reused across the fragments of one candidate.
///
/// Every fragment runs the same handful of reweighting passes over the same eleven-odd
/// cycles, so allocating per fragment — let alone per pass — dominates the cost of the fit
/// itself by an order of magnitude.
pub struct Workspace {
    residuals: Vec<f32>,
    weights: Vec<f32>,
    scratch: Vec<f32>,
}

impl Workspace {
    pub fn new(n_cycles: usize) -> Self {
        Self {
            residuals: vec![0.0; n_cycles],
            weights: vec![0.0; n_cycles],
            scratch: vec![0.0; n_cycles],
        }
    }
}

/// Scale a profile so that its largest value is one. Purely for numerical conditioning: the
/// reported area is invariant to the scale of the template because the amplitude absorbs it.
pub fn normalize_to_apex(template: &[f32]) -> Vec<f32> {
    let peak = template.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return template.to_vec();
    }
    template.iter().map(|&value| value / peak).collect()
}

/// Cycles that carry enough of the template to influence the amplitude. Its complement is
/// the flank.
pub fn support_mask(template: &[f32]) -> Vec<bool> {
    let peak = template.iter().copied().fold(0.0f32, f32::max);
    let threshold = peak * TEMPLATE_SUPPORT_FRACTION;
    template.iter().map(|&value| value > threshold).collect()
}

/// Amplitude of `template` in `observed`, above a background measured on the flanks and with
/// the influence of interfered cycles bounded by Huber reweighting.
///
/// `support` marks the peak cycles, as returned by [`support_mask`]. Zero
/// `robust_iterations` gives plain least squares over a background-corrected trace.
pub fn robust_amplitude(
    template: &[f32],
    observed: &[f32],
    support: &[bool],
    params: &QuantificationParameters,
    workspace: &mut Workspace,
) -> f32 {
    let n_cycles = template.len();
    if observed.len() != n_cycles || workspace.residuals.len() != n_cycles {
        return 0.0;
    }

    let mut background = 0.0f32;
    let mut amplitude = weighted_amplitude(template, observed, None, background);

    if params.projection_fit_baseline {
        background = estimate_background(template, observed, support, amplitude, workspace);
        amplitude = weighted_amplitude(template, observed, None, background);
    }

    for _ in 0..params.robust_iterations {
        if amplitude <= 0.0 {
            break;
        }

        for idx in 0..n_cycles {
            workspace.residuals[idx] =
                (observed[idx] - amplitude * template[idx] - background).abs();
        }

        let scale = robust_scale(&workspace.residuals, support, &mut workspace.scratch);
        if scale <= amplitude * MIN_RELATIVE_SCALE {
            break;
        }

        let cutoff = params.huber_k * scale;
        for idx in 0..n_cycles {
            let residual = workspace.residuals[idx];
            workspace.weights[idx] = if residual <= cutoff {
                1.0
            } else {
                cutoff / residual
            };
        }

        if params.projection_fit_baseline {
            background = estimate_background(template, observed, support, amplitude, workspace);
        }
        amplitude = weighted_amplitude(template, observed, Some(&workspace.weights), background);
    }

    amplitude
}

/// Weighted least-squares amplitude `Σ w p (I - b) / Σ w p²`, accumulated in double
/// precision because intensities span several orders of magnitude within one window.
fn weighted_amplitude(
    template: &[f32],
    observed: &[f32],
    weights: Option<&[f32]>,
    background: f32,
) -> f32 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for idx in 0..template.len() {
        let weight = weights.map_or(1.0f64, |w| w[idx] as f64);
        let expected = template[idx] as f64;
        numerator += weight * expected * (observed[idx] - background) as f64;
        denominator += weight * expected * expected;
    }

    if denominator <= 0.0 {
        return 0.0;
    }
    (numerator / denominator) as f32
}

/// Median of `I - a p` over the flank cycles, clamped at zero.
///
/// Subtracting the modelled peak first matters: the tails of a real peak reach into the
/// flanks, and ignoring them would charge that signal to the background.
fn estimate_background(
    template: &[f32],
    observed: &[f32],
    support: &[bool],
    amplitude: f32,
    workspace: &mut Workspace,
) -> f32 {
    let mut count = 0;
    for (idx, &in_support) in support.iter().enumerate() {
        if !in_support {
            workspace.scratch[count] = observed[idx] - amplitude * template[idx];
            count += 1;
        }
    }

    if count < MIN_FLANK_CYCLES {
        return 0.0;
    }
    median_in_place(&mut workspace.scratch[..count]).max(0.0)
}

/// Robust scale of the residuals over the peak: the median absolute residual about zero,
/// scaled to a standard-deviation equivalent. Falls back to all cycles when the peak covers
/// none of them.
fn robust_scale(residuals: &[f32], support: &[bool], scratch: &mut [f32]) -> f32 {
    let mut count = 0;
    for (idx, &in_support) in support.iter().enumerate() {
        if in_support {
            scratch[count] = residuals[idx];
            count += 1;
        }
    }

    if count == 0 {
        scratch[..residuals.len()].copy_from_slice(residuals);
        count = residuals.len();
    }

    MAD_TO_SIGMA * median_in_place(&mut scratch[..count])
}

/// Median of `values`, reordering them in place.
fn median_in_place(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}
