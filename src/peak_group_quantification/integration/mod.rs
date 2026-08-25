//! Strategies that reduce a dense XIC window to a single intensity per fragment.
//!
//! The window handed to quantification is a fixed number of cycles centred on the apex
//! found during selection — `peak_length` cycles on either side. Summing every cycle in
//! that window, the historical [`QuantificationMethod::Sum`], therefore adds baseline,
//! chemical background and whatever co-eluting species share a fragment m/z to every
//! fragment area. The strategies here differ in how they decide *which* part of the window
//! carries peptide signal, and in how much structure they assume while doing so:
//!
//! | method | assumption | recovers |
//! |---|---|---|
//! | [`Trapezoid`](QuantificationMethod::Trapezoid) | none | correct units under uneven cycle timing |
//! | [`BoundaryTrapezoid`](QuantificationMethod::BoundaryTrapezoid) | the peak is contiguous around the apex | baseline and flanking interference |
//! | [`ProfileProjection`](QuantificationMethod::ProfileProjection) | fragments co-elute | noise variance, on-peak interference |
//! | [`EmgFit`](QuantificationMethod::EmgFit) | peaks are exponentially modified Gaussians | coarse-sampling bias, truncated tails |
//!
//! # Comparability
//!
//! Every retention-time aware strategy reports an area in intensity × seconds, and each is
//! constructed so that a noise-free trace which matches its model exactly reproduces the
//! plain trapezoidal area of that trace — up to the finer integration grid in the case of
//! [`EmgFit`]. The four are therefore directly comparable and differ only in how they
//! suppress baseline, noise and interference. [`Sum`](QuantificationMethod::Sum) is the
//! exception: it reports intensity × cycles and remains the default so that existing
//! searches are unaffected.
//!
//! # One elution profile per precursor
//!
//! All three new strategies share one consensus elution profile across the fragments of a
//! precursor rather than making per-fragment decisions. That is deliberate: label-free
//! quantification consumes fragment *ratios*, and letting a noisy fragment pick its own
//! integration bounds or its own peak shape distorts exactly those ratios. Fragments differ
//! in one number only — their amplitude.

use numpy::ndarray::{Array2, Axis};

use super::parameters::{QuantificationMethod, QuantificationParameters};

pub mod boundary;
pub mod emg;
pub mod projection;
pub mod smoothing;
pub mod template;

#[cfg(test)]
mod tests;

/// Below this many cycles no area over retention time can be formed.
pub const MIN_CYCLES_FOR_INTEGRATION: usize = 2;

/// Everything an integration strategy needs to turn one candidate's dense XIC into one
/// area per fragment.
pub struct IntegrationContext<'a> {
    /// Raw intensities of the candidate window, shape `[n_fragments, n_cycles]`.
    pub xic: &'a Array2<f32>,

    /// Retention time of every cycle in the window, in seconds. Length `n_cycles`.
    pub rt: &'a [f32],

    /// Consensus elution profile shared by all fragments of the precursor, obtained as the
    /// median of the apex-normalised fragment profiles. Length `n_cycles`.
    pub template: &'a [f32],

    /// Index of the apex cycle *within the window*.
    pub apex: usize,
}

impl<'a> IntegrationContext<'a> {
    pub fn n_fragments(&self) -> usize {
        self.xic.nrows()
    }

    pub fn n_cycles(&self) -> usize {
        self.rt.len()
    }

    /// True when the window cannot support a retention-time aware strategy: fewer than two
    /// cycles, inconsistent shapes, an apex outside the window, or a consensus profile
    /// without a single positive value to anchor a peak on.
    fn is_degenerate(&self) -> bool {
        self.n_cycles() < MIN_CYCLES_FOR_INTEGRATION
            || self.xic.ncols() != self.n_cycles()
            || self.template.len() != self.n_cycles()
            || self.apex >= self.n_cycles()
            || !self.template.iter().any(|&value| value > 0.0)
    }
}

/// Reduce the dense XIC of one candidate to a single intensity per fragment.
pub fn integrate(ctx: &IntegrationContext, params: &QuantificationParameters) -> Vec<f32> {
    // Degenerate windows fall back to the plain sum rather than silently reporting zero
    // for every fragment, which would drop the precursor from the results entirely.
    if matches!(params.method, QuantificationMethod::Sum) || ctx.is_degenerate() {
        return ctx.xic.sum_axis(Axis(1)).to_vec();
    }

    // Every strategy but the plain trapezoid reads the consensus profile — its curvature,
    // its shape, or both — so it is smoothed once here rather than in each of them.
    let mut smoothed = ctx.template.to_vec();
    smoothing::whittaker_henderson(ctx.rt, &mut smoothed, params.template_smoothing_lambda);
    let ctx = &IntegrationContext {
        template: &smoothed,
        ..*ctx
    };

    match params.method {
        QuantificationMethod::Sum => unreachable!("handled above"),
        QuantificationMethod::Trapezoid => integrate_full_window(ctx),
        QuantificationMethod::BoundaryTrapezoid => boundary::integrate(ctx, params),
        QuantificationMethod::ProfileProjection => projection::integrate(ctx, params),
        QuantificationMethod::EmgFit => emg::integrate(ctx, params),
    }
}

/// Plain trapezoidal rule over the whole candidate window: no boundary detection, no
/// baseline correction. The reference point the other strategies are measured against.
fn integrate_full_window(ctx: &IntegrationContext) -> Vec<f32> {
    (0..ctx.n_fragments())
        .map(|fragment_idx| {
            let profile = ctx.xic.row(fragment_idx);
            trapezoid(ctx.rt, profile.as_slice().expect("XIC rows are contiguous"))
        })
        .collect()
}

/// Trapezoidal rule on an irregular grid: `∫ y dx ≈ Σ (x[i] - x[i-1]) · (y[i] + y[i-1]) / 2`.
///
/// Returns `0.0` for mismatched or too short inputs, which is the correct area of a signal
/// that cannot be integrated.
pub fn trapezoid(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < MIN_CYCLES_FOR_INTEGRATION {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 1..x.len() {
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) * 0.5;
    }
    area
}

/// Index of the largest finite value in `values`, ties resolved towards the lower index.
///
/// Returns `None` for an empty slice or when every value is non-finite.
pub fn argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;

    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        match best {
            Some((_, best_value)) if value <= best_value => {}
            _ => best = Some((idx, value)),
        }
    }

    best.map(|(idx, _)| idx)
}
