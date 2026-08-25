//! Exponentially modified Gaussian fitted to the consensus elution profile.
//!
//! The two model-free strategies can only ever see the cycles that were acquired. At the
//! default selection settings a candidate window holds around eleven cycles and a peak is
//! sampled a handful of times across its full width, so two errors are unavoidable: the
//! trapezoidal rule under-reads a curved peak sampled that coarsely, and a tailing peak
//! loses whatever tail falls outside the window. Both are systematic, and both depend on
//! where the apex happens to land between two cycles — which is exactly the kind of error
//! that shows up as run-to-run variance in label-free quantification.
//!
//! Fitting a chromatographic peak shape addresses both. Liquid chromatography peaks are
//! well described by an exponentially modified Gaussian, the convolution of a Gaussian
//! with a one-sided exponential that accounts for the tailing every real column produces:
//!
//! ```text
//! f(t; μ, σ, τ) = 1/(2τ) · exp(σ²/(2τ²) - (t-μ)/τ) · erfc( (σ/τ - (t-μ)/σ) / √2 )
//! ```
//!
//! `f` is a probability density, so it integrates to one and a fitted amplitude *is* an
//! area. The three shape parameters are fitted once per precursor on the consensus
//! profile — never per fragment, so all fragments share one elution model and their ratios
//! stay consistent — and each fragment then contributes only its amplitude:
//!
//! ```text
//! a_i    = argmin Σ_c w_c (I_ic - a f(t_c) - b_i)²   (same robust solver as profile_projection)
//! area_i = a_i · ∫ f(t) dt over the integration range
//! ```
//!
//! The integral is evaluated on a grid `emg_upsample_factor` times finer than the
//! acquisition cycles, which removes the coarse-sampling bias, and optionally over a
//! window widened by `emg_extrapolation_factor` to recover a truncated tail.
//!
//! When the fit fails — a shape that no exponentially modified Gaussian describes, a
//! degenerate window, an optimiser that does not converge — quantification falls back to
//! [`super::projection`], which makes the same co-elution assumption without assuming a
//! functional form.

use super::IntegrationContext;
use crate::peak_group_quantification::parameters::QuantificationParameters;

use std::f64::consts::{PI, SQRT_2};

/// Above this standardised argument `erfc` underflows against the exponential prefactor and
/// the density is evaluated through its Gaussian limit instead.
const ERFC_ASYMPTOTIC_THRESHOLD: f64 = 4.0;

/// Squared cosine similarity between the consensus profile and the fitted shape below which
/// the fit is rejected. A well-sampled chromatographic peak reaches well above this.
const MIN_FIT_QUALITY: f64 = 0.8;

/// Shape bounds relative to the window, as multiples of the mean cycle time (lower) and of
/// the window span (upper). They keep the optimiser inside shapes the sampling can
/// actually resolve.
const MIN_SIGMA_CYCLES: f64 = 0.2;
const MAX_SIGMA_SPANS: f64 = 1.0;
const MIN_TAU_CYCLES: f64 = 0.05;
const MAX_TAU_SPANS: f64 = 1.0;

/// The apex may drift by at most one window span in either direction.
const MAX_APEX_DRIFT_SPANS: f64 = 1.0;

/// Nelder-Mead settings. The objective is three-dimensional and cheap, so a generous
/// iteration budget still costs little per precursor.
const NM_MAX_ITERATIONS: usize = 200;
const NM_TOLERANCE: f64 = 1e-7;
const NM_REFLECT: f64 = 1.0;
const NM_EXPAND: f64 = 2.0;
const NM_CONTRACT: f64 = 0.5;
const NM_SHRINK: f64 = 0.5;

/// Upper bound on the integration grid, so that a large extrapolation factor combined with
/// a large upsampling factor cannot blow up the per-candidate cost.
const MAX_GRID_POINTS: usize = 4096;

/// A fitted exponentially modified Gaussian.
#[derive(Clone, Copy, Debug)]
pub struct EmgShape {
    pub mu: f64,
    pub sigma: f64,
    pub tau: f64,
}

impl EmgShape {
    /// Density at `t`, evaluated in the numerically stable branch for the given argument.
    pub fn density(&self, t: f64) -> f64 {
        let offset = t - self.mu;
        let z = (self.sigma / self.tau - offset / self.sigma) / SQRT_2;

        if z < ERFC_ASYMPTOTIC_THRESHOLD {
            let exponent =
                self.sigma * self.sigma / (2.0 * self.tau * self.tau) - offset / self.tau;
            return exponent.exp() * erfc(z) / (2.0 * self.tau);
        }

        // For large z the exponential prefactor and erfc(z) cancel into the Gaussian limit:
        // exp(σ²/2τ² - Δ/τ) · exp(-z²) / (z√π) = exp(-Δ²/2σ²) / (z√π).
        let gaussian = (-offset * offset / (2.0 * self.sigma * self.sigma)).exp();
        gaussian / (2.0 * self.tau * z * PI.sqrt())
    }
}

pub fn integrate(ctx: &IntegrationContext, params: &QuantificationParameters) -> Vec<f32> {
    let rt: Vec<f64> = ctx.rt.iter().map(|&value| value as f64).collect();
    let template: Vec<f64> = ctx.template.iter().map(|&value| value as f64).collect();

    let Some(shape) = fit(&rt, &template) else {
        return super::projection::integrate(ctx, params);
    };

    let sampled: Vec<f32> = rt.iter().map(|&t| shape.density(t) as f32).collect();
    let model_area = integrate_shape(&shape, &rt, params) as f32;

    if !model_area.is_finite() || model_area <= 0.0 {
        return super::projection::integrate(ctx, params);
    }

    let support = super::projection::support_mask(&sampled);
    let mut workspace = super::projection::Workspace::new(ctx.n_cycles());

    (0..ctx.n_fragments())
        .map(|fragment_idx| {
            let profile = ctx.xic.row(fragment_idx);
            let amplitude = super::projection::robust_amplitude(
                &sampled,
                profile.as_slice().expect("XIC rows are contiguous"),
                &support,
                params,
                &mut workspace,
            );
            (amplitude * model_area).max(0.0)
        })
        .collect()
}

/// Integral of the fitted shape over the reported range: the observed retention time
/// window, widened by `emg_extrapolation_factor`, on a grid `emg_upsample_factor` times
/// finer than the acquisition cycles.
fn integrate_shape(shape: &EmgShape, rt: &[f64], params: &QuantificationParameters) -> f64 {
    let span = rt[rt.len() - 1] - rt[0];
    let margin = 0.5 * span * (params.emg_extrapolation_factor.max(1.0) as f64 - 1.0);
    let lower = rt[0] - margin;
    let upper = rt[rt.len() - 1] + margin;

    let mean_cycle_time = span / (rt.len() - 1) as f64;
    let step = mean_cycle_time / params.emg_upsample_factor.max(1) as f64;
    let n_points = (((upper - lower) / step).ceil() as usize + 1).clamp(2, MAX_GRID_POINTS);
    let step = (upper - lower) / (n_points - 1) as f64;

    let mut area = 0.0;
    let mut previous = shape.density(lower);
    for idx in 1..n_points {
        let current = shape.density(lower + step * idx as f64);
        area += 0.5 * step * (previous + current);
        previous = current;
    }
    area
}

/// Fit `(μ, σ, τ)` to a consensus elution profile.
///
/// The amplitude is profiled out analytically, so only the three shape parameters are
/// searched. Returns `None` when the profile carries no peak or the best fit is worse than
/// [`MIN_FIT_QUALITY`].
pub fn fit(rt: &[f64], template: &[f64]) -> Option<EmgShape> {
    let bounds = Bounds::new(rt)?;
    let start = initial_guess(rt, template, &bounds)?;

    let objective = |parameters: [f64; 3]| -> f64 {
        match bounds.to_shape(parameters) {
            Some(shape) => -fit_quality_numerator(&shape, rt, template),
            None => f64::INFINITY,
        }
    };

    let simplex_step = [bounds.mean_cycle_time, LOG_STEP_SIGMA, LOG_STEP_TAU];
    let (best, best_value) = nelder_mead(objective, bounds.to_parameters(&start), simplex_step);

    let shape = bounds.to_shape(best)?;
    let template_energy: f64 = template.iter().map(|value| value * value).sum();
    if template_energy <= 0.0 {
        return None;
    }

    // -best_value is (Σ t·m)² / Σ m²; dividing by Σ t² gives the squared cosine similarity
    // between the consensus profile and the fitted shape.
    let quality = -best_value / template_energy;
    if !quality.is_finite() || quality < MIN_FIT_QUALITY {
        return None;
    }

    Some(shape)
}

/// Initial simplex edge lengths in the log-scale search space, i.e. a factor of roughly
/// `e^0.4 ≈ 1.5` in sigma and tau.
const LOG_STEP_SIGMA: f64 = 0.4;
const LOG_STEP_TAU: f64 = 0.4;

/// `(Σ t·m)² / Σ m²`, the part of the residual sum of squares that depends on the shape
/// once the amplitude has been profiled out. Maximising it minimises the residual.
fn fit_quality_numerator(shape: &EmgShape, rt: &[f64], template: &[f64]) -> f64 {
    let mut cross = 0.0;
    let mut energy = 0.0;

    for idx in 0..rt.len() {
        let model = shape.density(rt[idx]);
        if !model.is_finite() {
            return 0.0;
        }
        cross += template[idx] * model;
        energy += model * model;
    }

    if energy <= 0.0 || cross <= 0.0 {
        return 0.0;
    }
    cross * cross / energy
}

/// Shape bounds derived from the sampling of the window, plus the mapping between the
/// bounded shape and the unconstrained search space the optimiser works in.
struct Bounds {
    mu_min: f64,
    mu_max: f64,
    sigma_min: f64,
    sigma_max: f64,
    tau_min: f64,
    tau_max: f64,
    mean_cycle_time: f64,
}

impl Bounds {
    fn new(rt: &[f64]) -> Option<Self> {
        let span = rt[rt.len() - 1] - rt[0];
        if !span.is_finite() || span <= 0.0 {
            return None;
        }
        let mean_cycle_time = span / (rt.len() - 1) as f64;

        Some(Self {
            mu_min: rt[0] - MAX_APEX_DRIFT_SPANS * span,
            mu_max: rt[rt.len() - 1] + MAX_APEX_DRIFT_SPANS * span,
            sigma_min: MIN_SIGMA_CYCLES * mean_cycle_time,
            sigma_max: MAX_SIGMA_SPANS * span,
            tau_min: MIN_TAU_CYCLES * mean_cycle_time,
            tau_max: MAX_TAU_SPANS * span,
            mean_cycle_time,
        })
    }

    /// Search space is `(μ, ln σ, ln τ)` so that the widths stay positive without the
    /// optimiser needing to know about constraints.
    fn to_parameters(&self, shape: &EmgShape) -> [f64; 3] {
        [shape.mu, shape.sigma.ln(), shape.tau.ln()]
    }

    fn to_shape(&self, parameters: [f64; 3]) -> Option<EmgShape> {
        let [mu, log_sigma, log_tau] = parameters;
        if !mu.is_finite() || !log_sigma.is_finite() || !log_tau.is_finite() {
            return None;
        }

        let sigma = log_sigma.exp();
        let tau = log_tau.exp();

        if mu < self.mu_min
            || mu > self.mu_max
            || sigma < self.sigma_min
            || sigma > self.sigma_max
            || tau < self.tau_min
            || tau > self.tau_max
        {
            return None;
        }

        Some(EmgShape { mu, sigma, tau })
    }
}

/// Method-of-moments starting point.
///
/// Treating the consensus profile as a distribution over retention time, an exponentially
/// modified Gaussian has mean `μ + τ`, variance `σ² + τ²` and third central moment `2τ³`,
/// which inverts directly into a starting guess.
fn initial_guess(rt: &[f64], template: &[f64], bounds: &Bounds) -> Option<EmgShape> {
    let mut weight_sum = 0.0;
    for &value in template {
        if value > 0.0 {
            weight_sum += value;
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }

    let weight = |value: f64| if value > 0.0 { value } else { 0.0 };

    let mut mean = 0.0;
    for idx in 0..rt.len() {
        mean += weight(template[idx]) * rt[idx];
    }
    mean /= weight_sum;

    let mut variance = 0.0;
    let mut third_moment = 0.0;
    for idx in 0..rt.len() {
        let centred = rt[idx] - mean;
        let w = weight(template[idx]);
        variance += w * centred * centred;
        third_moment += w * centred * centred * centred;
    }
    variance /= weight_sum;
    third_moment /= weight_sum;

    let tau = (third_moment.max(0.0) / 2.0)
        .cbrt()
        .clamp(bounds.tau_min, bounds.tau_max);
    let sigma = (variance - tau * tau)
        .max(bounds.sigma_min * bounds.sigma_min)
        .sqrt()
        .clamp(bounds.sigma_min, bounds.sigma_max);
    let mu = (mean - tau).clamp(bounds.mu_min, bounds.mu_max);

    Some(EmgShape { mu, sigma, tau })
}

/// Nelder-Mead downhill simplex in three dimensions.
///
/// Derivative free, which matters because the density involves `erfc` and switches between
/// two evaluation branches; a gradient method would have to differentiate through both.
fn nelder_mead(
    mut objective: impl FnMut([f64; 3]) -> f64,
    start: [f64; 3],
    step: [f64; 3],
) -> ([f64; 3], f64) {
    const N: usize = 3;

    let mut simplex = [start; N + 1];
    for dim in 0..N {
        simplex[dim + 1][dim] += step[dim];
    }
    let mut values = simplex.map(&mut objective);

    for _ in 0..NM_MAX_ITERATIONS {
        // Order the simplex so that index 0 is best and index N is worst.
        let mut order: [usize; N + 1] = [0, 1, 2, 3];
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ordered_simplex = order.map(|idx| simplex[idx]);
        let ordered_values = order.map(|idx| values[idx]);
        simplex = ordered_simplex;
        values = ordered_values;

        if converged(&values) {
            break;
        }

        let centroid = centroid_excluding_worst(&simplex);
        let worst = simplex[N];

        let reflected = combine(&centroid, &worst, NM_REFLECT);
        let reflected_value = objective(reflected);

        if reflected_value < values[0] {
            let expanded = combine(&centroid, &worst, NM_EXPAND);
            let expanded_value = objective(expanded);
            if expanded_value < reflected_value {
                simplex[N] = expanded;
                values[N] = expanded_value;
            } else {
                simplex[N] = reflected;
                values[N] = reflected_value;
            }
            continue;
        }

        if reflected_value < values[N - 1] {
            simplex[N] = reflected;
            values[N] = reflected_value;
            continue;
        }

        let contracted = combine(&centroid, &worst, -NM_CONTRACT);
        let contracted_value = objective(contracted);
        if contracted_value < values[N] {
            simplex[N] = contracted;
            values[N] = contracted_value;
            continue;
        }

        // Nothing improved: shrink the whole simplex towards the best vertex.
        let best = simplex[0];
        for idx in 1..=N {
            for dim in 0..N {
                simplex[idx][dim] = best[dim] + NM_SHRINK * (simplex[idx][dim] - best[dim]);
            }
            values[idx] = objective(simplex[idx]);
        }
    }

    let mut best_idx = 0;
    for idx in 1..=N {
        if values[idx] < values[best_idx] {
            best_idx = idx;
        }
    }
    (simplex[best_idx], values[best_idx])
}

/// Converged once the spread of objective values across the simplex is negligible relative
/// to their magnitude.
fn converged(values: &[f64; 4]) -> bool {
    let best = values[0];
    let worst = values[values.len() - 1];
    if !best.is_finite() {
        return false;
    }
    (worst - best).abs() <= NM_TOLERANCE * (best.abs() + worst.abs() + NM_TOLERANCE)
}

fn centroid_excluding_worst(simplex: &[[f64; 3]; 4]) -> [f64; 3] {
    let mut centroid = [0.0f64; 3];
    for point in simplex.iter().take(simplex.len() - 1) {
        for dim in 0..3 {
            centroid[dim] += point[dim];
        }
    }
    let count = (simplex.len() - 1) as f64;
    for value in centroid.iter_mut() {
        *value /= count;
    }
    centroid
}

/// `centroid + factor * (centroid - worst)`, the single geometric operation Nelder-Mead
/// needs for reflection, expansion and contraction.
fn combine(centroid: &[f64; 3], worst: &[f64; 3], factor: f64) -> [f64; 3] {
    let mut result = [0.0f64; 3];
    for dim in 0..3 {
        result[dim] = centroid[dim] + factor * (centroid[dim] - worst[dim]);
    }
    result
}

/// Complementary error function, Chebyshev approximation with a fractional error below
/// 1.2e-7 — well inside what the `f32` intensities warrant.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let value = t * poly.exp();

    if x >= 0.0 {
        value
    } else {
        2.0 - value
    }
}
