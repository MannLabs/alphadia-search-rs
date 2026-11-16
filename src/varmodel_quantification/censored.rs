use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use super::noise::{
    FitParameters, CAUCHY_TUNING_CONSTANT, MAD_TO_SIGMA, MIN_OFFSET_VARIANCE, MIN_RESIDUAL_SCALE,
    MIN_VARIANCE_BIN_SIZE, VARIANCE_BINS,
};
use super::numeric::{
    array_mean, array_sum, array_variance, clip, expit, interpolate, log_normal_cdf, logit, median,
    nan_to_num, nelder_mead, quantile,
};

const MAX_SWEEPS: usize = 60;
const TOLERANCE: f64 = 1e-4;
/// A run quantity needs this many observed fragments.
const MIN_IONS: usize = 2;
/// Precursor runs with this many observed fragments have a trustworthy expected intensity for
/// every fragment.
const CURVE_MIN_OBSERVED: usize = 4;
const CURVE_ROUNDS: usize = 2;
pub const DEFAULT_CURVE_MAX_CELLS: usize = 300_000;
const CURVE_SEED: u64 = 0;
const CURVE_START_QUANTILE: f64 = 0.02;
const CURVE_START_PLATEAU: f64 = 1e-3;
const PROBABILITY_FLOOR: f64 = 1e-12;
/// log(sqrt(2 pi))
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MissingFragments {
    Drop,
    Censored,
}

pub struct CensoredConfig {
    pub missing_fragments: MissingFragments,
    /// gamma of the per-measurement variance factor `exp(gamma * (1 - sqrt(C)))`.
    pub quality_exponent: f64,
    /// The dropout curve of a run is fitted on a random subset of this many measurements.
    pub curve_max_cells: usize,
}

/// Per run: `P(missing | mu) = plateau + (1 - plateau) * Phi((limit - mu) / width)`.
pub struct Dropout {
    pub limit: Vec<f64>,
    pub width: Vec<f64>,
    pub plateau: Vec<f64>,
}

pub struct CensoredResult {
    /// Run-major `(n_runs, n_precursors)`, NaN where too few fragments were observed.
    pub quantity: Vec<f64>,
    pub parameters: FitParameters,
    pub dropout: Option<Dropout>,
}

/// One precursor in f64, ion-major: `y = log2(I / s)` (NaN where missing), the log2 library
/// share, the share recovered from it, and the measurement variances once the noise is known
/// (before that, the log variance factor of the measurement quality).
struct Group {
    y: Vec<f64>,
    log_share: Vec<f64>,
    share: Vec<f64>,
    variance: Vec<f64>,
    n_ions: usize,
    n_runs: usize,
}

impl Group {
    fn index(&self, ion: usize, run: usize) -> usize {
        ion * self.n_runs + run
    }

    fn observed(&self, ion: usize, run: usize) -> bool {
        self.y[self.index(ion, run)].is_finite()
    }

    fn n_observed(&self, run: usize) -> usize {
        (0..self.n_ions)
            .filter(|&ion| self.observed(ion, run))
            .count()
    }
}

/// Fragment offsets and run quantities of one precursor during the fit.
struct State {
    quantity: Vec<f64>,
    offset: Vec<f64>,
}

struct Noise {
    parameters: FitParameters,
    curve_x: Vec<f64>,
    curve_log_variance: Vec<f64>,
}

impl Noise {
    fn variance(&self, expected: f64) -> f64 {
        interpolate(expected, &self.curve_x, &self.curve_log_variance).exp()
    }
}

/// The variance-model fit in f64 with missing fragments of quantified runs as censored
/// observations and the variances weighted by the measurement quality, as configured.
#[allow(clippy::too_many_arguments)]
pub fn fit_censored(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    correlation: Option<&[f32]>,
    n_runs: usize,
    n_precursors: usize,
    config: &CensoredConfig,
) -> CensoredResult {
    let mut groups = build_groups(
        intensity,
        intensity_library,
        precursor,
        correlation,
        config.quality_exponent,
        n_runs,
        n_precursors,
    );
    let initial: Vec<Vec<f64>> = groups.par_iter().map(initial_quantities).collect();
    let noise = estimate_noise(&groups, &initial);
    groups
        .par_iter_mut()
        .zip(&initial)
        .for_each(|(group, quantity)| set_variances(group, quantity, &noise));

    let mut states = fit(&groups, &initial, &noise, None);
    let mut dropout = None;
    if config.missing_fragments == MissingFragments::Censored {
        for _ in 0..CURVE_ROUNDS {
            let curve = fit_dropout(&groups, &states, n_runs, config.curve_max_cells);
            states = fit(&groups, &initial, &noise, Some(&curve));
            dropout = Some(curve);
        }
    }
    CensoredResult {
        quantity: run_major(&states, n_runs, n_precursors),
        parameters: noise.parameters,
        dropout,
    }
}

fn build_groups(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    correlation: Option<&[f32]>,
    quality_exponent: f64,
    n_runs: usize,
    n_precursors: usize,
) -> Vec<Group> {
    let n_ions = intensity_library.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_precursors];
    for (ion, &index) in precursor.iter().enumerate() {
        let l = intensity_library[ion] as f64;
        if l.is_finite() && l > 0.0 {
            members[index].push(ion);
        }
    }
    members
        .into_par_iter()
        .map(|ions| {
            let intensity_library_sum: f64 =
                ions.iter().map(|&ion| intensity_library[ion] as f64).sum();
            let original_share: Vec<f64> = ions
                .iter()
                .map(|&ion| intensity_library[ion] as f64 / intensity_library_sum)
                .collect();
            let log_share: Vec<f64> = original_share.iter().map(|s| s.log2()).collect();
            let share = log_share.iter().map(|l| l.exp2()).collect();
            let mut y = Vec::with_capacity(ions.len() * n_runs);
            let mut variance = Vec::new();
            for (&ion, &s) in ions.iter().zip(&original_share) {
                for run in 0..n_runs {
                    let i = intensity[run * n_ions + ion] as f64;
                    y.push(if i > 0.0 && i.is_finite() {
                        (i / s).log2()
                    } else {
                        f64::NAN
                    });
                    if let Some(correlation) = correlation.filter(|_| quality_exponent > 0.0) {
                        let c = clip(nan_to_num(correlation[run * n_ions + ion] as f64), 0.0, 1.0);
                        variance.push(quality_exponent * (1.0 - c.sqrt()));
                    }
                }
            }
            Group {
                y,
                log_share,
                share,
                variance,
                n_ions: ions.len(),
                n_runs,
            }
        })
        .collect()
}

/// The library-share weighted mean of the observed fragments, without offsets.
fn initial_quantities(group: &Group) -> Vec<f64> {
    (0..group.n_runs)
        .map(|run| {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for ion in 0..group.n_ions {
                if group.observed(ion, run) {
                    let w = group.share[ion];
                    if w > 0.0 {
                        numerator += w * group.y[group.index(ion, run)];
                    }
                    denominator += w;
                }
            }
            if denominator > 0.0 {
                numerator / denominator
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn set_variances(group: &mut Group, initial: &[f64], noise: &Noise) {
    let log_variance_factor = std::mem::take(&mut group.variance);
    let mut variance = Vec::with_capacity(group.n_ions * group.n_runs);
    for ion in 0..group.n_ions {
        for run in 0..group.n_runs {
            let factor = log_variance_factor
                .get(group.index(ion, run))
                .copied()
                .unwrap_or(0.0);
            variance.push(noise.variance(group.log_share[ion] + initial[run]) * factor.exp());
        }
    }
    group.variance = variance;
}

fn robust_scale(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return MIN_RESIDUAL_SCALE;
    }
    let center = median(values);
    for value in values.iter_mut() {
        *value = (*value - center).abs();
    }
    (MAD_TO_SIGMA * median(values)).max(MIN_RESIDUAL_SCALE)
}

/// Residuals (expected log2 intensity, value) around the unshrunk offsets with leverage and
/// degrees-of-freedom corrections, and the unshrunk offsets (mean deviation, count) of the
/// fragments seen in at least two usable runs.
#[allow(clippy::type_complexity)]
fn first_sweep_residuals(group: &Group, initial: &[f64]) -> (Vec<(f64, f64)>, Vec<(f64, usize)>) {
    let n_runs = group.n_runs;
    let usable_run: Vec<bool> = (0..n_runs)
        .map(|run| group.n_observed(run) >= MIN_IONS)
        .collect();
    let usable = |ion: usize, run: usize| usable_run[run] && group.observed(ion, run);
    let mut share_sum = vec![0.0; n_runs];
    let mut share_square_sum = vec![0.0; n_runs];
    for ion in 0..group.n_ions {
        let share = group.share[ion];
        for run in 0..n_runs {
            if usable(ion, run) {
                share_sum[run] += share;
                share_square_sum[run] += share * share;
            }
        }
    }
    let mut residuals = Vec::new();
    let mut raw_offsets = Vec::new();
    let mut usable_deviations = vec![0.0; n_runs];
    for ion in 0..group.n_ions {
        let mut n = 0usize;
        for run in 0..n_runs {
            usable_deviations[run] = if usable(ion, run) {
                n += 1;
                group.y[group.index(ion, run)] - initial[run]
            } else {
                0.0
            };
        }
        if n < MIN_IONS {
            continue;
        }
        let mean = array_sum(&usable_deviations) / n as f64;
        let scale = (n as f64 / (n - 1) as f64).sqrt();
        for run in 0..n_runs {
            if !usable(ion, run) {
                continue;
            }
            let w = group.share[ion] / share_sum[run];
            let leverage =
                1.0 - 2.0 * w + share_square_sum[run] / (share_sum[run] * share_sum[run]);
            if leverage > 0.0 {
                let deviation = group.y[group.index(ion, run)] - initial[run];
                residuals.push((
                    group.log_share[ion] + initial[run],
                    (deviation - mean) / leverage.sqrt() * scale,
                ));
            }
        }
        raw_offsets.push((mean, n));
    }
    (residuals, raw_offsets)
}

/// Residual scale, variance curve over the expected intensity and offset shrinkage.
fn estimate_noise(groups: &[Group], initial: &[Vec<f64>]) -> Noise {
    let per_group: Vec<_> = groups
        .par_iter()
        .zip(initial)
        .map(|(group, quantity)| first_sweep_residuals(group, quantity))
        .collect();
    let mut residuals = Vec::with_capacity(per_group.iter().map(|(r, _)| r.len()).sum());
    let mut raw_offsets = Vec::with_capacity(per_group.iter().map(|(_, o)| o.len()).sum());
    for (group_residuals, group_raw_offsets) in per_group {
        residuals.extend(group_residuals);
        raw_offsets.extend(group_raw_offsets);
    }
    let mut values: Vec<f64> = residuals.iter().map(|&(_, value)| value).collect();
    let sigma = robust_scale(&mut values);

    residuals.par_sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = residuals.len();
    let n_bins = (n / MIN_VARIANCE_BIN_SIZE).clamp(1, VARIANCE_BINS);
    let mut curve_x = Vec::with_capacity(n_bins);
    let mut curve_log_variance = Vec::with_capacity(n_bins);
    for bin in 0..n_bins {
        let (start, end) = (bin * n / n_bins, (bin + 1) * n / n_bins);
        curve_x.push(residuals.get((start + end) / 2).map_or(0.0, |r| r.0));
        let mut bin_values: Vec<f64> = residuals[start..end].iter().map(|r| r.1).collect();
        curve_log_variance.push(2.0 * robust_scale(&mut bin_values).ln());
    }
    // a measurement is at least as noisy as any brighter one
    for bin in (1..n_bins).rev() {
        curve_log_variance[bin - 1] = curve_log_variance[bin - 1].max(curve_log_variance[bin]);
    }

    let lambda = if raw_offsets.is_empty() {
        0.0
    } else {
        let noise_part: Vec<f64> = raw_offsets
            .iter()
            .map(|&(_, n)| sigma * sigma / n as f64)
            .collect();
        let means: Vec<f64> = raw_offsets.iter().map(|&(mean, _)| mean).collect();
        sigma * sigma / (array_variance(&means) - array_mean(&noise_part)).max(MIN_OFFSET_VARIANCE)
    };
    Noise {
        parameters: FitParameters {
            sigma,
            tau: CAUCHY_TUNING_CONSTANT * sigma,
            lambda,
        },
        curve_x,
        curve_log_variance,
    }
}

/// Value (y units) and weight of a missing measurement's pseudo-observation: its conditional
/// mean under the censored component, weighted by the posterior censoring probability.
fn pseudo_observation(mu: f64, log_share: f64, dropout: &Dropout, run: usize) -> (f64, f64) {
    let s = dropout.width[run];
    let z = (dropout.limit[run] - mu) / s;
    let log_cdf = log_normal_cdf(z);
    let mills = (-0.5 * (z * z) - LOG_SQRT_2PI - log_cdf).exp();
    let cdf = log_cdf.exp();
    let pi = dropout.plateau[run];
    let responsibility = (1.0 - pi) * cdf / (pi + (1.0 - pi) * cdf);
    // a run whose curve never censors (limit -inf, plateau 0) gives 0/0 here; its missing
    // measurements carry no information, and a NaN would spread through the fragment offsets
    // into the other runs
    let weight = nan_to_num(responsibility / (s * s));
    let value = if weight > 0.0 {
        mu - s * mills - log_share
    } else {
        0.0
    };
    (value, weight)
}

struct SweepConstants<'a> {
    loss: f64,
    prior_weight: f64,
    dropout: Option<&'a Dropout>,
}

/// One sweep of one precursor: weights, offsets, quantities. Returns the largest change of a
/// run quantity.
fn sweep(group: &Group, quantified: &[bool], state: &mut State, constants: &SweepConstants) -> f64 {
    let (n_ions, n_runs) = (group.n_ions, group.n_runs);
    let mut weights = vec![0.0; n_ions * n_runs];
    let mut values = vec![0.0; n_ions * n_runs];
    for ion in 0..n_ions {
        for run in 0..n_runs {
            let index = group.index(ion, run);
            let y = group.y[index];
            if y.is_finite() {
                let e = y - state.quantity[run] - state.offset[ion];
                weights[index] = 1.0 / (group.variance[index] + constants.loss * (e * e));
                values[index] = y;
            } else if let Some(dropout) = constants.dropout.filter(|_| quantified[run]) {
                let mu = state.quantity[run] + state.offset[ion] + group.log_share[ion];
                (values[index], weights[index]) =
                    pseudo_observation(mu, group.log_share[ion], dropout, run);
            }
        }
    }
    let mut numerator = vec![0.0; n_runs];
    for ion in 0..n_ions {
        let row = ion * n_runs..(ion + 1) * n_runs;
        for (run, term) in numerator.iter_mut().enumerate() {
            let index = row.start + run;
            *term = weights[index] * (values[index] - state.quantity[run]);
        }
        state.offset[ion] =
            array_sum(&numerator) / (array_sum(&weights[row]) + constants.prior_weight);
    }
    let mut change: f64 = 0.0;
    for run in 0..n_runs {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for ion in 0..n_ions {
            let w = weights[group.index(ion, run)];
            if w > 0.0 {
                numerator += w * nan_to_num(values[group.index(ion, run)] - state.offset[ion]);
            }
            denominator += w;
        }
        let quantity = if !quantified[run] {
            0.0
        } else if denominator > 0.0 {
            nan_to_num(numerator / denominator)
        } else {
            0.0
        };
        change = change.max((quantity - state.quantity[run]).abs());
        state.quantity[run] = quantity;
    }
    change
}

/// Alternates weights, offsets and quantities of all precursors in lockstep until no run
/// quantity of any precursor moves by `TOLERANCE`. Quantities with fewer than `MIN_IONS`
/// observed fragments are NaN.
fn fit(
    groups: &[Group],
    initial: &[Vec<f64>],
    noise: &Noise,
    dropout: Option<&Dropout>,
) -> Vec<State> {
    let FitParameters { sigma, tau, lambda } = noise.parameters;
    let constants = SweepConstants {
        loss: (sigma / tau).powi(2),
        prior_weight: lambda / (sigma * sigma),
        dropout,
    };
    let quantified: Vec<Vec<bool>> = groups
        .par_iter()
        .map(|group| {
            (0..group.n_runs)
                .map(|run| group.n_observed(run) > 0)
                .collect()
        })
        .collect();
    let mut states: Vec<State> = groups
        .iter()
        .zip(initial)
        .map(|(group, quantity)| State {
            quantity: quantity.iter().map(|&q| nan_to_num(q)).collect(),
            offset: vec![0.0; group.n_ions],
        })
        .collect();
    for _ in 0..MAX_SWEEPS {
        let change = states
            .par_iter_mut()
            .zip(groups)
            .zip(&quantified)
            .map(|((state, group), quantified)| sweep(group, quantified, state, &constants))
            .reduce(|| 0.0, f64::max);
        if change < TOLERANCE {
            break;
        }
    }
    states
        .par_iter_mut()
        .zip(groups)
        .for_each(|(state, group)| {
            for run in 0..group.n_runs {
                if group.n_observed(run) < MIN_IONS {
                    state.quantity[run] = f64::NAN;
                }
            }
        });
    states
}

/// Per-run maximum-likelihood dropout curves on the precursor runs whose expected intensities
/// are trustworthy.
fn fit_dropout(groups: &[Group], states: &[State], n_runs: usize, max_cells: usize) -> Dropout {
    let curves: Vec<(f64, f64, f64)> = (0..n_runs)
        .into_par_iter()
        .map(|run| fit_dropout_curve(groups, states, run, max_cells))
        .collect();
    Dropout {
        limit: curves.iter().map(|c| c.0).collect(),
        width: curves.iter().map(|c| c.1).collect(),
        plateau: curves.iter().map(|c| c.2).collect(),
    }
}

fn fit_dropout_curve(
    groups: &[Group],
    states: &[State],
    run: usize,
    max_cells: usize,
) -> (f64, f64, f64) {
    let mut expected = Vec::new();
    let mut missing = Vec::new();
    for (group, state) in groups.iter().zip(states) {
        let q = state.quantity[run];
        if !q.is_finite() || group.n_observed(run) < CURVE_MIN_OBSERVED {
            continue;
        }
        for ion in 0..group.n_ions {
            expected.push(q + state.offset[ion] + group.log_share[ion]);
            missing.push(!group.observed(ion, run));
        }
    }
    let n_missing = missing.iter().filter(|&&m| m).count();
    if n_missing == 0 || n_missing == missing.len() {
        // no missing (or no observed) fragments to fit a curve on: a curve that never censors
        return (f64::NEG_INFINITY, 1.0, 0.0);
    }
    if expected.len() > max_cells {
        let mut rng = StdRng::seed_from_u64(CURVE_SEED.wrapping_add(run as u64));
        let mut pick = rand::seq::index::sample(&mut rng, expected.len(), max_cells).into_vec();
        pick.sort_unstable();
        expected = pick.iter().map(|&i| expected[i]).collect();
        missing = pick.iter().map(|&i| missing[i]).collect();
    }
    let n_missing = missing.iter().filter(|&&m| m).count();
    let mut observed_expected: Vec<f64> = expected
        .iter()
        .zip(&missing)
        .filter(|(_, &m)| !m)
        .map(|(&e, _)| e)
        .collect();
    let missing_share = n_missing as f64 / missing.len() as f64;
    let start = [
        quantile(&mut observed_expected, CURVE_START_QUANTILE),
        0.0,
        logit((missing_share / 2.0).max(CURVE_START_PLATEAU)),
    ];
    let mut terms = vec![0.0; expected.len()];
    let negative_log_likelihood = |theta: &[f64; 3]| {
        let [limit, log_width, plateau_logit] = *theta;
        let pi = expit(plateau_logit);
        let width = log_width.exp();
        terms
            .par_iter_mut()
            .zip(&expected)
            .zip(&missing)
            .for_each(|((term, &mu), &miss)| {
                let cdf = log_normal_cdf((limit - mu) / width).exp();
                let probability = clip(
                    pi + (1.0 - pi) * cdf,
                    PROBABILITY_FLOOR,
                    1.0 - PROBABILITY_FLOOR,
                );
                *term = if miss {
                    probability.ln()
                } else {
                    (-probability).ln_1p()
                };
            });
        -array_sum(&terms)
    };
    let [limit, log_width, plateau_logit] = nelder_mead(negative_log_likelihood, start);
    (limit, log_width.exp(), expit(plateau_logit))
}

fn run_major(states: &[State], n_runs: usize, n_precursors: usize) -> Vec<f64> {
    let mut quantity = vec![f64::NAN; n_runs * n_precursors];
    for (index, state) in states.iter().enumerate() {
        for (run, &q) in state.quantity.iter().enumerate() {
            quantity[run * n_precursors + index] = q;
        }
    }
    quantity
}
