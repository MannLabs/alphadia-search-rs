use rayon::prelude::*;

use super::algorithm::{FitConfig, Group};

/// Cauchy loss tuning constant for 95 % efficiency at the normal distribution (Holland & Welsch 1977).
const CAUCHY_TUNING_CONSTANT: f64 = 2.385;
/// Median absolute deviation (MAD) to standard deviation at the normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Keeps the offset shrinkage finite when the fragment offsets spread no wider than the noise.
const MIN_OFFSET_VARIANCE: f64 = 1e-4;
/// Floor of the robust residual scale (log2): data whose majority of residuals is exactly zero
/// would otherwise get a zero loss scale.
const MIN_RESIDUAL_SCALE: f64 = 1e-3;
const VARIANCE_BINS: usize = 30;
/// Small data gets fewer bins, so that every bin variance rests on enough residuals.
const MIN_VARIANCE_BIN_SIZE: usize = 50;

/// Residual scale `sigma`, Cauchy loss scale `tau` and offset shrinkage `lambda`, in log2.
#[derive(Clone, Copy)]
pub struct FitParameters {
    pub sigma: f64,
    pub tau: f64,
    pub lambda: f64,
}

/// Step 2 of the procedure: the scalars of the fit and the variance curve, estimated from the
/// residuals of all precursors.
pub struct NoiseModel {
    pub parameters: FitParameters,
    pub variance: VarianceCurve,
}

impl NoiseModel {
    pub fn estimate(groups: &[Group], initial: &[Vec<f64>], config: &FitConfig) -> Self {
        let per_group: Vec<(Vec<Residual>, Vec<RawOffset>)> = groups
            .par_iter()
            .zip(initial)
            .map(|(group, quantity)| first_sweep_residuals(group, quantity))
            .collect();
        // concatenated once: a parallel reduce with extend holds both halves and a regrown
        // buffer at once, which doubles the peak memory
        let mut residuals = Vec::with_capacity(per_group.iter().map(|(r, _)| r.len()).sum());
        let mut raw_offsets = Vec::with_capacity(per_group.iter().map(|(_, o)| o.len()).sum());
        for (group_residuals, group_raw_offsets) in per_group {
            residuals.extend(group_residuals);
            raw_offsets.extend(group_raw_offsets);
        }
        let mut values: Vec<f32> = residuals.iter().map(|residual| residual.value).collect();
        let (sigma, variance) = rayon::join(
            || residual_scale(&mut values),
            || VarianceCurve::from_residuals(residuals),
        );
        let parameters = FitParameters {
            sigma,
            tau: config.tau.unwrap_or(CAUCHY_TUNING_CONSTANT * sigma),
            lambda: config
                .lambda
                .unwrap_or_else(|| offset_shrinkage(&raw_offsets, sigma)),
        };
        Self {
            parameters,
            variance,
        }
    }
}

/// A first-sweep residual and the expected log2 intensity of its fragment measurement.
struct Residual {
    expected_intensity: f32,
    value: f32,
}

/// The unshrunk offset of a fragment: the mean of its deviations over its `n` observed runs.
struct RawOffset {
    mean: f64,
    n: usize,
}

/// Residuals around the unshrunk offsets, so that the library offsets do not count as noise,
/// scaled by sqrt(n / (n - 1)) for the offset fitted to the same n measurements.
///
/// The first-sweep quantity of a run is the share-weighted mean `sum_g w_g y_g` over the
/// fragments observed in that run, so it contains the fragment's own noise. With equal noise
/// variance v, its deviation has variance `v * (1 - 2 w_f + sum_g w_g^2)`: high-share
/// fragments would look too precise, and since high share means high expected intensity, the
/// variance curve would fall although the noise is flat. Each deviation is therefore divided
/// by the square root of this leverage factor. Runs with a single observed fragment have a
/// factor of zero and carry no information on noise or offsets.
fn first_sweep_residuals(group: &Group, quantity: &[f64]) -> (Vec<Residual>, Vec<RawOffset>) {
    let mut share_sum = vec![0.0; group.n_runs];
    let mut share_square_sum = vec![0.0; group.n_runs];
    let mut n_observed = vec![0usize; group.n_runs];
    for (ion, &share) in group.share.iter().enumerate() {
        for run in 0..group.n_runs {
            if group.deviation(ion, run, quantity).is_some() {
                share_sum[run] += share;
                share_square_sum[run] += share * share;
                n_observed[run] += 1;
            }
        }
    }
    let mut residuals = Vec::new();
    let mut raw_offsets = Vec::with_capacity(group.n_ions);
    let mut deviations = Vec::with_capacity(group.n_runs);
    for ion in 0..group.n_ions {
        deviations.clear();
        deviations.extend((0..group.n_runs).filter_map(|run| {
            if n_observed[run] < 2 {
                return None;
            }
            group.deviation(ion, run, quantity).map(|d| {
                let w = group.share[ion] / share_sum[run];
                let leverage = 1.0 - 2.0 * w + share_square_sum[run] / share_sum[run].powi(2);
                (quantity[run], d, leverage.sqrt())
            })
        }));
        let n = deviations.len();
        let mean = deviations.iter().map(|(_, d, _)| d).sum::<f64>() / n.max(1) as f64;
        if n >= 2 {
            let scale = (n as f64 / (n - 1) as f64).sqrt();
            let log_share = group.share[ion].log2();
            residuals.extend(deviations.iter().map(|(q, d, leverage)| Residual {
                expected_intensity: (log_share + q) as f32,
                value: ((d - mean) / leverage * scale) as f32,
            }));
        }
        raw_offsets.push(RawOffset { mean, n });
    }
    (residuals, raw_offsets)
}

fn median(values: &mut [f32]) -> f64 {
    let n = values.len();
    let (_, &mut upper, _) = values.select_nth_unstable_by(n / 2, f32::total_cmp);
    if n % 2 == 1 {
        return upper as f64;
    }
    let lower = values[..n / 2]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (lower as f64 + upper as f64) / 2.0
}

/// Robust standard deviation: the scaled MAD.
fn residual_scale(residuals: &mut [f32]) -> f64 {
    if residuals.is_empty() {
        return MIN_RESIDUAL_SCALE;
    }
    let center = median(residuals);
    for r in residuals.iter_mut() {
        *r = (*r as f64 - center).abs() as f32;
    }
    (MAD_TO_SIGMA * median(residuals)).max(MIN_RESIDUAL_SCALE)
}

/// Ratio of noise to offset variance (random effects, method of moments).
fn offset_shrinkage(raw_offsets: &[RawOffset], sigma: f64) -> f64 {
    let enough: Vec<&RawOffset> = raw_offsets.iter().filter(|offset| offset.n >= 2).collect();
    if enough.is_empty() {
        return 0.0;
    }
    let count = enough.len() as f64;
    let mean = enough.iter().map(|offset| offset.mean).sum::<f64>() / count;
    let variance = enough
        .iter()
        .map(|offset| (offset.mean - mean).powi(2))
        .sum::<f64>()
        / count;
    let noise = enough
        .iter()
        .map(|offset| sigma * sigma / offset.n as f64)
        .sum::<f64>()
        / count;
    sigma * sigma / (variance - noise).max(MIN_OFFSET_VARIANCE)
}

/// Residual variance against expected log2 intensity: the robust log variance of equal-count
/// bins at their median expected intensity, interpolated linearly between bins.
pub struct VarianceCurve {
    expected_intensity: Vec<f64>,
    log_variance: Vec<f64>,
}

impl VarianceCurve {
    fn from_residuals(mut residuals: Vec<Residual>) -> Self {
        residuals
            .par_sort_unstable_by(|a, b| a.expected_intensity.total_cmp(&b.expected_intensity));
        let n_bins = (residuals.len() / MIN_VARIANCE_BIN_SIZE).clamp(1, VARIANCE_BINS);
        let mut expected_intensity = Vec::with_capacity(n_bins);
        let mut log_variance = Vec::with_capacity(n_bins);
        for bin in 0..n_bins {
            let residuals =
                &residuals[bin * residuals.len() / n_bins..(bin + 1) * residuals.len() / n_bins];
            expected_intensity.push(
                residuals
                    .get(residuals.len() / 2)
                    .map_or(0.0, |residual| residual.expected_intensity as f64),
            );
            let mut values: Vec<f32> = residuals.iter().map(|residual| residual.value).collect();
            log_variance.push(2.0 * residual_scale(&mut values).ln());
        }
        // a measurement is at least as noisy as any brighter one
        for bin in (1..n_bins).rev() {
            log_variance[bin - 1] = log_variance[bin - 1].max(log_variance[bin]);
        }
        Self {
            expected_intensity,
            log_variance,
        }
    }

    pub fn variance(&self, expected_intensity: f64) -> f64 {
        let (x, y) = (&self.expected_intensity, &self.log_variance);
        let upper = x.partition_point(|&m| m < expected_intensity);
        let log_variance = if upper == 0 {
            y[0]
        } else if upper == x.len() {
            y[upper - 1]
        } else {
            let t = (expected_intensity - x[upper - 1]) / (x[upper] - x[upper - 1]);
            y[upper - 1] + t * (y[upper] - y[upper - 1])
        };
        log_variance.exp()
    }
}
