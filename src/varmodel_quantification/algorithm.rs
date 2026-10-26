use rayon::prelude::*;

/// Cauchy loss tuning constant for 95 % efficiency at the normal distribution (Holland & Welsch 1977).
const CAUCHY_TUNING_CONSTANT: f64 = 2.385;
/// MAD to standard deviation at the normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
const MIN_OFFSET_VARIANCE: f64 = 1e-4;
/// Floor of the robust residual scale (log2): data whose majority of residuals is exactly zero
/// would otherwise get a zero loss scale.
const MIN_RESIDUAL_SCALE: f64 = 1e-3;
const VARIANCE_BINS: usize = 30;
/// Small data gets fewer bins, so that every bin variance rests on enough residuals.
const MIN_VARIANCE_BIN_SIZE: usize = 50;

const DEFAULT_MAX_SWEEPS: usize = 20;
const DEFAULT_MIN_IONS: usize = 2;
const DEFAULT_TOLERANCE: f64 = 1e-3;

/// The settings of the fit. The Python class always runs `DEFAULT`, which derives `tau` and
/// `lambda` from the data; the tests pin them to check the fit independently of the derivation.
pub struct FitConfig {
    pub tau: Option<f64>,
    pub lambda: Option<f64>,
    pub max_sweeps: usize,
    pub min_ions: usize,
    pub tolerance: f64,
}

impl FitConfig {
    pub const DEFAULT: Self = Self {
        tau: None,
        lambda: None,
        max_sweeps: DEFAULT_MAX_SWEEPS,
        min_ions: DEFAULT_MIN_IONS,
        tolerance: DEFAULT_TOLERANCE,
    };
}

#[derive(Clone, Copy)]
pub struct FitParameters {
    pub sigma: f64,
    pub tau: f64,
    pub lambda: f64,
}

pub struct FitResult {
    pub quantity: Vec<f64>,
    pub parameters: FitParameters,
}

pub fn validate_shapes(
    intensity_shape: (usize, usize),
    n_intensity_library: usize,
    precursor: &[usize],
    n_precursors: usize,
) -> Result<(), String> {
    let (_, n_ions) = intensity_shape;
    if n_intensity_library != n_ions || precursor.len() != n_ions {
        return Err(format!(
            "intensity has {n_ions} ions, intensity_library {n_intensity_library} and precursor {}",
            precursor.len()
        ));
    }
    if let Some(&index) = precursor.iter().find(|&&index| index >= n_precursors) {
        return Err(format!(
            "precursor index {index} out of range for {n_precursors} precursors"
        ));
    }
    Ok(())
}

struct Group {
    values: Vec<f64>,
    share: Vec<f64>,
    n_ions: usize,
}

impl Group {
    fn value(&self, ion: usize, run: usize, n_runs: usize) -> f64 {
        self.values[ion * n_runs + run]
    }
}

fn build_groups(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
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
            let share: Vec<f64> = ions
                .iter()
                .map(|&ion| intensity_library[ion] as f64 / intensity_library_sum)
                .collect();
            let mut values = Vec::with_capacity(ions.len() * n_runs);
            for (&ion, &s) in ions.iter().zip(&share) {
                for run in 0..n_runs {
                    let i = intensity[run * n_ions + ion] as f64;
                    values.push(if i > 0.0 && i.is_finite() {
                        (i / s).log2()
                    } else {
                        f64::NAN
                    });
                }
            }
            Group {
                n_ions: ions.len(),
                values,
                share,
            }
        })
        .collect()
}

fn run_quantities(group: &Group, weights: &[f64], offsets: &[f64], n_runs: usize) -> Vec<f64> {
    let mut numerator = vec![0.0; n_runs];
    let mut denominator = vec![0.0; n_runs];
    for ion in 0..group.n_ions {
        for run in 0..n_runs {
            let y = group.value(ion, run, n_runs);
            let w = weights[ion * n_runs + run];
            if y.is_finite() && w > 0.0 {
                numerator[run] += w * (y - offsets[ion]);
                denominator[run] += w;
            }
        }
    }
    numerator
        .iter()
        .zip(&denominator)
        .map(|(&n, &d)| if d > 0.0 { n / d } else { f64::NAN })
        .collect()
}

fn share_weights(group: &Group, n_runs: usize) -> Vec<f64> {
    let mut weights = Vec::with_capacity(group.n_ions * n_runs);
    for &share in &group.share {
        weights.extend(std::iter::repeat_n(share, n_runs));
    }
    weights
}

fn centered(group: &Group, quantity: &[f64], ion: usize, run: usize, n_runs: usize) -> Option<f64> {
    let y = group.value(ion, run, n_runs);
    let q = quantity[run];
    (y.is_finite() && q.is_finite()).then_some(y - q)
}

fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (_, &mut upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return upper;
    }
    let lower = values[..n / 2]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (lower + upper) / 2.0
}

fn residual_scale(residuals: &mut [f64]) -> f64 {
    if residuals.is_empty() {
        return MIN_RESIDUAL_SCALE;
    }
    let center = median(residuals);
    for r in residuals.iter_mut() {
        *r = (*r - center).abs();
    }
    (MAD_TO_SIGMA * median(residuals)).max(MIN_RESIDUAL_SCALE)
}

fn offset_shrinkage(raw_offsets: &[(f64, usize)], sigma: f64) -> f64 {
    let enough: Vec<&(f64, usize)> = raw_offsets.iter().filter(|(_, n)| *n >= 2).collect();
    if enough.is_empty() {
        return 0.0;
    }
    let count = enough.len() as f64;
    let mean = enough.iter().map(|(offset, _)| offset).sum::<f64>() / count;
    let variance = enough
        .iter()
        .map(|(offset, _)| (offset - mean).powi(2))
        .sum::<f64>()
        / count;
    let noise = enough
        .iter()
        .map(|(_, n)| sigma * sigma / *n as f64)
        .sum::<f64>()
        / count;
    sigma * sigma / (variance - noise).max(MIN_OFFSET_VARIANCE)
}

struct VarianceCurve {
    mu: Vec<f64>,
    log_variance: Vec<f64>,
}

impl VarianceCurve {
    fn from_residuals(mut measurements: Vec<(f64, f64)>) -> Self {
        measurements.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let n_bins = (measurements.len() / MIN_VARIANCE_BIN_SIZE).clamp(1, VARIANCE_BINS);
        let mut mu = Vec::with_capacity(n_bins);
        let mut log_variance = Vec::with_capacity(n_bins);
        for bin in 0..n_bins {
            let measurements = &measurements
                [bin * measurements.len() / n_bins..(bin + 1) * measurements.len() / n_bins];
            mu.push(
                measurements
                    .get(measurements.len() / 2)
                    .map_or(0.0, |&(mu, _)| mu),
            );
            let mut residuals: Vec<f64> =
                measurements.iter().map(|&(_, residual)| residual).collect();
            log_variance.push(2.0 * residual_scale(&mut residuals).ln());
        }
        // a measurement is at least as noisy as any brighter one
        for bin in (1..n_bins).rev() {
            log_variance[bin - 1] = log_variance[bin - 1].max(log_variance[bin]);
        }
        Self { mu, log_variance }
    }

    fn variance(&self, mu: f64) -> f64 {
        let (x, y) = (&self.mu, &self.log_variance);
        let upper = x.partition_point(|&m| m < mu);
        let log_variance = if upper == 0 {
            y[0]
        } else if upper == x.len() {
            y[upper - 1]
        } else {
            let t = (mu - x[upper - 1]) / (x[upper] - x[upper - 1]);
            y[upper - 1] + t * (y[upper] - y[upper - 1])
        };
        log_variance.exp()
    }
}

fn fit_group(
    group: &Group,
    mut quantity: Vec<f64>,
    curve: &VarianceCurve,
    parameters: FitParameters,
    config: &FitConfig,
    n_runs: usize,
) -> Vec<f64> {
    let FitParameters { sigma, tau, lambda } = parameters;
    let loss = (sigma / tau).powi(2);
    // fixed at the share-weighted first pass, so that the weights cannot chase the fit
    let mut variance = Vec::with_capacity(group.n_ions * n_runs);
    for &share in &group.share {
        let log_share = share.log2();
        variance.extend(quantity.iter().map(|&q| curve.variance(log_share + q)));
    }
    let mut offsets = vec![0.0; group.n_ions];
    let mut weights = vec![0.0; group.n_ions * n_runs];
    for _ in 0..config.max_sweeps {
        for ion in 0..group.n_ions {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for run in 0..n_runs {
                let measurement = ion * n_runs + run;
                match centered(group, &quantity, ion, run, n_runs) {
                    Some(deviation) => {
                        let e = deviation - offsets[ion];
                        let w = 1.0 / (variance[measurement] + loss * e * e);
                        weights[measurement] = w;
                        numerator += w * deviation;
                        denominator += w;
                    }
                    None => weights[measurement] = 0.0,
                }
            }
            // lambda counts observations of variance sigma ** 2
            offsets[ion] = numerator / (denominator + lambda / (sigma * sigma));
        }
        let previous = quantity;
        quantity = run_quantities(group, &weights, &offsets, n_runs);
        let largest_change = quantity
            .iter()
            .zip(&previous)
            .map(|(q, p)| (q - p).abs())
            .filter(|change| change.is_finite())
            .fold(0.0, f64::max);
        if largest_change < config.tolerance {
            break;
        }
    }
    quantity
}

pub fn fit(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    n_runs: usize,
    n_precursors: usize,
    config: &FitConfig,
) -> FitResult {
    let groups = build_groups(
        intensity,
        intensity_library,
        precursor,
        n_runs,
        n_precursors,
    );

    let initial: Vec<Vec<f64>> = groups
        .par_iter()
        .map(|group| {
            run_quantities(
                group,
                &share_weights(group, n_runs),
                &vec![0.0; group.n_ions],
                n_runs,
            )
        })
        .collect();

    // Residuals around the unshrunk offsets, so that the library offsets do not count as noise,
    // scaled by sqrt(n / (n - 1)) for the offset fitted to the same n measurements.
    let (measurements, raw_offsets): (Vec<(f64, f64)>, Vec<(f64, usize)>) = groups
        .par_iter()
        .zip(&initial)
        .map(|(group, quantity)| {
            let mut residuals = Vec::new();
            let mut raw_offsets = Vec::with_capacity(group.n_ions);
            let mut deviations = Vec::with_capacity(n_runs);
            for ion in 0..group.n_ions {
                deviations.clear();
                deviations.extend((0..n_runs).filter_map(|run| {
                    centered(group, quantity, ion, run, n_runs).map(|d| (quantity[run], d))
                }));
                let n = deviations.len();
                let raw_offset = deviations.iter().map(|(_, d)| d).sum::<f64>() / n.max(1) as f64;
                if n >= 2 {
                    let scale = (n as f64 / (n - 1) as f64).sqrt();
                    let log_share = group.share[ion].log2();
                    residuals.extend(
                        deviations
                            .iter()
                            .map(|(q, d)| (log_share + q, (d - raw_offset) * scale)),
                    );
                }
                raw_offsets.push((raw_offset, n));
            }
            (residuals, raw_offsets)
        })
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut residuals, mut offsets), (r, o)| {
                residuals.extend(r);
                offsets.extend(o);
                (residuals, offsets)
            },
        );
    let sigma = residual_scale(
        &mut measurements
            .iter()
            .map(|&(_, residual)| residual)
            .collect::<Vec<_>>(),
    );
    let parameters = FitParameters {
        sigma,
        tau: config.tau.unwrap_or(CAUCHY_TUNING_CONSTANT * sigma),
        lambda: config
            .lambda
            .unwrap_or_else(|| offset_shrinkage(&raw_offsets, sigma)),
    };
    let curve = VarianceCurve::from_residuals(measurements);

    let fitted: Vec<Vec<f64>> = groups
        .par_iter()
        .zip(initial)
        .map(|(group, quantity)| {
            let quantity = fit_group(group, quantity, &curve, parameters, config, n_runs);
            let mut n_observed = vec![0usize; n_runs];
            for ion in 0..group.n_ions {
                for (run, count) in n_observed.iter_mut().enumerate() {
                    if group.value(ion, run, n_runs).is_finite() {
                        *count += 1;
                    }
                }
            }
            quantity
                .into_iter()
                .zip(n_observed)
                .map(|(q, n)| if n >= config.min_ions { q } else { f64::NAN })
                .collect()
        })
        .collect();

    let mut quantity = vec![f64::NAN; n_runs * n_precursors];
    for (index, group_quantity) in fitted.iter().enumerate() {
        for (run, &q) in group_quantity.iter().enumerate() {
            quantity[run * n_precursors + index] = q;
        }
    }
    FitResult {
        quantity,
        parameters,
    }
}
