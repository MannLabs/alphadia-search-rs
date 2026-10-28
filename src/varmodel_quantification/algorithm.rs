use rayon::prelude::*;

use super::noise::{FitParameters, NoiseModel, VarianceCurve};

const DEFAULT_MAX_SWEEPS: usize = 20;
const DEFAULT_MIN_IONS: usize = 2;
const DEFAULT_TOLERANCE: f64 = 1e-3;

/// `tau` and `lambda` are derived from the data unless set.
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

/// Run quantities, run-major `(n_runs, n_precursors)`, NaN where too few fragments were observed.
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
    let initial: Vec<Vec<f64>> = groups.par_iter().map(initial_quantities).collect();
    let noise = NoiseModel::estimate(&groups, &initial, config);
    let fitted: Vec<Vec<f64>> = groups
        .par_iter()
        .zip(initial)
        .map(|(group, quantity)| {
            let quantity = fit_group(group, quantity, &noise, config);
            group.mask_underdetermined(quantity, config.min_ions)
        })
        .collect();
    FitResult {
        quantity: run_major(&fitted, n_runs, n_precursors),
        parameters: noise.parameters,
    }
}

/// One precursor: the library share `s` of each fragment with a library intensity, and its
/// fragment measurements as `log2(I / s)`, ion-major, NaN where missing.
/// Arrays with one entry per fragment measurement are f32 to bound memory; sums and scalars
/// are f64.
pub struct Group {
    values: Vec<f32>,
    pub share: Vec<f64>,
    pub n_ions: usize,
    pub n_runs: usize,
}

impl Group {
    fn index(&self, ion: usize, run: usize) -> usize {
        ion * self.n_runs + run
    }

    fn value(&self, ion: usize, run: usize) -> f64 {
        self.values[self.index(ion, run)] as f64
    }

    pub fn deviation(&self, ion: usize, run: usize, quantity: &[f64]) -> Option<f64> {
        let y = self.value(ion, run);
        let q = quantity[run];
        (y.is_finite() && q.is_finite()).then_some(y - q)
    }

    fn mask_underdetermined(&self, quantity: Vec<f64>, min_ions: usize) -> Vec<f64> {
        quantity
            .into_iter()
            .enumerate()
            .map(|(run, q)| {
                let n_observed = (0..self.n_ions)
                    .filter(|&ion| self.value(ion, run).is_finite())
                    .count();
                if n_observed >= min_ions {
                    q
                } else {
                    f64::NAN
                }
            })
            .collect()
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
                        (i / s).log2() as f32
                    } else {
                        f32::NAN
                    });
                }
            }
            Group {
                values,
                share,
                n_ions: ions.len(),
                n_runs,
            }
        })
        .collect()
}

/// Step 1 of the procedure: the library-share weighted mean, without fragment offsets.
fn initial_quantities(group: &Group) -> Vec<f64> {
    let mut weights = Vec::with_capacity(group.n_ions * group.n_runs);
    for &share in &group.share {
        weights.extend(std::iter::repeat_n(share as f32, group.n_runs));
    }
    run_quantities(group, &weights, &vec![0.0; group.n_ions])
}

/// Step 3 of the procedure: alternate weights, fragment offsets and run quantities.
fn fit_group(
    group: &Group,
    mut quantity: Vec<f64>,
    noise: &NoiseModel,
    config: &FitConfig,
) -> Vec<f64> {
    let FitParameters { sigma, tau, lambda } = noise.parameters;
    let loss = (sigma / tau).powi(2);
    // lambda counts observations of variance sigma ** 2
    let prior_weight = lambda / (sigma * sigma);
    // fixed at the share-weighted first pass, so that the weights cannot chase the fit
    let variance = measurement_variances(group, &quantity, &noise.variance);
    let mut offsets = vec![0.0; group.n_ions];
    let mut weights = vec![0.0f32; group.n_ions * group.n_runs];
    for _ in 0..config.max_sweeps {
        update_weights(group, &quantity, &offsets, &variance, loss, &mut weights);
        update_offsets(group, &quantity, &weights, prior_weight, &mut offsets);
        let previous = quantity;
        quantity = run_quantities(group, &weights, &offsets);
        if largest_change(&quantity, &previous) < config.tolerance {
            break;
        }
    }
    quantity
}

fn measurement_variances(group: &Group, quantity: &[f64], curve: &VarianceCurve) -> Vec<f32> {
    let mut variance = Vec::with_capacity(group.n_ions * group.n_runs);
    for &share in &group.share {
        let log_share = share.log2();
        variance.extend(
            quantity
                .iter()
                .map(|&q| curve.variance(log_share + q) as f32),
        );
    }
    variance
}

/// Inverse variance, downweighted by the Cauchy loss of the residual.
fn update_weights(
    group: &Group,
    quantity: &[f64],
    offsets: &[f64],
    variance: &[f32],
    loss: f64,
    weights: &mut [f32],
) {
    for ion in 0..group.n_ions {
        for run in 0..group.n_runs {
            let index = group.index(ion, run);
            weights[index] = match group.deviation(ion, run, quantity) {
                Some(deviation) => {
                    let e = deviation - offsets[ion];
                    (1.0 / (variance[index] as f64 + loss * e * e)) as f32
                }
                None => 0.0,
            };
        }
    }
}

fn update_offsets(
    group: &Group,
    quantity: &[f64],
    weights: &[f32],
    prior_weight: f64,
    offsets: &mut [f64],
) {
    for (ion, offset) in offsets.iter_mut().enumerate() {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for run in 0..group.n_runs {
            if let Some(deviation) = group.deviation(ion, run, quantity) {
                let w = weights[group.index(ion, run)] as f64;
                numerator += w * deviation;
                denominator += w;
            }
        }
        *offset = numerator / (denominator + prior_weight);
    }
}

fn run_quantities(group: &Group, weights: &[f32], offsets: &[f64]) -> Vec<f64> {
    let mut numerator = vec![0.0; group.n_runs];
    let mut denominator = vec![0.0; group.n_runs];
    for ion in 0..group.n_ions {
        for run in 0..group.n_runs {
            let y = group.value(ion, run);
            let w = weights[group.index(ion, run)] as f64;
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

fn largest_change(quantity: &[f64], previous: &[f64]) -> f64 {
    quantity
        .iter()
        .zip(previous)
        .map(|(q, p)| (q - p).abs())
        .filter(|change| change.is_finite())
        .fold(0.0, f64::max)
}

fn run_major(fitted: &[Vec<f64>], n_runs: usize, n_precursors: usize) -> Vec<f64> {
    let mut quantity = vec![f64::NAN; n_runs * n_precursors];
    for (index, group_quantity) in fitted.iter().enumerate() {
        for (run, &q) in group_quantity.iter().enumerate() {
            quantity[run * n_precursors + index] = q;
        }
    }
    quantity
}
