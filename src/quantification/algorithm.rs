//! Library-anchored fit of the precursor run quantities. `DOCS.md` defines the terms used here
//! and gives the algorithm.
//!
//! The intensity matrix holds `f32` like the fragment tables of the caller. The fit is computed
//! in `f64`.

use rayon::prelude::*;

/// Cauchy loss tuning constant for 95 % efficiency at the normal distribution (Holland & Welsch 1977).
const CAUCHY_TUNING_CONSTANT: f64 = 2.385;
/// Robust scale: MAD to standard deviation at the normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
const MIN_OFFSET_VARIANCE: f64 = 1e-4;
/// Floor of the robust residual scale (log2): data whose majority of residuals is exactly zero
/// would otherwise get a zero loss scale.
const MIN_RESIDUAL_SCALE: f64 = 1e-3;

pub const DEFAULT_MAX_SWEEPS: usize = 20;
pub const DEFAULT_MIN_IONS: usize = 2;
pub const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Parameters of the fit, see `DOCS.md` for the terms.
pub struct FitParameters {
    /// Scale of the Cauchy loss in log2, `None` to derive it from the data.
    pub tau: Option<f64>,
    /// Shrinkage of the fragment offsets towards the library pattern, `None` to derive it.
    pub lambda: Option<f64>,
    /// Maximum number of reweighting sweeps per precursor.
    pub max_sweeps: usize,
    /// Minimum observed fragments per precursor and run for a defined quantity.
    pub min_ions: usize,
    /// Largest change of any quantity of a precursor (log2) at which its sweeps stop.
    pub tolerance: f64,
}

impl FitParameters {
    // the Python constructor spells out every field, the tests vary single ones
    #[cfg(test)]
    pub const DEFAULT: Self = Self {
        tau: None,
        lambda: None,
        max_sweeps: DEFAULT_MAX_SWEEPS,
        min_ions: DEFAULT_MIN_IONS,
        tolerance: DEFAULT_TOLERANCE,
    };

    pub fn validate(&self) -> Result<(), String> {
        if let Some(tau) = self.tau {
            if !(tau.is_finite() && tau > 0.0) {
                return Err(format!("tau must be a positive number, got {tau}"));
            }
        }
        if let Some(lambda) = self.lambda {
            if !(lambda.is_finite() && lambda >= 0.0) {
                return Err(format!(
                    "lambda must be a non-negative number, got {lambda}"
                ));
            }
        }
        if self.min_ions == 0 {
            return Err("min_ions must be at least 1".to_string());
        }
        if !(self.tolerance.is_finite() && self.tolerance >= 0.0) {
            return Err(format!(
                "tolerance must be a non-negative number, got {}",
                self.tolerance
            ));
        }
        Ok(())
    }
}

/// Result of the fit.
pub struct FitResult {
    /// Run-major `(n_runs, n_precursors)` log2 quantities, NaN where undefined.
    pub quantity: Vec<f64>,
    pub residual_scale: f64,
    pub loss_scale: f64,
    pub offset_shrinkage: f64,
}

/// Checks that the arrays agree in shape and every precursor index is in range.
pub fn validate_shapes(
    intensity_shape: (usize, usize),
    n_library: usize,
    precursor: &[usize],
    n_precursors: usize,
) -> Result<(), String> {
    let (_, n_ions) = intensity_shape;
    if n_library != n_ions || precursor.len() != n_ions {
        return Err(format!(
            "intensity has {n_ions} ions, library {n_library} and precursor {}",
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

/// The fragments of one precursor: ion-major `y = log2(I / s)` (NaN where missing) and the
/// base weights `s`.
struct Group {
    values: Vec<f64>,
    base: Vec<f64>,
    n_ions: usize,
}

impl Group {
    fn value(&self, ion: usize, run: usize, n_runs: usize) -> f64 {
        self.values[ion * n_runs + run]
    }
}

/// Builds one group per precursor from the run-major intensity matrix.
fn build_groups(
    intensity: &[f32],
    library: &[f32],
    precursor: &[usize],
    n_runs: usize,
    n_precursors: usize,
) -> Vec<Group> {
    let n_ions = library.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_precursors];
    for (ion, &index) in precursor.iter().enumerate() {
        let l = library[ion] as f64;
        if l.is_finite() && l > 0.0 {
            members[index].push(ion);
        }
    }
    members
        .into_par_iter()
        .map(|ions| {
            let library_sum: f64 = ions.iter().map(|&ion| library[ion] as f64).sum();
            let base: Vec<f64> = ions
                .iter()
                .map(|&ion| library[ion] as f64 / library_sum)
                .collect();
            let mut values = Vec::with_capacity(ions.len() * n_runs);
            for (&ion, &share) in ions.iter().zip(&base) {
                for run in 0..n_runs {
                    let i = intensity[run * n_ions + ion] as f64;
                    values.push(if i > 0.0 && i.is_finite() {
                        (i / share).log2()
                    } else {
                        f64::NAN
                    });
                }
            }
            Group {
                n_ions: ions.len(),
                values,
                base,
            }
        })
        .collect()
}

/// Weighted mean over the fragments of `y - offset` per run, NaN where no weight.
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

fn base_weights(group: &Group, n_runs: usize) -> Vec<f64> {
    let mut weights = Vec::with_capacity(group.n_ions * n_runs);
    for &base in &group.base {
        weights.extend(std::iter::repeat_n(base, n_runs));
    }
    weights
}

/// Deviation `y - q` of an observed cell, None where the cell or its quantity is missing.
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

/// Robust standard deviation (MAD-based) of the first-sweep residuals around the fragment
/// offsets, floored.
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

/// Random-effects shrinkage of the fragment offsets: residual variance / offset variance.
///
/// The offset variance is the variance of the unshrunk per-fragment mean deviations minus the
/// noise they carry (method of moments), floored at `MIN_OFFSET_VARIANCE`.
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

/// Reweighting sweeps of one precursor from its unweighted quantities.
fn fit_group(
    group: &Group,
    mut quantity: Vec<f64>,
    tau: f64,
    lambda: f64,
    params: &FitParameters,
    n_runs: usize,
) -> Vec<f64> {
    let mut offsets = vec![0.0; group.n_ions];
    let mut weights = vec![0.0; group.n_ions * n_runs];
    for _ in 0..params.max_sweeps {
        for ion in 0..group.n_ions {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for run in 0..n_runs {
                let cell = ion * n_runs + run;
                match centered(group, &quantity, ion, run, n_runs) {
                    Some(deviation) => {
                        let residual = (deviation - offsets[ion]) / tau;
                        let w = group.base[ion] / (1.0 + residual * residual);
                        weights[cell] = w;
                        numerator += w * deviation;
                        denominator += w;
                    }
                    None => weights[cell] = 0.0,
                }
            }
            // lambda counts observations of unit weight, a cell of this fragment weighs base[ion]
            offsets[ion] = numerator / (denominator + lambda * group.base[ion]);
        }
        let previous = quantity;
        quantity = run_quantities(group, &weights, &offsets, n_runs);
        let largest_change = quantity
            .iter()
            .zip(&previous)
            .map(|(q, p)| (q - p).abs())
            .filter(|change| change.is_finite())
            .fold(0.0, f64::max);
        if largest_change < params.tolerance {
            break;
        }
    }
    quantity
}

/// Fits the run quantities of every precursor, see `DOCS.md`.
pub fn fit(
    intensity: &[f32],
    library: &[f32],
    precursor: &[usize],
    n_runs: usize,
    n_precursors: usize,
    params: &FitParameters,
) -> FitResult {
    let groups = build_groups(intensity, library, precursor, n_runs, n_precursors);

    let initial: Vec<Vec<f64>> = groups
        .par_iter()
        .map(|group| {
            run_quantities(
                group,
                &base_weights(group, n_runs),
                &vec![0.0; group.n_ions],
                n_runs,
            )
        })
        .collect();

    // residuals and unshrunk offsets of the first, unweighted sweep over all precursors. The
    // residuals are taken around the unshrunk offsets, so that the library offsets do not count
    // as noise, and scaled by sqrt(n / (n - 1)) for the offset fitted to the same n cells.
    // Fragments observed once have no residual around their offset.
    let (mut residuals, raw_offsets): (Vec<f64>, Vec<(f64, usize)>) = groups
        .par_iter()
        .zip(&initial)
        .map(|(group, quantity)| {
            let mut residuals = Vec::new();
            let mut raw_offsets = Vec::with_capacity(group.n_ions);
            let mut deviations = Vec::with_capacity(n_runs);
            for ion in 0..group.n_ions {
                deviations.clear();
                deviations.extend(
                    (0..n_runs).filter_map(|run| centered(group, quantity, ion, run, n_runs)),
                );
                let n = deviations.len();
                let raw_offset = deviations.iter().sum::<f64>() / n.max(1) as f64;
                if n >= 2 {
                    let scale = (n as f64 / (n - 1) as f64).sqrt();
                    residuals.extend(deviations.iter().map(|d| (d - raw_offset) * scale));
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
    let sigma = residual_scale(&mut residuals);
    let tau = params.tau.unwrap_or(CAUCHY_TUNING_CONSTANT * sigma);
    let lambda = params
        .lambda
        .unwrap_or_else(|| offset_shrinkage(&raw_offsets, sigma));

    let fitted: Vec<Vec<f64>> = groups
        .par_iter()
        .zip(initial)
        .map(|(group, quantity)| {
            let quantity = fit_group(group, quantity, tau, lambda, params, n_runs);
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
                .map(|(q, n)| if n >= params.min_ions { q } else { f64::NAN })
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
        residual_scale: sigma,
        loss_scale: tau,
        offset_shrinkage: lambda,
    }
}
