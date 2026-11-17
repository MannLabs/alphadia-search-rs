//! The quality exponent gamma of the variance factor `exp(gamma * x)`, `x = 1 - sqrt(C)`,
//! estimated from the standardised residuals of the fit without quality weights.

use rayon::prelude::*;

use super::censored::{Group, State};

/// Residuals of precursor runs with fewer observed fragments are dominated by the fit's own
/// leverage.
const MIN_OBSERVED: usize = 4;
const INTENSITY_BINS: usize = 20;
const QUALITY_BINS: usize = 10;
/// A (intensity, quality) bin enters the slope with at least this many measurements.
const MIN_CELLS: usize = 30;
pub const MAX_QUALITY_EXPONENT: f64 = 2.0;
const PERCENT: f64 = 100.0;

/// An observed measurement: expected log2 intensity, quality x and squared standardised
/// residual.
struct Cell {
    expected: f64,
    quality: f64,
    z2: f64,
}

/// gamma in `[0, MAX_QUALITY_EXPONENT]` from the fit without quality weights (`states`, with
/// the variances of `groups` at gamma 0); 0 when the data give no slope.
///
/// Within bins of the expected intensity, `log median z^2 = a_bin + gamma * x`: medians of a
/// scale family scale with the variance, so the slope needs no distribution constant. The
/// measurements are binned by intensity quantiles, then by quality quantiles within each
/// intensity bin, and gamma is the count-weighted within-bin least-squares slope of the bins'
/// log median `z^2` on their mean `x`.
pub(super) fn estimate_quality_exponent(
    groups: &[Group],
    initial: &[Vec<f64>],
    states: &[State],
) -> f64 {
    let cells: Vec<Cell> = groups
        .par_iter()
        .zip(initial)
        .zip(states)
        .map(|((group, initial), state)| standardised_residuals(group, initial, state))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();
    let gamma = slope_of_log_median(&cells);
    if gamma.is_finite() {
        gamma.clamp(0.0, MAX_QUALITY_EXPONENT)
    } else {
        0.0
    }
}

/// `z^2 = r^2 / (v (1 - h) (1 - 1/m))` of the observed measurements of precursor runs with at
/// least `MIN_OBSERVED` fragments: `r` the residual, `v` the noise variance at the
/// share-weighted first pass, `h` the measurement's share of its precursor run's weight `1/v`,
/// `m` the runs its fragment is observed in (the offset is fitted from them).
fn standardised_residuals(group: &Group, initial: &[f64], state: &State) -> Vec<Cell> {
    let (n_ions, n_runs) = (group.n_ions, group.n_runs);
    let mut weight_sum = vec![0.0; n_runs];
    for ion in 0..n_ions {
        for run in 0..n_runs {
            if group.observed(ion, run) {
                weight_sum[run] += 1.0 / group.variance[group.index(ion, run)];
            }
        }
    }
    let usable: Vec<bool> = (0..n_runs)
        .map(|run| group.n_observed(run) >= MIN_OBSERVED)
        .collect();
    let mut cells = Vec::new();
    for ion in 0..n_ions {
        let m = (0..n_runs).filter(|&run| group.observed(ion, run)).count();
        if m < 2 {
            continue;
        }
        for run in 0..n_runs {
            if !usable[run] || !group.observed(ion, run) {
                continue;
            }
            let index = group.index(ion, run);
            let variance = group.variance[index];
            let leverage = (1.0 / variance) / weight_sum[run];
            let residual = group.y[index] - state.quantity[run] - state.offset[ion];
            let z2 = residual * residual / (variance * (1.0 - leverage) * (1.0 - 1.0 / m as f64));
            if z2.is_finite() {
                cells.push(Cell {
                    expected: group.log_share[ion] + initial[run],
                    quality: group.quality[index],
                    z2,
                });
            }
        }
    }
    cells
}

/// Edges at the `INTENSITY_BINS` quantiles of the sorted values (linear interpolation),
/// duplicates dropped.
fn quantile_edges(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    let mut edges: Vec<f64> = (0..=INTENSITY_BINS)
        .map(|i| {
            let probability = if i == INTENSITY_BINS {
                1.0
            } else {
                i as f64 * (1.0 / INTENSITY_BINS as f64)
            };
            // the probability passes through percent, as the quantile binning does
            let probability = probability * PERCENT / PERCENT;
            let virtual_index = (n - 1) as f64 * probability;
            if virtual_index >= (n - 1) as f64 {
                return sorted[n - 1];
            }
            let previous = virtual_index.floor();
            let gamma = virtual_index - previous;
            let (a, b) = (sorted[previous as usize], sorted[previous as usize + 1]);
            let difference = b - a;
            if gamma >= 0.5 {
                b - difference * (1.0 - gamma)
            } else {
                a + difference * gamma
            }
        })
        .collect();
    edges.dedup();
    edges
}

/// Right-closed bins `(e_i, e_i+1]`, the first one closed on both sides.
fn intensity_bin(value: f64, edges: &[f64]) -> usize {
    if value == edges[0] {
        return 0;
    }
    edges.partition_point(|&edge| edge < value) - 1
}

/// Quality bin of every measurement of one intensity bin, by its average rank in x.
fn quality_bins(quality: &[f64]) -> Vec<usize> {
    let n = quality.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| quality[a].total_cmp(&quality[b]));
    let mut bins = vec![0; n];
    let mut start = 0;
    while start < n {
        let mut end = start;
        while end + 1 < n && quality[order[end + 1]] == quality[order[start]] {
            end += 1;
        }
        let rank = ((start + 1) + (end + 1)) as f64 / 2.0;
        let bin = ((rank / n as f64 * QUALITY_BINS as f64) as usize).min(QUALITY_BINS - 1);
        for &i in &order[start..=end] {
            bins[i] = bin;
        }
        start = end + 1;
    }
    bins
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn slope_of_log_median(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = cells.iter().map(|cell| cell.expected).collect();
    sorted.sort_by(f64::total_cmp);
    let edges = quantile_edges(&sorted);
    let n_intensity_bins = (edges.len() - 1).max(1);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_intensity_bins];
    for (index, cell) in cells.iter().enumerate() {
        members[intensity_bin(cell.expected, &edges).min(n_intensity_bins - 1)].push(index);
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for members in &members {
        let quality: Vec<f64> = members.iter().map(|&i| cells[i].quality).collect();
        let bins = quality_bins(&quality);
        // (count, mean x, log median z^2) per quality bin with enough measurements
        let mut table = Vec::new();
        for bin in 0..QUALITY_BINS {
            let in_bin: Vec<usize> = (0..members.len()).filter(|&k| bins[k] == bin).collect();
            if in_bin.len() < MIN_CELLS {
                continue;
            }
            let count = in_bin.len() as f64;
            let mean_x = in_bin.iter().map(|&k| quality[k]).sum::<f64>() / count;
            let mut z2: Vec<f64> = in_bin.iter().map(|&k| cells[members[k]].z2).collect();
            table.push((count, mean_x, median(&mut z2).ln()));
        }
        let total: f64 = table.iter().map(|row| row.0).sum();
        if total == 0.0 {
            continue;
        }
        let x_centre = table.iter().map(|row| row.0 * row.1).sum::<f64>() / total;
        let l_centre = table.iter().map(|row| row.0 * row.2).sum::<f64>() / total;
        for &(count, x, l) in &table {
            numerator += count * (x - x_centre) * (l - l_centre);
            denominator += count * (x - x_centre) * (x - x_centre);
        }
    }
    numerator / denominator
}
