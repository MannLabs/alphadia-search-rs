//! Numeric core of the intensity drift correction.
//!
//! Intensities can drift along the gradient in a run-specific way (e.g. one condition being
//! 0.3-0.6 log2 low after 1000 s). Per run, the log2 deviation from the across-run level of
//! every ion is binned along retention time, the location of each bin is interpolated to
//! every ion and divided out.
//!
//! The curve is re-centred to zero mean over the run's ions before it is applied, so only
//! the *shape* along the gradient is removed. The level of a run is deliberately left to the
//! downstream label-free quantification; removing it here would silently force the majority
//! species to ratio 1.

use std::ops::Range;

use pyo3::prelude::*;
use rayon::prelude::*;

/// Histogram the mode is read from: bin width in log2 and the central percentile range that
/// keeps a few outliers from stretching the histogram.
const MODE_BIN_WIDTH: f64 = 0.02;
const MODE_PERCENTILE_LOW: f64 = 2.0;
const MODE_PERCENTILE_HIGH: f64 = 98.0;
/// Window the mode histogram is smoothed over. A distribution narrower than this window has
/// no resolvable mode.
const MODE_SMOOTHING_BINS: usize = 11;
/// Window the curve is smoothed over its retention time bins with.
const CURVE_SMOOTHING_BINS: usize = 3;

/// Per-bin location estimator of the deviation.
#[derive(Clone, Copy)]
pub enum Statistic {
    /// Mode of a smoothed histogram. Follows the unregulated majority of the ions even when
    /// a part of the proteome changes between the runs.
    Mode,
    /// Median of the bin.
    Median,
}

impl Statistic {
    /// The identifiers the caller selects a statistic by.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "mode" => Ok(Self::Mode),
            "median" => Ok(Self::Median),
            _ => Err(format!(
                "unknown intensity drift statistic '{name}', expected 'mode' or 'median'"
            )),
        }
    }
}

/// Configuration of the estimator. The defaults live with the caller.
pub struct DriftParameters {
    /// Width of the fixed retention time bins the curve is estimated in.
    pub bin_seconds: f64,
    /// Minimum mean cross-run correlation for an ion to shape a curve.
    pub min_correlation: f64,
    /// Minimum number of runs an ion must be observed in to shape a curve.
    pub min_observations: usize,
    /// Consecutive sparse bins are merged until they hold this many ions.
    pub min_ions_per_bin: usize,
    /// Runs with fewer usable ions are left unchanged.
    pub min_ions_per_run: usize,
    pub statistic: Statistic,
}

/// The log2 correction of one run, at the mean retention time of every merged bin.
#[pyclass(frozen, get_all)]
pub struct DriftCurve {
    /// Ions the curve was estimated from.
    pub n_ions: usize,
    /// Mean retention time of every merged bin, ascending.
    pub rt: Vec<f64>,
    /// Re-centred log2 correction of every merged bin.
    pub correction: Vec<f64>,
}

impl DriftCurve {
    /// The log2 correction at `rt`, linear between the bins and constant outside them.
    ///
    /// An ion that no run places in retention time is not corrected.
    fn at(&self, rt: f64) -> f64 {
        if rt.is_finite() {
            interpolate(rt, &self.rt, &self.correction)
        } else {
            0.0
        }
    }
}

/// Corrects the retention time dependent intensity drift of every run.
///
/// The matrices are run-major: row `run` of each holds the `n_ions` cells of that run.
/// `intensity` is linear and zero marks a missing value, `correlation` holds the cross-run
/// fragment correlations and `rt` the observed retention time of every ion's precursor, NaN
/// where the run has no identification for it. `n_ions` must not be zero.
///
/// Gives the corrected intensities in the same layout, and the curve of every run with
/// `None` for the runs that were left unchanged.
pub fn correct_drift(
    intensity: &[f64],
    correlation: &[f64],
    rt: &[f64],
    n_ions: usize,
    parameters: &DriftParameters,
) -> (Vec<f64>, Vec<Option<DriftCurve>>) {
    let reference = reference_levels(intensity, correlation, n_ions, parameters);
    let mean_rt = mean_retention_times(rt, n_ions);

    let mut curves: Vec<Option<DriftCurve>> = intensity
        .par_chunks_exact(n_ions)
        .zip(rt.par_chunks_exact(n_ions))
        .map(|(run_intensity, run_rt)| {
            estimate_curve(run_intensity, run_rt, &reference, &mean_rt, parameters)
        })
        .collect();

    let mut corrected = intensity.to_vec();
    corrected
        .par_chunks_exact_mut(n_ions)
        .zip(rt.par_chunks_exact(n_ions))
        .zip(curves.par_iter_mut())
        .for_each(|((run_corrected, run_rt), curve)| {
            if let Some(curve) = curve {
                apply_curve(run_corrected, run_rt, &mean_rt, curve);
            }
        });

    (corrected, curves)
}

/// The across-run log2 level of every ion, NaN for the ions that must not shape a curve.
///
/// An ion only shapes a curve if it was observed in `min_observations` runs and its mean
/// cross-run correlation reaches `min_correlation`: a poorly correlating fragment carries
/// interference rather than the drift of the run. Ions without a level are still corrected.
fn reference_levels(
    intensity: &[f64],
    correlation: &[f64],
    n_ions: usize,
    parameters: &DriftParameters,
) -> Vec<f64> {
    let n_runs = intensity.len() / n_ions;
    (0..n_ions)
        .into_par_iter()
        .map_init(
            || Vec::with_capacity(n_runs),
            |levels: &mut Vec<f64>, ion| {
                levels.clear();
                let mut correlation_sum = 0.0;
                for run in 0..n_runs {
                    let cell = run * n_ions + ion;
                    if intensity[cell] > 0.0 {
                        levels.push(intensity[cell].log2());
                        // a missing correlation counts as no correlation
                        if correlation[cell].is_finite() {
                            correlation_sum += correlation[cell];
                        }
                    }
                }
                if levels.len() < parameters.min_observations
                    || correlation_sum / (levels.len() as f64) < parameters.min_correlation
                {
                    return f64::NAN;
                }
                levels.sort_by(f64::total_cmp);
                percentile_of_sorted(levels, 50.0)
            },
        )
        .collect()
}

/// Mean retention time of every ion over the runs that identified it, NaN if none did.
fn mean_retention_times(rt: &[f64], n_ions: usize) -> Vec<f64> {
    let n_runs = rt.len() / n_ions;
    (0..n_ions)
        .into_par_iter()
        .map(|ion| {
            let mut sum = 0.0;
            let mut n_runs_with_rt = 0;
            for run in 0..n_runs {
                let value = rt[run * n_ions + ion];
                if value.is_finite() {
                    sum += value;
                    n_runs_with_rt += 1;
                }
            }
            if n_runs_with_rt == 0 {
                f64::NAN
            } else {
                sum / n_runs_with_rt as f64
            }
        })
        .collect()
}

/// The curve of one run, `None` if the run cannot be estimated and is left unchanged.
fn estimate_curve(
    intensity: &[f64],
    rt: &[f64],
    reference: &[f64],
    mean_rt: &[f64],
    parameters: &DriftParameters,
) -> Option<DriftCurve> {
    // retention time and log2 deviation from the across-run level of the ions that shape the
    // curve, ordered along the gradient
    let mut points = Vec::new();
    for ion in 0..intensity.len() {
        let ion_rt = ion_rt(rt[ion], mean_rt[ion]);
        if intensity[ion] > 0.0 && reference[ion].is_finite() && ion_rt.is_finite() {
            points.push((ion_rt, intensity[ion].log2() - reference[ion]));
        }
    }
    if points.len() < parameters.min_ions_per_run {
        return None;
    }
    points.sort_by(|left, right| left.0.total_cmp(&right.0));

    let (bin_rt, location) = bin_points(&points, parameters);
    // a single bin carries no shape along the gradient
    if bin_rt.len() < 2 {
        return None;
    }
    Some(DriftCurve {
        n_ions: points.len(),
        rt: bin_rt,
        correction: moving_average(&location, CURVE_SMOOTHING_BINS),
    })
}

/// Divides the curve out of one run, then multiplies its mean over the run's ions back in.
///
/// Only the shape along the gradient is removed this way; the level of a run stays with the
/// downstream normalization. The bin curve is re-centred by the same mean, so that it reports
/// what was applied.
fn apply_curve(intensity: &mut [f64], rt: &[f64], mean_rt: &[f64], curve: &mut DriftCurve) {
    let mut correction_sum = 0.0;
    let mut n_corrected = 0;
    for ion in 0..intensity.len() {
        if intensity[ion] <= 0.0 {
            continue;
        }
        let correction = curve.at(ion_rt(rt[ion], mean_rt[ion]));
        intensity[ion] /= correction.exp2();
        correction_sum += correction;
        n_corrected += 1;
    }

    debug_assert!(n_corrected > 0, "a curve is estimated from observed ions");
    let centre = correction_sum / n_corrected as f64;
    let level = centre.exp2();
    intensity
        .iter_mut()
        .filter(|intensity| **intensity > 0.0)
        .for_each(|intensity| *intensity *= level);
    curve
        .correction
        .iter_mut()
        .for_each(|correction| *correction -= centre);
}

/// Retention time of an ion in one run.
///
/// A precursor without an identification in a run is placed at its mean retention time over
/// the other runs, so that a cell with an intensity but no identification is still corrected.
fn ion_rt(rt: f64, mean_rt: f64) -> f64 {
    if rt.is_finite() {
        rt
    } else {
        mean_rt
    }
}

/// Mean retention time and deviation location of every merged bin.
fn bin_points(points: &[(f64, f64)], parameters: &DriftParameters) -> (Vec<f64>, Vec<f64>) {
    let edges = bin_edges(points, parameters);
    let mut bin_rt = Vec::with_capacity(edges.len() - 1);
    let mut location = Vec::with_capacity(edges.len() - 1);
    let mut deviation = Vec::new();
    for edge in edges.windows(2) {
        let bin = &points[edge[0]..edge[1]];
        bin_rt.push(bin.iter().map(|&(rt, _)| rt).sum::<f64>() / bin.len() as f64);
        deviation.clear();
        deviation.extend(bin.iter().map(|&(_, deviation)| deviation));
        location.push(location_of(&mut deviation, parameters.statistic));
    }
    (bin_rt, location)
}

/// Start of every merged bin, plus the end of the last one.
///
/// The fixed `bin_seconds` bins of `points` are merged with their successors until they hold
/// `min_ions_per_bin` ions. The trailing ions that do not fill a bin join the last one.
fn bin_edges(points: &[(f64, f64)], parameters: &DriftParameters) -> Vec<usize> {
    let n_points = points.len();
    let fixed_bin = |point: usize| (points[point].0 / parameters.bin_seconds).floor();

    let mut edges = vec![0];
    for stop in 1..=n_points {
        let ends_fixed_bin = stop == n_points || fixed_bin(stop) != fixed_bin(stop - 1);
        if ends_fixed_bin && stop - edges[edges.len() - 1] >= parameters.min_ions_per_bin {
            edges.push(stop);
        }
    }

    if edges[edges.len() - 1] < n_points {
        if edges.len() > 1 {
            // extend the last bin over the trailing ions
            *edges.last_mut().unwrap() = n_points;
        } else {
            // no bin ever filled: all ions form a single bin
            edges.push(n_points);
        }
    }
    edges
}

/// Location of the deviations of one bin. `deviation` is scratch space and gets sorted.
fn location_of(deviation: &mut [f64], statistic: Statistic) -> f64 {
    deviation.sort_by(f64::total_cmp);
    match statistic {
        Statistic::Median => percentile_of_sorted(deviation, 50.0),
        Statistic::Mode => mode_of_sorted(deviation),
    }
}

/// Mode of sorted deviations, read from a smoothed histogram of their central percentile
/// range.
///
/// The histogram is smoothed with a moving sum over `MODE_SMOOTHING_BINS`, counting the bins
/// outside the histogram as empty. This pulls its ends down, so that the peak is not read off
/// the trimmed edges.
fn mode_of_sorted(deviation: &[f64]) -> f64 {
    let low = percentile_of_sorted(deviation, MODE_PERCENTILE_LOW);
    let high = percentile_of_sorted(deviation, MODE_PERCENTILE_HIGH);
    let n_bins = ((high - low) / MODE_BIN_WIDTH).ceil() as usize;
    // a distribution narrower than the smoothing window has no resolvable mode
    if n_bins < MODE_SMOOTHING_BINS {
        return percentile_of_sorted(deviation, 50.0);
    }

    let width = (high - low) / n_bins as f64;
    let mut counts = vec![0.0; n_bins];
    for value in deviation
        .iter()
        .filter(|&&value| (low..=high).contains(&value))
    {
        let bin = (((value - low) / width) as usize).min(n_bins - 1);
        counts[bin] += 1.0;
    }

    let smoothed = moving_sum(&counts, MODE_SMOOTHING_BINS);
    let mut peak = 0;
    for (bin, &count) in smoothed.iter().enumerate() {
        if count > smoothed[peak] {
            peak = bin;
        }
    }
    low + (peak as f64 + 0.5) * width
}

/// Moving sum over the `window` values centred on each value.
fn moving_sum(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            values[neighbourhood(index, window, values.len())]
                .iter()
                .sum()
        })
        .collect()
}

/// Moving average over the `window` values centred on each value.
///
/// The average divides by the neighbours inside the array rather than by the window width.
/// Unlike zero padding this keeps the ends of the curve unbiased, which matters at the start
/// and the end of the gradient where the drift is usually largest.
fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            let neighbours = &values[neighbourhood(index, window, values.len())];
            neighbours.iter().sum::<f64>() / neighbours.len() as f64
        })
        .collect()
}

/// The `window` values centred on `index`, clipped to the array. `window` must be odd.
fn neighbourhood(index: usize, window: usize, len: usize) -> Range<usize> {
    index.saturating_sub(window / 2)..(index + window / 2 + 1).min(len)
}

/// Linear interpolated percentile of sorted values, as `numpy.percentile` computes it.
fn percentile_of_sorted(sorted: &[f64], percentile: f64) -> f64 {
    let rank = percentile / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    sorted[lower] + (rank - lower as f64) * (sorted[upper] - sorted[lower])
}

/// Linear interpolation of `ys` at `x`, constant outside the range of the ascending `xs`.
fn interpolate(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let upper = xs.partition_point(|&value| value < x);
    if upper == 0 {
        return ys[0];
    }
    if upper == xs.len() {
        return ys[ys.len() - 1];
    }
    let span = xs[upper] - xs[upper - 1];
    if span <= 0.0 {
        return ys[upper];
    }
    ys[upper - 1] + (x - xs[upper - 1]) / span * (ys[upper] - ys[upper - 1])
}
