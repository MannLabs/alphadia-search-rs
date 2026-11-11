//! Correction of the retention time (RT) dependent intensity drift of every run. `DOCS.md`
//! defines the terms used here and gives the algorithm.
//!
//! The matrices hold `f32` like the fragment tables of the caller, so the correction needs no
//! `f64` copy of them. The statistics are computed in `f64`.

use std::ops::Range;

use rayon::prelude::*;

/// Mode histogram: bin width in log2 and the percentile range it spans, excluding outliers.
const MODE_BIN_WIDTH: f64 = 0.02;
const MODE_PERCENTILE_LOW: f64 = 2.0;
const MODE_PERCENTILE_HIGH: f64 = 98.0;
/// Moving-sum window of the mode histogram, in histogram bins. Deviations spanning fewer bins
/// fall back to the median.
const MODE_SMOOTHING_BINS: usize = 11;
/// Moving-average window of the curve, in RT bins.
const CURVE_SMOOTHING_BINS: usize = 3;

/// Location estimator of the deviations in one bin.
#[derive(Clone, Copy)]
pub enum Statistic {
    /// Mode of the smoothed histogram. Robust to a subset of ions with a true abundance
    /// change between the runs.
    Mode,
    /// Median.
    // kept as the reference estimator the mode is compared against in the tests
    #[allow(dead_code)]
    Median,
}

pub const DEFAULT_BIN_SECONDS: f64 = 30.0;
pub const DEFAULT_MIN_CORRELATION: f64 = 0.8;
pub const DEFAULT_MIN_OBSERVATIONS: usize = 3;
pub const DEFAULT_MIN_IONS_PER_BIN: usize = 100;
pub const DEFAULT_MIN_IONS_PER_RUN: usize = 500;

/// Parameters of the correction, see `DOCS.md` for the terms.
pub struct DriftParameters {
    /// Width of an RT bin before merging, in seconds.
    pub bin_seconds: f64,
    /// Minimum mean cross-run correlation of a usable ion.
    pub min_correlation: f64,
    /// Minimum number of runs a usable ion is observed in.
    pub min_observations: usize,
    /// Minimum number of usable ions observed in the run per bin.
    pub min_ions_per_bin: usize,
    /// Minimum number of usable ions observed in a run; runs below it are not corrected.
    pub min_ions_per_run: usize,
    pub statistic: Statistic,
}

impl DriftParameters {
    pub const DEFAULT: Self = Self {
        bin_seconds: DEFAULT_BIN_SECONDS,
        min_correlation: DEFAULT_MIN_CORRELATION,
        min_observations: DEFAULT_MIN_OBSERVATIONS,
        min_ions_per_bin: DEFAULT_MIN_IONS_PER_BIN,
        min_ions_per_run: DEFAULT_MIN_IONS_PER_RUN,
        statistic: Statistic::Mode,
    };

    pub fn validate(&self) -> Result<(), String> {
        if !(self.bin_seconds.is_finite() && self.bin_seconds > 0.0) {
            return Err(format!(
                "bin_seconds must be a positive number, got {}",
                self.bin_seconds
            ));
        }
        if !self.min_correlation.is_finite() {
            return Err(format!(
                "min_correlation must be a number, got {}",
                self.min_correlation
            ));
        }
        if self.min_observations == 0 {
            return Err("min_observations must be at least 1".to_string());
        }
        if self.min_ions_per_bin == 0 {
            return Err("min_ions_per_bin must be at least 1".to_string());
        }
        Ok(())
    }
}

/// Curve of one run.
pub struct DriftCurve {
    /// Mean RT of the usable ions in every bin, ascending.
    pub rt: Vec<f64>,
    /// Correction at `rt`, in log2.
    pub correction: Vec<f64>,
}

impl DriftCurve {
    /// Correction at `rt`: linear between bins, constant outside them, 0 for a non-finite `rt`.
    fn at(&self, rt: f64) -> f64 {
        if rt.is_finite() {
            interpolate(rt, &self.rt, &self.correction)
        } else {
            0.0
        }
    }
}

/// Checks that the `(rows, columns)` shapes of the matrices agree and that every precursor
/// index is a column of `rt`.
pub fn validate_shapes(
    intensity: (usize, usize),
    correlation: (usize, usize),
    rt: (usize, usize),
    precursor: &[usize],
) -> Result<(), String> {
    if correlation != intensity {
        return Err(format!(
            "intensity and correlation must have the same shape, got {intensity:?} and {correlation:?}"
        ));
    }
    let (n_runs, n_ions) = intensity;
    if rt.0 != n_runs || precursor.len() != n_ions {
        return Err(format!(
            "rt must have {n_runs} runs and precursor {n_ions} ions, got {rt:?} and {}",
            precursor.len()
        ));
    }
    if precursor.iter().any(|&index| index >= rt.1) {
        return Err(format!(
            "precursor indices must be smaller than the {} precursors of rt",
            rt.1
        ));
    }
    Ok(())
}

/// Divides every observed cell of `intensity` by `2^correction`, in place.
///
/// * `intensity` — run-major `n_runs x n_ions`, linear; 0 marks a missing cell.
/// * `correlation` — run-major `n_runs x n_ions`, cross-run fragment correlation.
/// * `rt` — run-major `n_runs x n_precursors`, observed precursor RT; NaN where the run has
///   no identification.
/// * `precursor` — precursor column of `rt` for every ion.
///
/// The shapes must pass `validate_shapes`. Returns the curve of every run, `None` for a run
/// that is not corrected.
pub fn correct_drift(
    intensity: &mut [f32],
    correlation: &[f32],
    rt: &[f32],
    precursor: &[usize],
    n_runs: usize,
    parameters: &DriftParameters,
) -> Vec<Option<DriftCurve>> {
    let n_ions = precursor.len();
    // the matrices are chunked by run, which needs at least one ion and one precursor
    if n_ions == 0 || rt.is_empty() {
        return (0..n_runs).map(|_| None).collect();
    }
    let n_precursors = rt.len() / n_runs;
    let reference = reference_levels(intensity, correlation, n_ions, parameters);
    let mean_rt = mean_retention_times(rt, n_precursors);

    let mut curves: Vec<Option<DriftCurve>> = intensity
        .par_chunks_exact(n_ions)
        .zip(rt.par_chunks_exact(n_precursors))
        .map(|(run_intensity, run_rt)| {
            let ion_rt = IonRt::new(run_rt, &mean_rt, precursor);
            estimate_curve(run_intensity, &ion_rt, &reference, parameters)
        })
        .collect();

    intensity
        .par_chunks_exact_mut(n_ions)
        .zip(rt.par_chunks_exact(n_precursors))
        .zip(curves.par_iter_mut())
        .for_each(|((run_intensity, run_rt), curve)| {
            if let Some(curve) = curve {
                let ion_rt = IonRt::new(run_rt, &mean_rt, precursor);
                apply_curve(run_intensity, &ion_rt, curve);
            }
        });

    curves
}

/// RT of every ion in one run: the RT of its precursor in the run, or the mean RT of the
/// precursor over all runs where the run has no identification.
struct IonRt<'a> {
    run_rt: &'a [f32],
    mean_rt: &'a [f64],
    precursor: &'a [usize],
}

impl<'a> IonRt<'a> {
    fn new(run_rt: &'a [f32], mean_rt: &'a [f64], precursor: &'a [usize]) -> Self {
        Self {
            run_rt,
            mean_rt,
            precursor,
        }
    }

    fn get(&self, ion: usize) -> f64 {
        let precursor = self.precursor[ion];
        let rt = f64::from(self.run_rt[precursor]);
        if rt.is_finite() {
            rt
        } else {
            self.mean_rt[precursor]
        }
    }
}

/// Reference level of every ion, NaN for an ion that is not usable.
///
/// The correlation threshold excludes fragments dominated by interference, whose deviation
/// does not reflect the drift of the run.
fn reference_levels(
    intensity: &[f32],
    correlation: &[f32],
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
                        levels.push(f64::from(intensity[cell]).log2());
                        // a missing correlation counts as no correlation
                        if correlation[cell].is_finite() {
                            correlation_sum += f64::from(correlation[cell]);
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

/// Mean RT of every precursor over the runs with an identification, NaN if there is none.
fn mean_retention_times(rt: &[f32], n_precursors: usize) -> Vec<f64> {
    let n_runs = rt.len() / n_precursors;
    (0..n_precursors)
        .into_par_iter()
        .map(|precursor| {
            let mut sum = 0.0;
            let mut n_runs_with_rt = 0;
            for run in 0..n_runs {
                let value = f64::from(rt[run * n_precursors + precursor]);
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

/// Curve of one run, `None` for fewer than `min_ions_per_run` usable ions observed in the run
/// or a single bin.
fn estimate_curve(
    intensity: &[f32],
    ion_rt: &IonRt,
    reference: &[f64],
    parameters: &DriftParameters,
) -> Option<DriftCurve> {
    let mut points = Vec::new();
    for ion in 0..intensity.len() {
        let rt = ion_rt.get(ion);
        if intensity[ion] > 0.0 && reference[ion].is_finite() && rt.is_finite() {
            points.push((rt, f64::from(intensity[ion]).log2() - reference[ion]));
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
        rt: bin_rt,
        correction: moving_average(&location, CURVE_SMOOTHING_BINS),
    })
}

/// Divides every observed cell of one run by `2^correction`, then centres the correction.
///
/// The mean correction over the observed cells is multiplied back in and subtracted from
/// `curve`, so the run keeps its mean log2 level and `curve` holds the applied correction.
fn apply_curve(intensity: &mut [f32], ion_rt: &IonRt, curve: &mut DriftCurve) {
    let mut correction_sum = 0.0;
    let mut n_corrected = 0;
    for ion in 0..intensity.len() {
        if intensity[ion] <= 0.0 {
            continue;
        }
        let correction = curve.at(ion_rt.get(ion));
        intensity[ion] = (f64::from(intensity[ion]) / correction.exp2()) as f32;
        correction_sum += correction;
        n_corrected += 1;
    }

    debug_assert!(n_corrected > 0, "a curve is estimated from observed ions");
    let centre = correction_sum / n_corrected as f64;
    let level = centre.exp2();
    intensity
        .iter_mut()
        .filter(|value| **value > 0.0)
        .for_each(|value| *value = (f64::from(*value) * level) as f32);
    curve
        .correction
        .iter_mut()
        .for_each(|correction| *correction -= centre);
}

/// Mean RT and location of the deviations of every bin.
fn bin_points(points: &[(f64, f64)], parameters: &DriftParameters) -> (Vec<f64>, Vec<f64>) {
    let edges = bin_edges(points, parameters);
    let mut bin_rt = Vec::with_capacity(edges.len() - 1);
    let mut location = Vec::with_capacity(edges.len() - 1);
    let mut deviation = Vec::new();
    for edge in edges.windows(2) {
        let bin = &points[edge[0]..edge[1]];
        bin_rt.push(bin.iter().map(|&(rt, _)| rt).sum::<f64>() / bin.len() as f64);
        deviation.clear();
        deviation.extend(bin.iter().map(|&(_, value)| value));
        location.push(location_of(&mut deviation, parameters.statistic));
    }
    (bin_rt, location)
}

/// Start index of every bin in the RT-sorted `points`, followed by the end of the last bin.
///
/// Points after the last full bin are added to it.
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
            *edges.last_mut().unwrap() = n_points;
        } else {
            // no bin ever filled: all ions form a single bin
            edges.push(n_points);
        }
    }
    edges
}

/// Location of the deviations of one bin. Sorts `deviation`.
fn location_of(deviation: &mut [f64], statistic: Statistic) -> f64 {
    deviation.sort_by(f64::total_cmp);
    match statistic {
        Statistic::Median => percentile_of_sorted(deviation, 50.0),
        Statistic::Mode => mode_of_sorted(deviation),
    }
}

/// Mode of sorted deviations: centre of the maximum of the moving-summed histogram.
///
/// The moving sum counts bins outside the histogram as empty, which lowers both ends so the
/// maximum is not taken at the percentile cut-offs.
fn mode_of_sorted(deviation: &[f64]) -> f64 {
    let low = percentile_of_sorted(deviation, MODE_PERCENTILE_LOW);
    let high = percentile_of_sorted(deviation, MODE_PERCENTILE_HIGH);
    let n_bins = ((high - low) / MODE_BIN_WIDTH).ceil() as usize;
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
/// Divides by the number of values inside the array rather than by `window`, which avoids
/// biasing the ends of the gradient towards 0, where the drift is usually largest.
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
