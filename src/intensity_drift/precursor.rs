//! Precursor-level correction of the intensity drift along the retention time (RT) and the
//! neutral precursor mass. `DOCS.md` defines the terms used here and gives the algorithm.
//!
//! The precursor table is small compared to the fragment tables, so the matrices hold `f64`.

use rayon::prelude::*;

use super::algorithm::{interpolate, percentile_of_sorted};

pub const DEFAULT_POINTS_PER_CELL: usize = 125;
pub const DEFAULT_N_STRATA: usize = 8;
pub const DEFAULT_MIN_RT_BINS_PER_STRATUM: usize = 10;
pub const DEFAULT_SIGMA_RT: f64 = 1.0;
pub const DEFAULT_SIGMA_MASS: f64 = 1.0;
pub const DEFAULT_MIN_POINTS_PER_CELL: usize = 30;
pub const DEFAULT_MIN_REFERENCE_PER_RUN: usize = 200;
pub const DEFAULT_MIN_RUN_FRACTION: f64 = 0.5;

/// A single run has no cross-run reference.
const MIN_RUNS: usize = 2;
/// A precursor quantified in a single run has no deviation from its row level.
const MIN_REFERENCE_RUNS: usize = 2;
/// Rounds of the alternating estimate of the row levels and the surfaces. Complete data is a
/// fixed point after the first round, and a few rounds settle the row levels of incomplete rows.
pub(super) const ROW_LEVEL_ROUNDS: usize = 3;
/// The rounds stop once no row level changes by more than this, in log2.
const ROW_LEVEL_TOLERANCE: f64 = 1e-6;
/// The half-sample mode averages the values once at most this many are left.
const HALF_SAMPLE_MODE_MIN_SIZE: usize = 3;
/// Radius of the Gaussian kernel in standard deviations, as in `scipy.ndimage.gaussian_filter`.
const GAUSSIAN_TRUNCATE: f64 = 4.0;

/// Parameters of the precursor-level correction, see `DOCS.md` for the terms.
pub struct PrecursorDriftParameters {
    /// Target number of reference precursors per cell; sets the number of RT bins of a run.
    pub points_per_cell: usize,
    /// Maximum number of mass strata.
    pub n_strata: usize,
    /// Fewer strata are used when the reference precursors cannot fill this many RT bins per
    /// stratum.
    pub min_rt_bins_per_stratum: usize,
    /// Standard deviation of the smoothing kernel along the RT bins, in bins; 0 disables it.
    pub sigma_rt: f64,
    /// Standard deviation of the smoothing kernel along the mass strata, in strata; 0 disables it.
    pub sigma_mass: f64,
    /// A cell with fewer reference precursors gets no location of its own.
    pub min_points_per_cell: usize,
    /// Runs with fewer reference precursors are not corrected.
    pub min_reference_per_run: usize,
    /// A reference precursor is quantified in at least this share of the runs, and in at least
    /// two.
    pub min_run_fraction: f64,
}

impl PrecursorDriftParameters {
    #[cfg(test)]
    pub const DEFAULT: Self = Self {
        points_per_cell: DEFAULT_POINTS_PER_CELL,
        n_strata: DEFAULT_N_STRATA,
        min_rt_bins_per_stratum: DEFAULT_MIN_RT_BINS_PER_STRATUM,
        sigma_rt: DEFAULT_SIGMA_RT,
        sigma_mass: DEFAULT_SIGMA_MASS,
        min_points_per_cell: DEFAULT_MIN_POINTS_PER_CELL,
        min_reference_per_run: DEFAULT_MIN_REFERENCE_PER_RUN,
        min_run_fraction: DEFAULT_MIN_RUN_FRACTION,
    };

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("points_per_cell", self.points_per_cell),
            ("n_strata", self.n_strata),
            ("min_rt_bins_per_stratum", self.min_rt_bins_per_stratum),
            ("min_points_per_cell", self.min_points_per_cell),
        ] {
            if value == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        for (name, value) in [("sigma_rt", self.sigma_rt), ("sigma_mass", self.sigma_mass)] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("{name} must be a non-negative number, got {value}"));
            }
        }
        if !(0.0..=1.0).contains(&self.min_run_fraction) {
            return Err(format!(
                "min_run_fraction must be between 0 and 1, got {}",
                self.min_run_fraction
            ));
        }
        Ok(())
    }

    /// Number of runs a reference precursor is quantified in, at least.
    fn min_reference_runs(&self, n_runs: usize) -> usize {
        ((self.min_run_fraction * n_runs as f64).ceil() as usize).max(MIN_REFERENCE_RUNS)
    }
}

/// Outcome of the correction.
pub struct PrecursorDriftSummary {
    /// Whether each run was corrected.
    pub corrected: Vec<bool>,
    /// Largest absolute correction of each run, in log2; 0 for a run that is not corrected.
    pub max_abs_correction: Vec<f64>,
    /// Number of mass strata used, 0 if there was no reference precursor.
    pub n_strata: usize,
}

impl PrecursorDriftSummary {
    fn uncorrected(n_runs: usize) -> Self {
        Self {
            corrected: vec![false; n_runs],
            max_abs_correction: vec![0.0; n_runs],
            n_strata: 0,
        }
    }
}

/// Checks that the `(rows, columns)` shapes of `intensity` and `rt` agree and that `mass` has
/// one value per precursor.
pub fn validate_precursor_shapes(
    intensity: (usize, usize),
    rt: (usize, usize),
    n_mass: usize,
) -> Result<(), String> {
    if rt != intensity || n_mass != intensity.1 {
        return Err(format!(
            "rt must have the shape {intensity:?} of intensity and mass {} values, got {rt:?} and {n_mass}",
            intensity.1
        ));
    }
    Ok(())
}

/// Divides every quantified cell of `intensity` by `2^correction`, in place.
///
/// * `intensity` — run-major `n_runs x n_precursors`, linear; 0 or NaN marks a missing cell.
/// * `rt` — run-major `n_runs x n_precursors`, observed RT; NaN where unknown.
/// * `mass` — neutral mass of every precursor.
///
/// The shapes must pass `validate_precursor_shapes`.
pub fn correct_precursor_drift(
    intensity: &mut [f64],
    rt: &[f64],
    mass: &[f64],
    n_runs: usize,
    parameters: &PrecursorDriftParameters,
) -> PrecursorDriftSummary {
    correct_precursor_drift_in_rounds(
        intensity,
        rt,
        mass,
        n_runs,
        parameters,
        ROW_LEVEL_ROUNDS,
        ROW_LEVEL_TOLERANCE,
    )
}

/// `correct_precursor_drift` with at most `rounds` rounds of the alternating estimate, stopping
/// once no row level changes by more than `tolerance`.
pub(super) fn correct_precursor_drift_in_rounds(
    intensity: &mut [f64],
    rt: &[f64],
    mass: &[f64],
    n_runs: usize,
    parameters: &PrecursorDriftParameters,
    rounds: usize,
    tolerance: f64,
) -> PrecursorDriftSummary {
    let n_precursors = mass.len();
    if n_runs < MIN_RUNS || n_precursors == 0 {
        return PrecursorDriftSummary::uncorrected(n_runs);
    }
    let rt = rt_with_mean_fallback(rt, n_precursors);
    let reference = Reference::new(
        intensity,
        mass,
        n_precursors,
        parameters.min_reference_runs(n_runs),
    );
    let n_reference = reference.precursors.len();
    if n_reference == 0 {
        return PrecursorDriftSummary::uncorrected(n_runs);
    }

    let n_strata = (n_reference
        / (parameters.min_rt_bins_per_stratum * parameters.points_per_cell))
        .clamp(1, parameters.n_strata);
    let reference_mass: Vec<f64> = reference.precursors.iter().map(|&p| mass[p]).collect();
    let strata = QuantileBins::new(&reference_mass, n_strata);
    let reference_stratum: Vec<usize> = reference_mass
        .iter()
        .map(|&value| strata.assign(value))
        .collect();
    let stratum_index: Vec<f64> = mass
        .iter()
        .map(|&value| strata.fractional_index(value))
        .collect();
    let reference_stratum_index: Vec<f64> = reference
        .precursors
        .iter()
        .map(|&precursor| stratum_index[precursor])
        .collect();
    let reference_rt: Vec<f64> = rt
        .chunks_exact(n_precursors)
        .flat_map(|run_rt| reference.precursors.iter().map(|&p| run_rt[p]))
        .collect();

    let mut offset = vec![0.0; n_runs * n_reference];
    let mut row_level: Vec<f64> = Vec::new();
    let mut surfaces: Vec<Option<Surface>> = Vec::new();
    for _ in 0..rounds {
        let next_row_level = reference.row_levels(&offset);
        if !row_level.is_empty() && max_abs_difference(&row_level, &next_row_level) <= tolerance {
            break;
        }
        row_level = next_row_level;
        let deviation = reference.deviations(&row_level);
        surfaces = reference_rt
            .par_chunks_exact(n_reference)
            .zip(deviation.par_chunks_exact(n_reference))
            .map(|(run_rt, run_deviation)| {
                Surface::estimate(
                    run_rt,
                    run_deviation,
                    &reference_stratum,
                    strata.centres.len(),
                    parameters,
                )
            })
            .collect();
        offset = zero_sum_offsets(&surfaces, &reference_rt, &reference_stratum_index);
    }

    let corrections: Vec<Option<Vec<f64>>> = intensity
        .par_chunks_exact(n_precursors)
        .zip(rt.par_chunks_exact(n_precursors))
        .zip(surfaces.par_iter())
        .map(|((run_intensity, run_rt), surface)| {
            surface
                .as_ref()
                .map(|surface| surface.corrections(run_intensity, run_rt, &stratum_index))
        })
        .collect();

    let mut summary = PrecursorDriftSummary {
        corrected: corrections.iter().map(Option::is_some).collect(),
        max_abs_correction: vec![0.0; n_runs],
        n_strata: strata.centres.len(),
    };
    intensity
        .par_chunks_exact_mut(n_precursors)
        .zip(corrections.par_iter())
        .zip(summary.max_abs_correction.par_iter_mut())
        .for_each(|((run_intensity, correction), max_abs)| {
            let Some(correction) = correction else {
                return;
            };
            for (value, &correction) in run_intensity.iter_mut().zip(correction) {
                if correction.is_finite() {
                    *value /= correction.exp2();
                    *max_abs = max_abs.max(correction.abs());
                }
            }
        });
    summary
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Run-major offsets of the reference precursors: the surface of every run with a surface at
/// the precursor's RT and stratum, minus the precursor's mean over those runs. The zero sum keeps
/// the offsets from moving the row levels of complete rows, which would otherwise drift together
/// with the surfaces from round to round. 0 in a run without a surface or a position.
fn zero_sum_offsets(
    surfaces: &[Option<Surface>],
    reference_rt: &[f64],
    reference_stratum_index: &[f64],
) -> Vec<f64> {
    let n_reference = reference_stratum_index.len();
    let mut offset: Vec<f64> = surfaces
        .par_iter()
        .zip(reference_rt.par_chunks_exact(n_reference))
        .flat_map_iter(|(surface, run_rt)| {
            run_rt
                .iter()
                .zip(reference_stratum_index)
                .map(move |(&rt, &stratum)| match surface {
                    Some(surface) => surface.at_position(rt, stratum),
                    None => f64::NAN,
                })
        })
        .collect();
    let n_runs = surfaces.len();
    for precursor in 0..n_reference {
        let cells = || (0..n_runs).map(|run| run * n_reference + precursor);
        let (sum, count) = cells()
            .map(|cell| offset[cell])
            .filter(|value| value.is_finite())
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
        let mean = if count == 0 { 0.0 } else { sum / count as f64 };
        for cell in cells() {
            offset[cell] = if offset[cell].is_finite() {
                offset[cell] - mean
            } else {
                0.0
            };
        }
    }
    offset
}

/// `rt` with every NaN replaced by the mean RT of its precursor over the runs where it is known.
fn rt_with_mean_fallback(rt: &[f64], n_precursors: usize) -> Vec<f64> {
    let n_runs = rt.len() / n_precursors;
    let mean_rt: Vec<f64> = (0..n_precursors)
        .map(|precursor| {
            let (sum, count) = (0..n_runs)
                .map(|run| rt[run * n_precursors + precursor])
                .filter(|value| value.is_finite())
                .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect();
    rt.iter()
        .enumerate()
        .map(|(cell, &value)| {
            if value.is_finite() {
                value
            } else {
                mean_rt[cell % n_precursors]
            }
        })
        .collect()
}

fn is_quantified(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Precursors with a finite mass quantified in at least the minimum number of runs, and their
/// log2 quantities.
struct Reference {
    precursors: Vec<usize>,
    /// Run-major `n_runs x precursors.len()`, NaN where the precursor is not quantified.
    log2: Vec<f64>,
}

impl Reference {
    fn new(intensity: &[f64], mass: &[f64], n_precursors: usize, min_runs: usize) -> Self {
        let n_runs = intensity.len() / n_precursors;
        let precursors: Vec<usize> = (0..n_precursors)
            .filter(|&precursor| {
                mass[precursor].is_finite()
                    && (0..n_runs)
                        .filter(|&run| is_quantified(intensity[run * n_precursors + precursor]))
                        .count()
                        >= min_runs
            })
            .collect();
        let log2 = intensity
            .chunks_exact(n_precursors)
            .flat_map(|run_intensity| {
                precursors.iter().map(|&precursor| {
                    let value = run_intensity[precursor];
                    if is_quantified(value) {
                        value.log2()
                    } else {
                        f64::NAN
                    }
                })
            })
            .collect();
        Self { precursors, log2 }
    }

    /// Row level of every precursor: the mean over its quantified runs of the log2 quantity
    /// minus the run's offset. The offsets make the level of an incomplete row independent of
    /// which runs it is missing in.
    fn row_levels(&self, offset: &[f64]) -> Vec<f64> {
        // the mean, not the median: with a median the median run of every precursor sits
        // exactly on its reference and the spike of zero deviations captures the mode
        let n_reference = self.precursors.len();
        let n_runs = self.log2.len() / n_reference;
        (0..n_reference)
            .map(|precursor| {
                let (sum, count) = (0..n_runs)
                    .map(|run| run * n_reference + precursor)
                    .filter(|&cell| self.log2[cell].is_finite())
                    .fold((0.0, 0), |(sum, count), cell| {
                        (sum + self.log2[cell] - offset[cell], count + 1)
                    });
                sum / count as f64
            })
            .collect()
    }

    /// Run-major log2 quantity minus the row level, NaN where the precursor is not quantified.
    fn deviations(&self, row_level: &[f64]) -> Vec<f64> {
        let n_reference = self.precursors.len();
        self.log2
            .iter()
            .enumerate()
            .map(|(cell, &value)| value - row_level[cell % n_reference])
            .collect()
    }
}

/// Bins at quantiles of a set of values: a value `v` falls into the last bin whose lower edge is
/// at most `v`; values below the first edge fall into the first bin.
struct QuantileBins {
    /// Lower edge of every bin, followed by infinity.
    edges: Vec<f64>,
    /// Median of the values in every bin, ascending.
    centres: Vec<f64>,
    /// Index of every bin as a number, the positions `fractional_index` interpolates.
    positions: Vec<f64>,
}

impl QuantileBins {
    /// `n_bins` bins at quantiles of `values`; fewer when ties collapse edges, without empty bins.
    fn new(values: &[f64], n_bins: usize) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut edges: Vec<f64> = (0..=n_bins)
            .map(|bin| percentile_of_sorted(&sorted, 100.0 * bin as f64 / n_bins as f64))
            .collect();
        edges.dedup();
        if edges.len() < 2 {
            edges = vec![f64::NEG_INFINITY, f64::INFINITY];
        } else {
            edges[0] = f64::NEG_INFINITY;
            *edges.last_mut().unwrap() = f64::INFINITY;
        }

        let all_bins = Self {
            edges,
            centres: Vec::new(),
            positions: Vec::new(),
        };
        let bin_of: Vec<usize> = sorted.iter().map(|&value| all_bins.assign(value)).collect();
        let mut edges: Vec<f64> = Vec::new();
        let mut centres = Vec::new();
        let mut start = 0;
        while start < sorted.len() {
            let stop = start + bin_of[start..].partition_point(|&bin| bin == bin_of[start]);
            edges.push(all_bins.edges[bin_of[start]]);
            centres.push(percentile_of_sorted(&sorted[start..stop], 50.0));
            start = stop;
        }
        edges.push(f64::INFINITY);
        let positions = (0..centres.len()).map(|bin| bin as f64).collect();
        Self {
            edges,
            centres,
            positions,
        }
    }

    fn assign(&self, value: f64) -> usize {
        let n_bins = self.edges.len() - 1;
        self.edges
            .partition_point(|&edge| edge <= value)
            .saturating_sub(1)
            .min(n_bins - 1)
    }

    /// Position of `value` between the bin centres in bin units, clamped to the outer centres;
    /// NaN for a non-finite `value`.
    fn fractional_index(&self, value: f64) -> f64 {
        if !value.is_finite() {
            return f64::NAN;
        }
        if self.centres.len() == 1 {
            return 0.0;
        }
        interpolate(value, &self.centres, &self.positions)
    }
}

/// Smoothed location of the deviations of one run on the RT bin x mass stratum grid.
struct Surface {
    rt_bins: QuantileBins,
    n_strata: usize,
    /// Row-major `rt_bins.centres.len() x n_strata`.
    grid: Vec<f64>,
}

impl Surface {
    /// `None` for fewer than `min_reference_per_run` reference precursors quantified in the run
    /// with an RT, or no cell with `min_points_per_cell` of them. `run_rt` and `run_deviation`
    /// hold the reference precursors of the run.
    fn estimate(
        run_rt: &[f64],
        run_deviation: &[f64],
        reference_stratum: &[usize],
        n_strata: usize,
        parameters: &PrecursorDriftParameters,
    ) -> Option<Self> {
        let points: Vec<(f64, usize, f64)> = run_rt
            .iter()
            .zip(reference_stratum)
            .zip(run_deviation)
            .map(|((&rt, &stratum), &deviation)| (rt, stratum, deviation))
            .filter(|&(rt, _, deviation)| rt.is_finite() && deviation.is_finite())
            .collect();
        if points.len() < parameters.min_reference_per_run {
            return None;
        }

        let n_rt_bins = (points.len() as f64 / (n_strata * parameters.points_per_cell) as f64)
            .round_ties_even()
            .max(1.0) as usize;
        let point_rt: Vec<f64> = points.iter().map(|&(rt, _, _)| rt).collect();
        let rt_bins = QuantileBins::new(&point_rt, n_rt_bins);
        let n_cells = rt_bins.centres.len() * n_strata;

        let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_cells];
        for &(rt, stratum, deviation) in &points {
            members[rt_bins.assign(rt) * n_strata + stratum].push(deviation);
        }
        let mut location = vec![0.0; n_cells];
        let mut weight = vec![0.0; n_cells];
        for (cell, values) in members.iter_mut().enumerate() {
            if values.len() >= parameters.min_points_per_cell {
                location[cell] = half_sample_mode(values);
                weight[cell] = values.len() as f64;
            }
        }
        if weight.iter().all(|&weight| weight == 0.0) {
            return None;
        }

        let shape = (rt_bins.centres.len(), n_strata);
        let grid = smooth(&location, &weight, shape, parameters);
        Some(Self {
            rt_bins,
            n_strata,
            grid,
        })
    }

    /// The surface at an RT and a fractional stratum index, NaN if either is not finite.
    fn at_position(&self, rt: f64, stratum_index: f64) -> f64 {
        if rt.is_finite() && stratum_index.is_finite() {
            self.at(self.rt_bins.fractional_index(rt), stratum_index)
        } else {
            f64::NAN
        }
    }

    /// Centred correction of every quantified cell of the run with an RT and a mass, NaN for the
    /// other cells. The centring keeps the mean log2 level of the run.
    fn corrections(
        &self,
        run_intensity: &[f64],
        run_rt: &[f64],
        stratum_index: &[f64],
    ) -> Vec<f64> {
        let mut correction: Vec<f64> = run_intensity
            .iter()
            .zip(run_rt)
            .zip(stratum_index)
            .map(|((&value, &rt), &stratum)| {
                if is_quantified(value) {
                    self.at_position(rt, stratum)
                } else {
                    f64::NAN
                }
            })
            .collect();
        let (sum, count) = correction
            .iter()
            .filter(|value| value.is_finite())
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
        if count > 0 {
            let centre = sum / count as f64;
            correction.iter_mut().for_each(|value| *value -= centre);
        }
        correction
    }

    /// Bilinear interpolation of the grid at a fractional (RT bin, stratum) position.
    fn at(&self, rt_index: f64, stratum_index: f64) -> f64 {
        let n_rt_bins = self.rt_bins.centres.len();
        let corners = |index: f64, n: usize| {
            let lower = index.floor() as usize;
            (lower, (lower + 1).min(n - 1), index - lower as f64)
        };
        let (r0, r1, tr) = corners(rt_index, n_rt_bins);
        let (s0, s1, ts) = corners(stratum_index, self.n_strata);
        let value = |r: usize, s: usize| self.grid[r * self.n_strata + s];
        (1.0 - tr) * ((1.0 - ts) * value(r0, s0) + ts * value(r0, s1))
            + tr * ((1.0 - ts) * value(r1, s0) + ts * value(r1, s1))
    }
}

/// Bickel's half-sample mode: repeatedly keeps the half of the values in the narrowest interval,
/// then averages the last few. Sorts `values`, which must not be empty.
fn half_sample_mode(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mut remaining: &[f64] = values;
    while remaining.len() > HALF_SAMPLE_MODE_MIN_SIZE {
        let half = remaining.len().div_ceil(2);
        let mut start = 0;
        let mut narrowest = f64::INFINITY;
        for (offset, window) in remaining.windows(half).enumerate() {
            let width = window[half - 1] - window[0];
            if width < narrowest {
                narrowest = width;
                start = offset;
            }
        }
        remaining = &remaining[start..start + half];
    }
    remaining.iter().sum::<f64>() / remaining.len() as f64
}

/// Count-weighted Gaussian smoothing (Nadaraya-Watson in cell units) of the cell locations of a
/// `(rows, columns)` grid. Cells without a smoothed value are filled by `fill_missing`.
fn smooth(
    location: &[f64],
    weight: &[f64],
    shape: (usize, usize),
    parameters: &PrecursorDriftParameters,
) -> Vec<f64> {
    let weighted: Vec<f64> = location.iter().zip(weight).map(|(l, w)| l * w).collect();
    let sigma = (parameters.sigma_rt, parameters.sigma_mass);
    let numerator = gaussian_filter(&weighted, shape, sigma);
    let denominator = gaussian_filter(weight, shape, sigma);
    let smoothed: Vec<f64> = numerator
        .iter()
        .zip(&denominator)
        .map(|(&n, &d)| if d > 0.0 { n / d } else { f64::NAN })
        .collect();
    fill_missing(&smoothed, shape)
}

/// Gaussian filter along both axes of a `(rows, columns)` grid; the grid continues beyond its
/// edges with the edge values.
fn gaussian_filter(values: &[f64], shape: (usize, usize), sigma: (f64, f64)) -> Vec<f64> {
    let (n_rows, n_columns) = shape;
    let mut out = values.to_vec();
    if sigma.0 > 0.0 {
        let kernel = gaussian_kernel(sigma.0);
        let radius = (kernel.len() / 2) as isize;
        let input = out.clone();
        for row in 0..n_rows {
            for column in 0..n_columns {
                out[row * n_columns + column] = kernel
                    .iter()
                    .enumerate()
                    .map(|(k, w)| {
                        let source = (row as isize + k as isize - radius)
                            .clamp(0, n_rows as isize - 1)
                            as usize;
                        w * input[source * n_columns + column]
                    })
                    .sum();
            }
        }
    }
    if sigma.1 > 0.0 {
        let kernel = gaussian_kernel(sigma.1);
        let radius = (kernel.len() / 2) as isize;
        let input = out.clone();
        for row in 0..n_rows {
            for column in 0..n_columns {
                out[row * n_columns + column] = kernel
                    .iter()
                    .enumerate()
                    .map(|(k, w)| {
                        let source = (column as isize + k as isize - radius)
                            .clamp(0, n_columns as isize - 1)
                            as usize;
                        w * input[row * n_columns + source]
                    })
                    .sum();
            }
        }
    }
    out
}

/// Normalised Gaussian weights over `GAUSSIAN_TRUNCATE` standard deviations on either side.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (GAUSSIAN_TRUNCATE * sigma + 0.5) as isize;
    let weights: Vec<f64> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Fills the NaN cells of a `(rows, columns)` grid: along the rows by linear interpolation
/// between the finite cells of the same column (constant beyond them), then columns without a
/// finite cell along the columns of every row.
fn fill_missing(grid: &[f64], shape: (usize, usize)) -> Vec<f64> {
    let (n_rows, n_columns) = shape;
    let mut out = grid.to_vec();
    let mut filled_columns = Vec::new();
    for column in 0..n_columns {
        let (rows, values): (Vec<f64>, Vec<f64>) = (0..n_rows)
            .filter(|&row| grid[row * n_columns + column].is_finite())
            .map(|row| (row as f64, grid[row * n_columns + column]))
            .unzip();
        if rows.is_empty() {
            continue;
        }
        for row in 0..n_rows {
            out[row * n_columns + column] = interpolate(row as f64, &rows, &values);
        }
        filled_columns.push(column);
    }
    if filled_columns.len() < n_columns && !filled_columns.is_empty() {
        let positions: Vec<f64> = filled_columns.iter().map(|&column| column as f64).collect();
        for row in 0..n_rows {
            let values: Vec<f64> = filled_columns
                .iter()
                .map(|&column| out[row * n_columns + column])
                .collect();
            for column in 0..n_columns {
                if !filled_columns.contains(&column) {
                    out[row * n_columns + column] = interpolate(column as f64, &positions, &values);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_sample_mode_finds_the_densest_half() {
        let mut values = vec![0.0, 0.01, 0.02, 0.03, 0.04, 5.0, 6.0, 7.0];
        assert!(half_sample_mode(&mut values) < 0.05);
    }

    #[test]
    fn test_quantile_bins_collapse_tied_edges_and_keep_every_bin_occupied() {
        let values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0];

        let bins = QuantileBins::new(&values, 4);

        assert_eq!(bins.centres, vec![1.0, 2.5]);
        assert_eq!(bins.assign(1.0), 0);
        assert_eq!(bins.assign(3.0), 1);
        assert_eq!(bins.assign(-10.0), 0);
    }

    #[test]
    fn test_reference_runs_are_a_share_of_the_runs_and_at_least_two() {
        let parameters = PrecursorDriftParameters::DEFAULT;

        let min_runs: Vec<usize> = [2, 3, 5, 6, 100]
            .iter()
            .map(|&n_runs| parameters.min_reference_runs(n_runs))
            .collect();

        assert_eq!(min_runs, vec![2, 2, 3, 3, 50]);
    }

    #[test]
    fn test_rejects_a_run_fraction_outside_zero_to_one() {
        for min_run_fraction in [-0.1, 1.5, f64::NAN] {
            let parameters = PrecursorDriftParameters {
                min_run_fraction,
                ..PrecursorDriftParameters::DEFAULT
            };

            assert!(parameters.validate().is_err());
        }
    }

    #[test]
    fn test_fill_missing_interpolates_along_rows_then_columns() {
        let nan = f64::NAN;
        let grid = [0.0, nan, nan, nan, 2.0, nan];

        let filled = fill_missing(&grid, (3, 2));

        assert_eq!(filled, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    }
}
