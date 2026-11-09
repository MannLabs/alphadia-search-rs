//! Abundance-aware fragment detection: a marked likelihood ratio of the apex scan for faint, one-scan peaks.
//!
//! `det_llr` (detection.rs) scores which fragment channels are hit, with the signal scale free, so a chance match whose
//! few hits sit on weak library fragments can pick a small scale that makes its missing strong fragments cheap. Here
//! the observed intensities pin the abundance: with a = log2 intensity of the strongest library fragment and
//! mu_i = a + log2(lib_i / lib_max),
//! - peptide present (H1): fragment i carries signal y ~ N(mu_i, scatter^2) in log2, seen with probability
//!   Phi((y - limit) / width), so it is seen at all with Phi((mu_i - limit) / sqrt(scatter^2 + width^2)); a channel
//!   can carry a background hit as well
//! - background only (H0): channel i is hit with its flank rate b_i, at an intensity drawn from the run's
//!   background-hit density f_B (log2 intensities of flank hits)
//!
//! A hit adds ln(1 + (1 - b_i) g_i(y_i) / (b_i f_B(y_i))), with g_i the signal density times the detection
//! probability; a miss adds ln P(not seen | mu_i). The sum is maximised over a. A real faint peptide misses exactly the
//! fragments its abundance predicts below the limit, and its hits sit at intensities that scale with the library; a
//! chance match hits channels at background intensities, irrespective of the library.
//!
//! The detection curve (limit, width, scatter) is fitted per run by maximum likelihood on the candidates with the
//! highest selection scores, targets and decoys alike, so it is the same for every candidate.

use numpy::ndarray::Array2;
use rayon::prelude::*;

/// Lower end of the log2-intensity histogram of background hits.
const DENSITY_MIN_LOG2: f64 = 0.0;
/// Bin width of that histogram in log2 units.
const DENSITY_BIN_LOG2: f64 = 0.1;
const DENSITY_NUM_BINS: usize = 320;
/// Gaussian smoothing of the histogram, in bins, and the kernel half-width in bins.
const DENSITY_SMOOTH_BINS: f64 = 2.0;
const DENSITY_KERNEL_HALF: i64 = 6;
/// Density floor per log2 unit: a hit far outside the background range must not give an infinite ratio.
const DENSITY_FLOOR: f64 = 1e-4;
/// Bounds that keep the log-likelihoods finite.
const MIN_PROBABILITY: f64 = 1e-6;
/// Profile grid over the abundance a: step, extent below the detection limit and above the largest abundance any
/// hit implies.
const ABUNDANCE_STEP: f64 = 0.25;
const ABUNDANCE_BELOW_LIMIT: f64 = 4.0;
const ABUNDANCE_ABOVE_HITS: f64 = 2.0;
/// Fine grid steps per coarse step of the profile search.
const COARSE_STRIDE: usize = 4;
/// The detection curve used when too few hits are available to fit it; the limit is then set from the background
/// hits, which sit about one log2 unit above the limit at the low end of the run's intensities.
const DEFAULT_WIDTH: f64 = 1.5;
const DEFAULT_SCATTER: f64 = 1.0;
const DEFAULT_LIMIT_BELOW_BACKGROUND_MEDIAN: f64 = 1.0;
/// Candidates with the highest selection scores that the detection curve is fitted on, and the hits it needs.
pub const FIT_CANDIDATES: usize = 5_000;
const MIN_FIT_HITS: usize = 500;
/// Coordinate-wise grid search of the curve: limit around the background median, width and scatter.
const LIMIT_SPAN: f64 = 3.0;
const LIMIT_STEP: f64 = 0.25;
const WIDTH_GRID: [f64; 6] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const SCATTER_GRID: [f64; 6] = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0];
const FIT_ROUNDS: usize = 3;

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// ln erfc(x), from the Chebyshev fit of Numerical Recipes (relative error < 1.2e-7), in log form so that it does
/// not underflow for large x.
fn ln_erfc(x: f64) -> f64 {
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
    let ln_pos = t.ln() + poly;
    if x >= 0.0 {
        ln_pos
    } else {
        (2.0 - ln_pos.exp()).ln()
    }
}

/// ln Phi(x) of the standard normal.
fn ln_ndtr(x: f64) -> f64 {
    ln_erfc(-x / std::f64::consts::SQRT_2) - std::f64::consts::LN_2
}

/// The per-run detection curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionCurve {
    /// log2 intensity at which a signal is seen with probability 1/2
    pub limit: f64,
    /// width of that threshold in log2 units
    pub width: f64,
    /// scatter of a fragment's log2 signal around its library prediction
    pub scatter: f64,
}

/// Density of the run's background hits over log2 intensity.
#[derive(Debug, Clone, PartialEq)]
pub struct BackgroundDensity {
    density: Vec<f64>,
    median_log2: f64,
}

impl BackgroundDensity {
    /// Smoothed histogram of the log2 intensities of background hits.
    pub fn from_log2(values: &[f64]) -> Self {
        let mut counts = vec![0.0f64; DENSITY_NUM_BINS];
        for &v in values {
            counts[Self::bin(v)] += 1.0;
        }
        let total = (values.len() as f64).max(1.0);
        let kernel: Vec<f64> = (-DENSITY_KERNEL_HALF..=DENSITY_KERNEL_HALF)
            .map(|d| (-0.5 * (d as f64 / DENSITY_SMOOTH_BINS).powi(2)).exp())
            .collect();
        let kernel_sum: f64 = kernel.iter().sum();
        let density = (0..DENSITY_NUM_BINS as i64)
            .map(|i| {
                let smoothed: f64 = kernel
                    .iter()
                    .enumerate()
                    .map(|(k, w)| {
                        let j = i + k as i64 - DENSITY_KERNEL_HALF;
                        if (0..DENSITY_NUM_BINS as i64).contains(&j) {
                            w * counts[j as usize]
                        } else {
                            0.0
                        }
                    })
                    .sum();
                (smoothed / kernel_sum / total / DENSITY_BIN_LOG2).max(DENSITY_FLOOR)
            })
            .collect();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median_log2 = if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        };
        Self {
            density,
            median_log2,
        }
    }

    fn bin(v: f64) -> usize {
        (((v - DENSITY_MIN_LOG2) / DENSITY_BIN_LOG2).floor().max(0.0) as usize)
            .min(DENSITY_NUM_BINS - 1)
    }

    pub fn at(&self, log2_intensity: f64) -> f64 {
        self.density[Self::bin(log2_intensity)]
    }

    /// The curve used without a fit: the limit one log2 unit below the median background hit.
    pub fn default_curve(&self) -> DetectionCurve {
        DetectionCurve {
            limit: self.median_log2 - DEFAULT_LIMIT_BELOW_BACKGROUND_MEDIAN,
            width: DEFAULT_WIDTH,
            scatter: DEFAULT_SCATTER,
        }
    }
}

/// The run-wide inputs of the faint-detection features.
#[derive(Debug, Clone, PartialEq)]
pub struct FaintModel {
    pub curve: DetectionCurve,
    pub background: BackgroundDensity,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FaintFeatures {
    /// max over the abundance of the marked hit/miss log-likelihood ratio at the apex scan
    pub det_faint_llr: f32,
    /// the hit terms of `det_faint_llr` at its best abundance
    pub det_faint_hits: f32,
    /// the miss terms of `det_faint_llr` at its best abundance (<= 0)
    pub det_faint_miss: f32,
}

impl FaintFeatures {
    pub fn missing() -> Self {
        Self::default()
    }
}

/// One fragment of a candidate at the apex scan.
#[derive(Debug, Clone, Copy)]
pub struct ApexFragment {
    /// log2(lib_i / lib_max)
    pub log2_relative: f64,
    /// log2 intensity at the apex scan, None if not hit
    pub log2_intensity: Option<f64>,
    /// the channel's background hit rate
    pub background_rate: f64,
}

/// The apex fragments of a detection XIC whose apex is column `apex_col`; None when the inputs do not fit.
pub fn apex_fragments(
    xic: &Array2<f32>,
    apex_col: usize,
    library_intensity: &[f32],
    background_rates: &[f32],
) -> Option<Vec<ApexFragment>> {
    let n = xic.nrows();
    if n == 0
        || apex_col >= xic.ncols()
        || library_intensity.len() != n
        || background_rates.len() != n
    {
        return None;
    }
    let top = library_intensity.iter().cloned().fold(0.0f32, f32::max);
    if top <= 0.0 {
        return None;
    }
    Some(
        (0..n)
            .filter(|&f| library_intensity[f] > 0.0)
            .map(|f| {
                let v = xic[[f, apex_col]];
                ApexFragment {
                    log2_relative: (library_intensity[f] as f64 / top as f64).log2(),
                    log2_intensity: (v > 0.0).then(|| (v as f64).log2()),
                    background_rate: (background_rates[f] as f64)
                        .clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY),
                }
            })
            .collect(),
    )
}

fn abundance_grid(fragments: &[ApexFragment], curve: &DetectionCurve) -> Vec<f64> {
    let low = curve.limit - ABUNDANCE_BELOW_LIMIT;
    let high = fragments
        .iter()
        .filter_map(|f| f.log2_intensity.map(|y| y - f.log2_relative))
        .fold(curve.limit + ABUNDANCE_BELOW_LIMIT, f64::max)
        + ABUNDANCE_ABOVE_HITS;
    let n = ((high - low) / ABUNDANCE_STEP).ceil() as usize + 1;
    (0..n).map(|i| low + i as f64 * ABUNDANCE_STEP).collect()
}

/// The maximum over the grid of `terms(a).0`, with the terms at that point: a coarse pass, then the fine grid within
/// one coarse step of the coarse maximum (the profile is smooth on the scale of a fragment's scatter).
fn profile_max<F: Fn(f64) -> (f64, f64, f64)>(grid: &[f64], terms: F) -> (f64, f64, f64) {
    let best_of = |indices: &mut dyn Iterator<Item = usize>| {
        indices
            .map(|i| (terms(grid[i]), i))
            .fold(((f64::NEG_INFINITY, 0.0, 0.0), 0), |acc, x| {
                if x.0 .0 > acc.0 .0 {
                    x
                } else {
                    acc
                }
            })
    };
    let (_, coarse) = best_of(&mut (0..grid.len()).step_by(COARSE_STRIDE));
    let lo = coarse.saturating_sub(COARSE_STRIDE - 1);
    let hi = (coarse + COARSE_STRIDE).min(grid.len());
    best_of(&mut (lo..hi)).0
}

/// ln of the signal density of a hit at log2 intensity y times its detection probability.
fn ln_signal_hit(y: f64, mu: f64, curve: &DetectionCurve) -> f64 {
    let z = (y - mu) / curve.scatter;
    -0.5 * z * z - curve.scatter.ln() - LN_SQRT_2PI + ln_ndtr((y - curve.limit) / curve.width)
}

/// ln P(signal not seen | mu).
fn ln_signal_miss(mu: f64, curve: &DetectionCurve) -> f64 {
    let total = curve.scatter.hypot(curve.width);
    ln_ndtr(-(mu - curve.limit) / total)
}

/// Faint-detection features of one candidate.
pub fn faint_features(fragments: &[ApexFragment], model: &FaintModel) -> FaintFeatures {
    if fragments.is_empty() {
        return FaintFeatures::missing();
    }
    let curve = &model.curve;
    // the parts of a hit term that do not depend on the abundance: ln((1 - b) / (b f_B(y))) + ln P(seen | y) - ln norm
    let hit_offset: Vec<f64> = fragments
        .iter()
        .map(|f| {
            f.log2_intensity.map_or(0.0, |y| {
                let b = f.background_rate;
                ((1.0 - b) / (b * model.background.at(y))).ln()
                    + ln_ndtr((y - curve.limit) / curve.width)
                    - curve.scatter.ln()
                    - LN_SQRT_2PI
            })
        })
        .collect();
    let terms = |a: f64| -> (f64, f64, f64) {
        let mut hits = 0.0f64;
        let mut misses = 0.0f64;
        for (f, &offset) in fragments.iter().zip(&hit_offset) {
            let mu = a + f.log2_relative;
            match f.log2_intensity {
                Some(y) => {
                    let z = (y - mu) / curve.scatter;
                    hits += (offset - 0.5 * z * z).exp().ln_1p();
                }
                None => misses += ln_signal_miss(mu, curve).max(MIN_PROBABILITY.ln()),
            }
        }
        (hits + misses, hits, misses)
    };
    let best = profile_max(&abundance_grid(fragments, curve), terms);
    FaintFeatures {
        det_faint_llr: best.0 as f32,
        det_faint_hits: best.1 as f32,
        det_faint_miss: best.2 as f32,
    }
}

/// Signal-only profile log-likelihood of a set of candidates under a curve (no background term).
fn profile_loglik(candidates: &[Vec<ApexFragment>], curve: &DetectionCurve) -> f64 {
    candidates
        .par_iter()
        .map(|fragments| {
            let terms = |a: f64| {
                let ll: f64 = fragments
                    .iter()
                    .map(|f| {
                        let mu = a + f.log2_relative;
                        match f.log2_intensity {
                            Some(y) => ln_signal_hit(y, mu, curve),
                            None => ln_signal_miss(mu, curve),
                        }
                    })
                    .sum();
                (ll, 0.0, 0.0)
            };
            profile_max(&abundance_grid(fragments, curve), terms).0
        })
        .sum()
}

/// The detection curve fitted on the apex fragments of high-scoring candidates, by a coordinate-wise grid search of
/// the signal-only profile likelihood; the default curve when they hold too few hits.
pub fn fit_curve(
    candidates: &[Vec<ApexFragment>],
    background: &BackgroundDensity,
) -> DetectionCurve {
    let default = background.default_curve();
    let n_hits: usize = candidates
        .iter()
        .map(|c| c.iter().filter(|f| f.log2_intensity.is_some()).count())
        .sum();
    if n_hits < MIN_FIT_HITS {
        return default;
    }
    let n_limits = (2.0 * LIMIT_SPAN / LIMIT_STEP).round() as usize + 1;
    let limits: Vec<f64> = (0..n_limits)
        .map(|i| default.limit - LIMIT_SPAN + i as f64 * LIMIT_STEP)
        .collect();
    let mut curve = default;
    let best_of = |candidates_: &[DetectionCurve]| -> DetectionCurve {
        candidates_
            .iter()
            .map(|c| (profile_loglik(candidates, c), *c))
            .fold((f64::NEG_INFINITY, default), |acc, x| {
                if x.0 > acc.0 {
                    x
                } else {
                    acc
                }
            })
            .1
    };
    for _ in 0..FIT_ROUNDS {
        curve = best_of(
            &limits
                .iter()
                .map(|&limit| DetectionCurve { limit, ..curve })
                .collect::<Vec<_>>(),
        );
        curve = best_of(
            &WIDTH_GRID
                .iter()
                .map(|&width| DetectionCurve { width, ..curve })
                .collect::<Vec<_>>(),
        );
        curve = best_of(
            &SCATTER_GRID
                .iter()
                .map(|&scatter| DetectionCurve { scatter, ..curve })
                .collect::<Vec<_>>(),
        );
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURVE: DetectionCurve = DetectionCurve {
        limit: 8.0,
        width: 1.5,
        scatter: 1.0,
    };

    fn model() -> FaintModel {
        // background hits around 2^9 +- 1.3
        let values: Vec<f64> = (0..2000)
            .map(|i| 9.0 + 1.3 * ((i as f64 / 2000.0) * 6.0 - 3.0))
            .collect();
        FaintModel {
            curve: CURVE,
            background: BackgroundDensity::from_log2(&values),
        }
    }

    fn library() -> Vec<f64> {
        // library intensities 1 .. 0.1 over 12 fragments
        (0..12).map(|f| (1.0 - f as f64 * 0.08).log2()).collect()
    }

    fn fragments(hit: &[(usize, f64)]) -> Vec<ApexFragment> {
        library()
            .into_iter()
            .enumerate()
            .map(|(f, log2_relative)| ApexFragment {
                log2_relative,
                log2_intensity: hit.iter().find(|(i, _)| *i == f).map(|(_, y)| *y),
                background_rate: 0.1,
            })
            .collect()
    }

    #[test]
    fn ln_ndtr_matches_reference_values() {
        // scipy.special.log_ndtr
        for (x, expected) in [
            (0.0, -std::f64::consts::LN_2),
            (1.5, -0.069_143_455_612_234),
            (-2.0, -3.783_184_333_682_031_7),
            (-10.0, -53.231_285_150_512_48),
            (5.0, -2.866_516_129_637_6e-7),
        ] {
            let got = ln_ndtr(x);
            assert!(
                (got - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "{x}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn background_density_integrates_to_one_and_has_a_floor() {
        let m = model();
        let integral: f64 = (0..DENSITY_NUM_BINS)
            .map(|i| {
                m.background
                    .at(DENSITY_MIN_LOG2 + (i as f64 + 0.5) * DENSITY_BIN_LOG2)
                    * DENSITY_BIN_LOG2
            })
            .sum();
        assert!((integral - 1.0).abs() < 0.05);
        assert_eq!(m.background.at(30.0), DENSITY_FLOOR);
        assert!((m.background.default_curve().limit - 8.0).abs() < 0.1);
    }

    #[test]
    fn library_consistent_faint_peptide_beats_a_random_match() {
        // abundance 2^10.5: the six strongest fragments are seen at their predicted intensities
        let consistent: Vec<(usize, f64)> = (0..6).map(|f| (f, 10.5 + library()[f])).collect();
        // as many hits, on weak fragments, at background intensities
        let random: Vec<(usize, f64)> = (6..12).map(|f| (f, 9.0)).collect();
        let c = faint_features(&fragments(&consistent), &model());
        let r = faint_features(&fragments(&random), &model());
        assert!(c.det_faint_llr > r.det_faint_llr + 5.0);
        assert!(c.det_faint_hits > 0.0);
        assert!(c.det_faint_miss <= 0.0);
    }

    #[test]
    fn intense_hits_make_missing_strong_fragments_costly() {
        // the same four hit channels, once at the limit and once far above it, with fragments 0-3 missing
        let faint: Vec<(usize, f64)> = (4..8).map(|f| (f, 8.5)).collect();
        let bright: Vec<(usize, f64)> = (4..8).map(|f| (f, 14.0)).collect();
        let f = faint_features(&fragments(&faint), &model());
        let b = faint_features(&fragments(&bright), &model());
        assert!(b.det_faint_miss < f.det_faint_miss);
    }

    #[test]
    fn no_hits_give_zero_and_empty_inputs_missing() {
        // without hits the best abundance is the lowest, where the misses cost almost nothing
        let none = faint_features(&fragments(&[]), &model());
        assert!(none.det_faint_llr <= 0.0 && none.det_faint_llr > -0.5);
        assert_eq!(faint_features(&[], &model()), FaintFeatures::missing());
        let xic = Array2::<f32>::zeros((3, 5));
        assert!(apex_fragments(&xic, 9, &[1.0, 0.5, 0.2], &[0.1, 0.1, 0.1]).is_none());
        assert!(apex_fragments(&xic, 2, &[1.0, 0.5], &[0.1, 0.1]).is_none());
    }

    #[test]
    fn fit_recovers_the_curve_of_simulated_candidates() {
        // deterministic pseudo-random candidates drawn from CURVE with abundances 2^7 .. 2^13
        let mut state: u64 = 20_261_109;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let normal = |u: &mut dyn FnMut() -> f64| {
            let (u1, u2) = (u(), u());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut candidates = Vec::new();
        for i in 0..1500 {
            let a = 7.0 + 6.0 * (i as f64 / 1500.0);
            let c: Vec<ApexFragment> = library()
                .into_iter()
                .map(|log2_relative| {
                    let y = a + log2_relative + CURVE.scatter * normal(&mut uniform);
                    let seen = (CURVE.width * normal(&mut uniform) + CURVE.limit) < y;
                    ApexFragment {
                        log2_relative,
                        log2_intensity: seen.then_some(y),
                        background_rate: 0.1,
                    }
                })
                .collect();
            candidates.push(c);
        }
        let fitted = fit_curve(&candidates, &model().background);
        assert!((fitted.limit - CURVE.limit).abs() <= 0.5, "{fitted:?}");
        assert!((fitted.width - CURVE.width).abs() <= 0.5, "{fitted:?}");
        assert!((fitted.scatter - CURVE.scatter).abs() <= 0.25, "{fitted:?}");
    }

    #[test]
    fn too_few_hits_fall_back_to_the_default_curve() {
        let fitted = fit_curve(&[fragments(&[(0, 10.0)])], &model().background);
        assert_eq!(fitted, model().background.default_curve());
    }
}
