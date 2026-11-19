use rayon::prelude::*;

use super::algorithm::{Group, State};
use super::numeric::{expit, log_normal_cdf, logit, nelder_mead, quantile};

pub const CURVE_ROUNDS: usize = 2;
/// Precursor runs with this many observed fragments have a trustworthy expected intensity for
/// every fragment.
const CURVE_MIN_OBSERVED: usize = 4;
/// The likelihood counts missing and observed measurements in bins of this width of the
/// expected log2 intensity; the CDF barely changes within a bin.
const CURVE_BIN_WIDTH: f64 = 0.02;
const CURVE_START_QUANTILE: f64 = 0.02;
const CURVE_START_PLATEAU: f64 = 1e-3;
/// Keeps the log-likelihood finite and the plateau inside (0, 1), so that the censoring
/// probability of a missing measurement is always defined.
const PROBABILITY_FLOOR: f64 = 1e-12;
/// log(sqrt(2 pi))
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// `P(missing | mu) = plateau + (1 - plateau) * Phi((limit - mu) / width)` of one run.
pub struct Curve {
    pub limit: f64,
    pub width: f64,
    pub plateau: f64,
}

/// One curve per run, `None` for a run without missing (or without observed) measurements to
/// fit it on: its missing measurements carry no censoring information.
pub struct Dropout {
    pub curves: Vec<Option<Curve>>,
}

impl Dropout {
    pub fn fit(groups: &[Group], states: &[State], n_runs: usize) -> Self {
        Self {
            curves: (0..n_runs)
                .into_par_iter()
                .map(|run| Curve::fit(groups, states, run))
                .collect(),
        }
    }
}

/// Measurements of one run binned by their expected log2 intensity `mu`.
struct Bin {
    mean_mu: f64,
    n_missing: f64,
    n_observed: f64,
}

impl Curve {
    /// Value (in units of `log2(I / s)`) and weight of a missing measurement with expected log2
    /// intensity `mu`: its conditional mean under the censored component, weighted by the
    /// posterior probability that it was censored.
    pub fn pseudo_observation(&self, mu: f64, log_share: f64) -> (f64, f64) {
        let z = (self.limit - mu) / self.width;
        let log_cdf = log_normal_cdf(z);
        let mills_ratio = (-0.5 * z * z - LOG_SQRT_2PI - log_cdf).exp();
        let cdf = log_cdf.exp();
        let censored = (1.0 - self.plateau) * cdf;
        let responsibility = censored / (self.plateau + censored);
        (
            mu - self.width * mills_ratio - log_share,
            responsibility / (self.width * self.width),
        )
    }

    /// Maximum likelihood on all measurements of the precursor runs with at least
    /// `CURVE_MIN_OBSERVED` observed fragments: the binomial likelihood of the missing counts in
    /// `CURVE_BIN_WIDTH` bins of the expected intensity, each at its mean expected intensity.
    fn fit(groups: &[Group], states: &[State], run: usize) -> Option<Self> {
        let mut cells: Vec<(i64, f64, bool)> = Vec::new();
        for (group, state) in groups.iter().zip(states) {
            if group.n_observed(run) < CURVE_MIN_OBSERVED {
                continue;
            }
            for ion in 0..group.n_ions {
                let mu = state.quantity[run] + state.offset[ion] + group.log_share[ion];
                let bin = (mu / CURVE_BIN_WIDTH).floor() as i64;
                cells.push((bin, mu, !group.observed(ion, run)));
            }
        }
        let n_missing = cells.iter().filter(|cell| cell.2).count();
        if n_missing == 0 || n_missing == cells.len() {
            return None;
        }
        let mut observed_mu: Vec<f64> = cells
            .iter()
            .filter(|cell| !cell.2)
            .map(|cell| cell.1)
            .collect();
        let missing_share = n_missing as f64 / cells.len() as f64;
        let start = [
            quantile(&mut observed_mu, CURVE_START_QUANTILE),
            0.0,
            logit((missing_share / 2.0).max(CURVE_START_PLATEAU)),
        ];

        cells.sort_unstable_by_key(|cell| cell.0);
        let bins: Vec<Bin> = cells
            .chunk_by(|a, b| a.0 == b.0)
            .map(|cells| {
                let n_missing = cells.iter().filter(|cell| cell.2).count() as f64;
                Bin {
                    mean_mu: cells.iter().map(|cell| cell.1).sum::<f64>() / cells.len() as f64,
                    n_missing,
                    n_observed: cells.len() as f64 - n_missing,
                }
            })
            .collect();
        let negative_log_likelihood = |theta: &[f64; 3]| {
            let [limit, log_width, plateau_logit] = *theta;
            let plateau = expit(plateau_logit);
            let width = log_width.exp();
            -bins
                .iter()
                .map(|bin| {
                    let cdf = log_normal_cdf((limit - bin.mean_mu) / width).exp();
                    let p = (plateau + (1.0 - plateau) * cdf)
                        .clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
                    bin.n_missing * p.ln() + bin.n_observed * (-p).ln_1p()
                })
                .sum::<f64>()
        };
        let [limit, log_width, plateau_logit] = nelder_mead(negative_log_likelihood, start);
        Some(Self {
            limit,
            width: log_width.exp(),
            plateau: expit(plateau_logit).clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR),
        })
    }
}
