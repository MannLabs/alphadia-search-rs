//! These tests cover the removal of a run's drift with both statistics, the level of a run
//! that must survive it, the ions and runs the estimator ignores, the merging of sparse bins
//! and the retention time fallback.

use approx::assert_abs_diff_eq;

use super::algorithm::{correct_drift, DriftParameters, Statistic};

const N_RUNS: usize = 6;
const N_IONS: usize = 2000;
const GRADIENT_SECONDS: f64 = 1500.0;
const MEAN_LEVEL_LOG2: f64 = 15.0;
/// Spread of the ion levels and of the noise on every cell, in log2. The noise has to be
/// wider than the mode histogram's smoothing window for the mode to be resolvable.
const LEVEL_SPREAD_LOG2: f64 = 4.0;
const NOISE_LOG2: f64 = 0.3;
const CORRELATION: f64 = 0.95;

/// Deterministic pseudo-random value in `[-0.5, 0.5)`.
fn noise(seed: usize) -> f64 {
    let hashed = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (hashed >> 33) as f64 / (1u64 << 31) as f64 - 0.5
}

/// Run-major tables of `N_RUNS` runs that share all ions, with `drift_log2` imposed on run 0
/// over the gradient. The ions elute evenly spread over the gradient.
struct Tables {
    intensity: Vec<f64>,
    correlation: Vec<f64>,
    rt: Vec<f64>,
}

impl Tables {
    fn new(drift_log2: f64) -> Self {
        let rt: Vec<f64> = (0..N_IONS)
            .map(|ion| GRADIENT_SECONDS * ion as f64 / N_IONS as f64)
            .collect();
        let mut intensity = Vec::with_capacity(N_RUNS * N_IONS);
        for run in 0..N_RUNS {
            for ion in 0..N_IONS {
                let level = MEAN_LEVEL_LOG2
                    + noise(ion) * LEVEL_SPREAD_LOG2
                    + noise(ion + 7919 * (run + 1)) * NOISE_LOG2;
                let drift = if run == 0 {
                    drift_log2 * rt[ion] / GRADIENT_SECONDS
                } else {
                    0.0
                };
                intensity.push((level + drift).exp2());
            }
        }
        Self {
            intensity,
            correlation: vec![CORRELATION; N_RUNS * N_IONS],
            rt: rt.repeat(N_RUNS),
        }
    }

    fn correct(&self, parameters: &DriftParameters) -> Vec<f64> {
        correct_drift(
            &self.intensity,
            &self.correlation,
            &self.rt,
            N_IONS,
            parameters,
        )
        .0
    }

    /// Log2 change the correction applied to one cell.
    fn change(&self, corrected: &[f64], run: usize, ion: usize) -> f64 {
        (corrected[run * N_IONS + ion] / self.intensity[run * N_IONS + ion]).log2()
    }
}

fn parameters(statistic: Statistic) -> DriftParameters {
    DriftParameters {
        bin_seconds: 30.0,
        min_correlation: 0.8,
        min_observations: 3,
        min_ions_per_bin: 100,
        min_ions_per_run: 500,
        statistic,
    }
}

/// Least squares slope of `y` over `x`.
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let covariance: f64 = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let variance: f64 = x.iter().map(|x| (x - mean_x).powi(2)).sum();
    covariance / variance
}

/// Slope of the log2 deviation of `run` from the mean of the other runs, per 1000 seconds.
fn deviation_slope(intensity: &[f64], rt: &[f64], run: usize) -> f64 {
    let deviation: Vec<f64> = (0..N_IONS)
        .map(|ion| {
            let others: f64 = (0..N_RUNS)
                .filter(|&other| other != run)
                .map(|other| intensity[other * N_IONS + ion].log2())
                .sum::<f64>()
                / (N_RUNS - 1) as f64;
            intensity[run * N_IONS + ion].log2() - others
        })
        .collect();
    let rt_per_1000s: Vec<f64> = (0..N_IONS).map(|ion| rt[ion] / 1000.0).collect();
    slope(&rt_per_1000s, &deviation)
}

fn mean_log2(intensity: &[f64], run: usize) -> f64 {
    intensity[run * N_IONS..(run + 1) * N_IONS]
        .iter()
        .map(|intensity| intensity.log2())
        .sum::<f64>()
        / N_IONS as f64
}

#[test]
fn test_removes_the_drift_of_one_run() {
    for statistic in [Statistic::Mode, Statistic::Median] {
        let tables = Tables::new(0.3);
        assert!(deviation_slope(&tables.intensity, &tables.rt, 0) > 0.15);

        let corrected = tables.correct(&parameters(statistic));

        assert!(deviation_slope(&corrected, &tables.rt, 0).abs() < 0.01);
    }
}

#[test]
fn test_keeps_the_level_of_every_run() {
    let tables = Tables::new(0.3);

    let corrected = tables.correct(&parameters(Statistic::Mode));

    // the correction changes the shape along the gradient only
    for run in 0..N_RUNS {
        assert_abs_diff_eq!(
            mean_log2(&corrected, run),
            mean_log2(&tables.intensity, run),
            epsilon = 1e-9
        );
    }
}

#[test]
fn test_leaves_runs_without_drift_almost_unchanged() {
    let tables = Tables::new(0.0);

    let corrected = tables.correct(&parameters(Statistic::Mode));

    for run in 0..N_RUNS {
        for ion in 0..N_IONS {
            assert!(tables.change(&corrected, run, ion).abs() < 0.1);
        }
    }
}

#[test]
fn test_keeps_missing_values_missing() {
    let mut tables = Tables::new(0.3);
    for ion in (0..N_IONS).step_by(7) {
        tables.intensity[ion] = 0.0;
    }

    let corrected = tables.correct(&parameters(Statistic::Mode));

    for ion in 0..N_IONS {
        if ion % 7 == 0 {
            assert_eq!(corrected[ion], 0.0);
        } else {
            assert!(corrected[ion] > 0.0);
        }
    }
}

#[test]
fn test_ignores_poorly_correlating_ions() {
    // the well correlating ions do not drift, the poorly correlating ones do
    let mut tables = Tables::new(0.0);
    for ion in (0..N_IONS).step_by(2) {
        tables.intensity[ion] *= (tables.rt[ion] / GRADIENT_SECONDS).exp2();
        for run in 0..N_RUNS {
            tables.correlation[run * N_IONS + ion] = 0.3;
        }
    }

    let corrected = tables.correct(&parameters(Statistic::Mode));

    let good_ions: Vec<usize> = (1..N_IONS).step_by(2).collect();
    let rt_per_1000s: Vec<f64> = good_ions
        .iter()
        .map(|&ion| tables.rt[ion] / 1000.0)
        .collect();
    let change: Vec<f64> = good_ions
        .iter()
        .map(|&ion| tables.change(&corrected, 0, ion))
        .collect();
    assert!(slope(&rt_per_1000s, &change).abs() < 0.02);
}

#[test]
fn test_skips_a_run_with_too_few_usable_ions() {
    let tables = Tables::new(0.3);
    let parameters = DriftParameters {
        min_ions_per_run: N_IONS + 1,
        ..parameters(Statistic::Mode)
    };

    let (corrected, curves) = correct_drift(
        &tables.intensity,
        &tables.correlation,
        &tables.rt,
        N_IONS,
        &parameters,
    );

    assert!(curves.iter().all(|curve| curve.is_none()));
    assert_eq!(corrected, tables.intensity);
}

#[test]
fn test_merges_sparse_bins_and_appends_trailing_ions() {
    // 250 ions one second apart, 10 second bins hold 10 ions each
    let n_ions = 250;
    let rt: Vec<f64> = (0..n_ions).map(|ion| ion as f64).collect();
    let deviating_run: Vec<f64> = rt
        .iter()
        .map(|&rt| {
            if rt < 100.0 {
                0.1_f64.exp2()
            } else {
                0.5_f64.exp2()
            }
        })
        .collect();
    let intensity = [deviating_run, vec![1.0; n_ions]].concat();
    let parameters = DriftParameters {
        bin_seconds: 10.0,
        min_observations: 2,
        min_ions_per_bin: 100,
        min_ions_per_run: 0,
        ..parameters(Statistic::Median)
    };

    let (_, curves) = correct_drift(
        &intensity,
        &vec![CORRELATION; 2 * n_ions],
        &rt.repeat(2),
        n_ions,
        &parameters,
    );

    // [0, 100) and [100, 250), the trailing 50 ions merged into the second bin
    let curve = curves[0].as_ref().expect("the run has a curve");
    assert_eq!(curve.n_ions, n_ions);
    assert_abs_diff_eq!(curve.rt[0], 49.5, epsilon = 1e-9);
    assert_abs_diff_eq!(curve.rt[1], 174.5, epsilon = 1e-9);
    assert_eq!(curve.rt.len(), 2);
}

#[test]
fn test_places_an_ion_without_rt_at_its_mean_rt() {
    let tables = Tables::new(0.3);
    // run 0 has no identification for this ion, late in the gradient where the drift is largest
    let ion = N_IONS - 100;
    let mut rt = tables.rt.clone();
    rt[ion] = f64::NAN;

    let corrected = correct_drift(
        &tables.intensity,
        &tables.correlation,
        &rt,
        N_IONS,
        &parameters(Statistic::Mode),
    )
    .0;

    assert_abs_diff_eq!(
        tables.change(&corrected, 0, ion),
        tables.change(&corrected, 0, ion + 1),
        epsilon = 0.01
    );
}

#[test]
fn test_rejects_an_unknown_statistic() {
    assert!(Statistic::from_name("mode").is_ok());
    assert!(Statistic::from_name("median").is_ok());
    assert!(Statistic::from_name("mean").is_err());
}
