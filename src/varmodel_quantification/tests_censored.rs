use approx::assert_relative_eq;

use super::algorithm::{fit, FitConfig, FitResult};
use super::numeric::{log_normal_cdf, nelder_mead, normal_cdf, quantile};

fn uniform(seed: usize) -> f64 {
    let hashed = (seed as u64 ^ 0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((hashed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn gaussian(seed: usize) -> f64 {
    let u1 = uniform(2 * seed);
    let u2 = uniform(2 * seed + 1);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Run-major linear intensities, library intensities, precursor index and correlations of a
/// simulated experiment: every precursor has `n_ions` fragments with decreasing library
/// intensity, a random library offset per fragment and Gaussian noise.
struct Simulation {
    intensity: Vec<f32>,
    intensity_library: Vec<f32>,
    precursor: Vec<usize>,
    correlation: Vec<f32>,
    n_runs: usize,
    n_precursors: usize,
}

const N_IONS: usize = 8;

/// `log2_intensity(precursor, ion, run)` gives the log2 fragment intensity, NaN for missing.
fn simulate(
    n_precursors: usize,
    n_runs: usize,
    log2_intensity: impl Fn(usize, usize, usize, f64) -> f64,
    correlation: impl Fn(usize, usize, usize) -> f64,
) -> Simulation {
    let n_total = n_precursors * N_IONS;
    let library: Vec<f64> = (0..N_IONS).map(|ion| 1.0 - 0.1 * ion as f64).collect();
    let library_sum: f64 = library.iter().sum();
    let mut intensity = vec![0.0f32; n_runs * n_total];
    let mut correlations = vec![0.0f32; n_runs * n_total];
    for p in 0..n_precursors {
        for ion in 0..N_IONS {
            let column = p * N_IONS + ion;
            let log_share = (library[ion] / library_sum).log2() + 0.2 * gaussian(column);
            for run in 0..n_runs {
                let value = log2_intensity(p, ion, run, log_share);
                if value.is_finite() {
                    intensity[run * n_total + column] = value.exp2() as f32;
                }
                correlations[run * n_total + column] = correlation(p, ion, run) as f32;
            }
        }
    }
    Simulation {
        intensity,
        intensity_library: (0..n_total).map(|i| library[i % N_IONS] as f32).collect(),
        precursor: (0..n_total).map(|i| i / N_IONS).collect(),
        correlation: correlations,
        n_runs,
        n_precursors,
    }
}

fn run(simulation: &Simulation, fragment_censoring_correction: bool, gamma: f64) -> FitResult {
    fit(
        &simulation.intensity,
        &simulation.intensity_library,
        &simulation.precursor,
        &simulation.correlation,
        simulation.n_runs,
        simulation.n_precursors,
        &FitConfig {
            fragment_censoring_correction,
            quality_exponent: gamma,
            ..FitConfig::DEFAULT
        },
    )
}

fn quantity(result: &FitResult, simulation: &Simulation, p: usize, run: usize) -> f64 {
    result.quantity[run * simulation.n_precursors + p]
}

const N_PRECURSORS: usize = 400;
const N_RUNS: usize = 6;
const TRUE_RATIO: f64 = 1.0;
const DETECTION_LIMIT: f64 = 5.0;

fn abundance(p: usize) -> f64 {
    7.0 + 4.0 * p as f64 / N_PRECURSORS as f64
}

/// Runs 3-5 are TRUE_RATIO log2 above runs 0-2; a fragment below the detection limit (with
/// some spread) is missing, except in the runs listed in `never_missing`.
fn censored_simulation(never_missing: &[usize]) -> Simulation {
    simulate(
        N_PRECURSORS,
        N_RUNS,
        |p, ion, run, log_share| {
            let seed = 1_000_000 + (p * N_IONS + ion) * N_RUNS + run;
            let true_value = abundance(p) + if run >= 3 { TRUE_RATIO } else { 0.0 } + log_share;
            let value = true_value + 0.3 * gaussian(seed);
            let limit = DETECTION_LIMIT + 0.3 * gaussian(seed + 7_777_777);
            if value < limit && !never_missing.contains(&run) {
                f64::NAN
            } else {
                value
            }
        },
        |_, _, _| 1.0,
    )
}

/// Fragments of a precursor missing in its low runs 0-2.
fn n_missing_low(simulation: &Simulation, p: usize) -> usize {
    (0..3)
        .flat_map(|run| (0..N_IONS).map(move |ion| (run, ion)))
        .filter(|&(run, ion)| {
            simulation.intensity[run * N_PRECURSORS * N_IONS + p * N_IONS + ion] == 0.0
        })
        .count()
}

/// Median of the estimated ratio over the precursors quantified in every run that miss at
/// least a sixth of their fragments in the low runs.
fn low_abundance_ratio(result: &FitResult, simulation: &Simulation) -> f64 {
    let mut ratios: Vec<f64> = (0..N_PRECURSORS)
        .filter(|&p| n_missing_low(simulation, p) >= 3 * N_IONS / 6)
        .filter_map(|p| {
            let q: Vec<f64> = (0..N_RUNS)
                .map(|run| quantity(result, simulation, p, run))
                .collect();
            q.iter()
                .all(|v| v.is_finite())
                .then(|| (q[3] + q[4] + q[5] - q[0] - q[1] - q[2]) / 3.0)
        })
        .collect();
    assert!(ratios.len() > 50);
    ratios.sort_by(f64::total_cmp);
    ratios[ratios.len() / 2]
}

#[test]
fn test_censoring_removes_ratio_compression_near_the_detection_limit() {
    // Given precursors whose low runs lose fragments below the detection limit
    let simulation = censored_simulation(&[]);

    // When
    let dropped = run(&simulation, false, 0.0);
    let censored = run(&simulation, true, 0.0);

    // Then the dropped fit reads the low runs too high, the censored one does not
    let dropped_error = (low_abundance_ratio(&dropped, &simulation) - TRUE_RATIO).abs();
    let censored_error = (low_abundance_ratio(&censored, &simulation) - TRUE_RATIO).abs();
    assert!(dropped_error > 0.05, "dropped error {dropped_error}");
    assert!(
        censored_error < dropped_error / 2.0,
        "censored error {censored_error}, dropped error {dropped_error}"
    );
    let dropout = censored.dropout.unwrap();
    for curve in &dropout.curves {
        assert!((curve.as_ref().unwrap().limit - DETECTION_LIMIT).abs() < 0.3);
    }
}

#[test]
fn test_a_run_that_never_censors_spoils_no_quantity() {
    // Given a run without any missing fragment next to runs with missing fragments
    let simulation = censored_simulation(&[1]);

    // When
    let result = run(&simulation, true, 1.5);

    // Then
    let dropout = result.dropout.as_ref().unwrap();
    assert!(dropout.curves[1].is_none());
    assert!(dropout.curves[0].is_some());
    for p in 0..N_PRECURSORS {
        for run in 0..N_RUNS {
            let n_observed = (0..N_IONS)
                .filter(|ion| {
                    simulation.intensity[run * N_PRECURSORS * N_IONS + p * N_IONS + ion] > 0.0
                })
                .count();
            let estimate = quantity(&result, &simulation, p, run);
            assert_eq!(estimate.is_finite(), n_observed >= 2);
            if estimate.is_finite() {
                let truth = abundance(p) + if run >= 3 { TRUE_RATIO } else { 0.0 };
                assert!(
                    (estimate - truth).abs() < 1.0,
                    "precursor {p} run {run}: {estimate} vs {truth}"
                );
            }
        }
    }
}

#[test]
fn test_precursors_without_missing_fragments_keep_the_uncensored_fit() {
    // Given
    let simulation = censored_simulation(&[]);

    // When
    let dropped = run(&simulation, false, 1.5);
    let censored = run(&simulation, true, 1.5);

    // Then
    let complete: Vec<usize> = (0..N_PRECURSORS)
        .filter(|&p| {
            (0..N_RUNS).all(|run| {
                (0..N_IONS).all(|ion| {
                    simulation.intensity[run * N_PRECURSORS * N_IONS + p * N_IONS + ion] > 0.0
                })
            })
        })
        .collect();
    assert!(complete.len() > 100);
    for p in complete {
        for run in 0..N_RUNS {
            assert_eq!(
                quantity(&censored, &simulation, p, run).to_bits(),
                quantity(&dropped, &simulation, p, run).to_bits()
            );
        }
    }
}

#[test]
fn test_quality_weights_downweight_interfered_values() {
    // Given one fragment per precursor interfered (log2 +0.6, three times the noise) in run 1,
    // without XIC correlation there
    let interfered = |p: usize, ion: usize, run: usize| run == 1 && ion == 1 + p % (N_IONS - 1);
    let true_q = |p: usize, run: usize| 12.0 + 0.5 * gaussian(p) + 0.3 * run as f64;
    let simulation = simulate(
        300,
        4,
        |p, ion, run, log_share| {
            let seed = 5_000_000 + (p * N_IONS + ion) * 4 + run;
            true_q(p, run)
                + log_share
                + 0.2 * gaussian(seed)
                + if interfered(p, ion, run) { 0.6 } else { 0.0 }
        },
        |p, ion, run| if interfered(p, ion, run) { 0.0 } else { 0.97 },
    );

    // When
    let error = |gamma: f64| {
        let result = run(&simulation, false, gamma);
        (0..300)
            .map(|p| {
                let estimate =
                    quantity(&result, &simulation, p, 1) - quantity(&result, &simulation, p, 0);
                (estimate - (true_q(p, 1) - true_q(p, 0))).abs()
            })
            .sum::<f64>()
            / 300.0
    };
    let unweighted = error(0.0);
    let weighted = error(1.5);

    // Then
    assert!(
        weighted < 0.7 * unweighted,
        "weighted {weighted}, unweighted {unweighted}"
    );
}

#[test]
fn test_a_vanishing_quality_exponent_gives_the_fit_without_quality_weights() {
    // Given correlations far from 1
    let mut simulation = censored_simulation(&[]);
    simulation.correlation.iter_mut().for_each(|c| *c = 0.2);

    // When
    let without = run(&simulation, true, 0.0);
    let vanishing = run(&simulation, true, 1e-12);

    // Then
    for (a, b) in without.quantity.iter().zip(&vanishing.quantity) {
        assert_eq!(a.is_nan(), b.is_nan());
        if a.is_finite() {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}

#[test]
fn test_censored_fit_is_deterministic() {
    let simulation = censored_simulation(&[]);
    let a = run(&simulation, true, 1.5);
    let b = run(&simulation, true, 1.5);
    for (x, y) in a.quantity.iter().zip(&b.quantity) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

/// Obviously correct where the CDF does not underflow: the log of the complementary error
/// function form.
fn reference_log_normal_cdf(x: f64) -> f64 {
    if x > 0.0 {
        (-0.5 * libm::erfc(x * std::f64::consts::FRAC_1_SQRT_2)).ln_1p()
    } else {
        (0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)).ln()
    }
}

#[test]
fn test_log_normal_cdf_matches_the_reference() {
    // the reference covers the switch to the asymptotic series at -20 up to where erfc underflows
    for i in 0..=900 {
        let x = -37.0 + 0.05 * i as f64;
        assert_relative_eq!(
            log_normal_cdf(x),
            reference_log_normal_cdf(x),
            max_relative = 1e-12
        );
    }
    assert_eq!(normal_cdf(0.0), 0.5);
}

#[test]
fn test_log_normal_cdf_follows_the_leading_tail_term_beyond_underflow() {
    for x in [-40.0f64, -100.0, -1e4] {
        let leading = -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_relative_eq!(log_normal_cdf(x), leading, max_relative = 1e-3);
        assert!(log_normal_cdf(x) < leading);
    }
}

#[test]
fn test_quantile_interpolates_the_sorted_values() {
    let values: Vec<f64> = (0..101).map(|i| ((i * 37) % 101) as f64).collect();
    for (probability, expected) in [(0.0, 0.0), (0.02, 2.0), (0.025, 2.5), (1.0, 100.0)] {
        assert_relative_eq!(quantile(&mut values.clone(), probability), expected);
    }
    assert_eq!(quantile(&mut [3.0], 0.5), 3.0);
}

#[test]
fn test_nelder_mead_finds_the_minimum() {
    let quadratic = |x: &[f64; 3]| {
        (x[0] - 1.0).powi(2) + 10.0 * (x[1] - x[0] * x[0]).powi(2) + (x[2] + 0.5).powi(2)
    };
    let minimum = nelder_mead(quadratic, [0.3, 0.0, -2.0]);
    for (found, expected) in minimum.iter().zip([1.0, 1.0, -0.5]) {
        assert!((found - expected).abs() < 1e-2, "{minimum:?}");
    }
    let rosenbrock = |x: &[f64; 2]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
    let minimum = nelder_mead(rosenbrock, [-1.2, 1.0]);
    for found in minimum {
        assert!((found - 1.0).abs() < 1e-2, "{minimum:?}");
    }
}
