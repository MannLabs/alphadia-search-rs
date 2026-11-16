use approx::assert_relative_eq;

use super::algorithm::{fit, FitConfig};
use super::censored::{
    fit_censored, CensoredConfig, CensoredResult, MissingFragments, DEFAULT_CURVE_MAX_CELLS,
};
use super::numeric::{log_normal_cdf, nelder_mead, pairwise_sum, quantile};
use super::quantify;

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

fn run(simulation: &Simulation, missing_fragments: MissingFragments, gamma: f64) -> CensoredResult {
    fit_censored(
        &simulation.intensity,
        &simulation.intensity_library,
        &simulation.precursor,
        Some(&simulation.correlation),
        simulation.n_runs,
        simulation.n_precursors,
        &CensoredConfig {
            missing_fragments,
            quality_exponent: gamma,
            curve_max_cells: DEFAULT_CURVE_MAX_CELLS,
        },
    )
}

fn quantity(result: &CensoredResult, simulation: &Simulation, p: usize, run: usize) -> f64 {
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
fn low_abundance_ratio(result: &CensoredResult, simulation: &Simulation) -> f64 {
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
    let dropped = run(&simulation, MissingFragments::Drop, 0.0);
    let censored = run(&simulation, MissingFragments::Censored, 0.0);

    // Then the dropped fit reads the low runs too high, the censored one does not
    let dropped_error = (low_abundance_ratio(&dropped, &simulation) - TRUE_RATIO).abs();
    let censored_error = (low_abundance_ratio(&censored, &simulation) - TRUE_RATIO).abs();
    assert!(dropped_error > 0.05, "dropped error {dropped_error}");
    assert!(
        censored_error < dropped_error / 2.0,
        "censored error {censored_error}, dropped error {dropped_error}"
    );
    let dropout = censored.dropout.unwrap();
    for run in 0..N_RUNS {
        assert!((dropout.limit[run] - DETECTION_LIMIT).abs() < 0.3);
    }
}

#[test]
fn test_a_run_that_never_censors_gives_no_nan() {
    // Given a run without any missing fragment next to runs with missing fragments
    let simulation = censored_simulation(&[1]);

    // When
    let result = run(&simulation, MissingFragments::Censored, 1.5);

    // Then
    let dropout = result.dropout.as_ref().unwrap();
    assert_eq!(dropout.limit[1], f64::NEG_INFINITY);
    assert!(dropout.limit[0].is_finite());
    for p in 0..N_PRECURSORS {
        for run in 0..N_RUNS {
            let n_observed = (0..N_IONS)
                .filter(|ion| {
                    simulation.intensity[run * N_PRECURSORS * N_IONS + p * N_IONS + ion] > 0.0
                })
                .count();
            assert_eq!(
                quantity(&result, &simulation, p, run).is_finite(),
                n_observed >= 2
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
        let result = run(&simulation, MissingFragments::Drop, gamma);
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
fn test_options_off_is_the_plain_fit() {
    // Given
    let simulation = censored_simulation(&[]);

    // When
    let options_off = quantify(
        &simulation.intensity,
        &simulation.intensity_library,
        &simulation.precursor,
        Some(&simulation.correlation),
        N_RUNS,
        N_PRECURSORS,
        &CensoredConfig {
            missing_fragments: MissingFragments::Drop,
            quality_exponent: 0.0,
            curve_max_cells: DEFAULT_CURVE_MAX_CELLS,
        },
    );
    let plain = fit(
        &simulation.intensity,
        &simulation.intensity_library,
        &simulation.precursor,
        N_RUNS,
        N_PRECURSORS,
        &FitConfig::DEFAULT,
    );

    // Then
    assert!(options_off.dropout.is_none());
    assert_eq!(options_off.quantity.len(), plain.quantity.len());
    for (a, b) in options_off.quantity.iter().zip(&plain.quantity) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(options_off.parameters.sigma, plain.parameters.sigma);
    assert_eq!(options_off.parameters.lambda, plain.parameters.lambda);
}

#[test]
fn test_censored_fit_is_deterministic() {
    let simulation = censored_simulation(&[]);
    let a = run(&simulation, MissingFragments::Censored, 1.5);
    let b = run(&simulation, MissingFragments::Censored, 1.5);
    for (x, y) in a.quantity.iter().zip(&b.quantity) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

// Reference values below are from the array and special-function libraries the fit was
// validated against.

fn reference_values() -> Vec<f64> {
    (0..1001)
        .map(|i| ((i * 7919) % 1000) as f64 / 37.0 - 13.0)
        .collect()
}

#[test]
fn test_pairwise_sum_matches_the_reference() {
    let values = reference_values();
    assert_eq!(pairwise_sum(&values), 487.0);
    assert_eq!(pairwise_sum(&values[..300]), 158.10810810810813);
    assert_eq!(pairwise_sum(&values[..13]), -15.432432432432428);
}

#[test]
fn test_quantile_matches_the_reference() {
    let values = reference_values();
    assert_eq!(quantile(&mut values.clone(), 0.02), -12.486486486486486);
    assert_eq!(
        quantile(&mut values[..37].to_vec(), 0.02),
        -12.455135135135135
    );
    assert_eq!(
        quantile(&mut values[..5].to_vec(), 0.02),
        -11.538378378378379
    );
}

#[test]
fn test_log_normal_cdf_matches_the_reference() {
    let reference = [
        (-45.0, -1017.2260942419525),
        (-25.0, -316.63940800802027),
        (-19.5, -194.01696577749752),
        (-3.2, -7.282975502903633),
        (-0.4, -1.065434049189577),
        (0.0, -std::f64::consts::LN_2),
        (0.5, -0.3689464152886563),
        (2.5, -0.006229025485860001),
        (6.0, -9.865876455243719e-10),
        (7.5, -3.190891672910936e-14),
    ];
    for (x, expected) in reference {
        assert_relative_eq!(log_normal_cdf(x), expected, max_relative = 1e-14);
    }
    assert_eq!(log_normal_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
}

#[test]
fn test_nelder_mead_matches_the_reference() {
    let quadratic = |x: &[f64; 3]| {
        (x[0] - 1.0).powi(2) + 10.0 * (x[1] - x[0] * x[0]).powi(2) + (x[2] + 0.5).powi(2)
    };
    assert_eq!(
        nelder_mead(quadratic, [0.3, 0.0, -2.0]),
        [0.9999971012946625, 0.999985703399723, -0.4999807497830643]
    );
    let rosenbrock = |x: &[f64; 3]| {
        100.0 * (x[1] - x[0] * x[0]).powi(2)
            + (1.0 - x[0]).powi(2)
            + 100.0 * (x[2] - x[1] * x[1]).powi(2)
            + (1.0 - x[1]).powi(2)
    };
    assert_eq!(
        nelder_mead(rosenbrock, [-1.2, 1.0, 0.5]),
        [1.0000160542863308, 1.0000294440802586, 1.0000624769428637]
    );
}
