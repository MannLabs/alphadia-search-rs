use approx::assert_abs_diff_eq;

use super::algorithm::{fit, validate_shapes, FitConfig};

fn noise(seed: usize) -> f64 {
    let hashed = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (hashed >> 33) as f64 / (1u64 << 31) as f64 - 0.5
}

fn run_major(log2_values: &[Vec<f64>]) -> Vec<f32> {
    let n_runs = log2_values[0].len();
    let mut intensity = vec![0.0f32; n_runs * log2_values.len()];
    for (ion, values) in log2_values.iter().enumerate() {
        for (run, &value) in values.iter().enumerate() {
            if value.is_finite() {
                intensity[run * log2_values.len() + ion] = value.exp2() as f32;
            }
        }
    }
    intensity
}

fn config(tau: Option<f64>, lambda: Option<f64>) -> FitConfig {
    FitConfig {
        tau,
        lambda,
        ..FitConfig::DEFAULT
    }
}

fn quantities(quantity: &[f64], index: usize, n_runs: usize, n_precursors: usize) -> Vec<f64> {
    (0..n_runs)
        .map(|run| quantity[run * n_precursors + index])
        .collect()
}

#[test]
fn test_recovers_run_ratios_with_an_interfered_fragment() {
    // Given one precursor following the library pattern with small offsets and noise, one
    // minor fragment interfered 8-fold in run 2 and one missing cell
    let library = [1.0f32, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    let true_q = [10.0, 11.0, 8.0, 10.5];
    let library_sum: f64 = library.iter().map(|&l| l as f64).sum();
    let mut log2_values: Vec<Vec<f64>> = library
        .iter()
        .enumerate()
        .map(|(ion, &l)| {
            let offset = 0.3 * noise(ion);
            true_q
                .iter()
                .enumerate()
                .map(|(run, q)| {
                    q + (l as f64 / library_sum).log2() + offset + 0.05 * noise(100 + ion * 4 + run)
                })
                .collect()
        })
        .collect();
    log2_values[6][2] += 3.0;
    log2_values[1][3] = f64::NAN;

    // When
    let result = fit(
        &run_major(&log2_values),
        &library,
        &[0; 8],
        4,
        1,
        &config(Some(0.7), Some(1.0)),
    );

    // Then
    for run in 0..4 {
        assert_abs_diff_eq!(
            result.quantity[run] - result.quantity[0],
            true_q[run] - true_q[0],
            epsilon = 0.1
        );
    }
}

#[test]
fn test_quantity_is_on_the_summed_fragment_scale() {
    // Given fragments that follow the library pattern exactly
    let library = [1.0f32, 0.5, 0.25];
    let totals = [1000.0f64, 4000.0];
    let log2_values: Vec<Vec<f64>> = library
        .iter()
        .map(|&l| {
            totals
                .iter()
                .map(|total| (total * l as f64 / 1.75).log2())
                .collect()
        })
        .collect();

    // When
    let result = fit(
        &run_major(&log2_values),
        &library,
        &[0; 3],
        2,
        1,
        &FitConfig::DEFAULT,
    );

    // Then
    for (run, total) in totals.iter().enumerate() {
        assert_abs_diff_eq!(result.quantity[run].exp2(), total, epsilon = 1e-2);
    }
}

#[test]
fn test_min_ions_gives_nan() {
    // Given a precursor with two fragments in run 0 and one in run 1
    let log2_values = vec![
        vec![0.0, f64::NAN],
        vec![0.1, f64::NAN],
        vec![f64::NAN, 0.2],
    ];

    // When
    let result = fit(
        &run_major(&log2_values),
        &[1.0; 3],
        &[0; 3],
        2,
        1,
        &config(Some(0.7), Some(1.0)),
    );

    // Then
    assert!(result.quantity[0].is_finite());
    assert!(result.quantity[1].is_nan());
}

#[test]
fn test_partially_observed_precursors_keep_their_observed_runs() {
    // Given precursor 0 complete and precursor 1 observed in runs 0 and 1 only, with its
    // fragments interleaved with those of precursor 0
    let log2_values = vec![
        vec![0.0, 1.0, 2.0],
        vec![5.0, 6.0, f64::NAN],
        vec![0.2, 1.2, f64::NAN],
        vec![5.1, 6.1, f64::NAN],
        vec![0.1, 1.1, 2.1],
    ];
    let precursor = [0, 1, 0, 1, 0];

    // When
    let result = fit(
        &run_major(&log2_values),
        &[1.0; 5],
        &precursor,
        3,
        2,
        &config(Some(1.0), Some(1.0)),
    );

    // Then
    let q0 = quantities(&result.quantity, 0, 3, 2);
    let q1 = quantities(&result.quantity, 1, 3, 2);
    assert!(q0.iter().all(|q| q.is_finite()));
    assert!(q1[0].is_finite() && q1[1].is_finite() && q1[2].is_nan());
    assert_abs_diff_eq!(q1[1] - q1[0], 1.0, epsilon = 1e-5);
}

#[test]
fn test_fragments_without_library_intensity_are_ignored() {
    // Given a precursor with one wild fragment that has no library intensity
    let true_q = [0.0, 1.0, -1.0];
    let log2_values: Vec<Vec<f64>> = [10.0, 9.0, 8.0, 12.0]
        .iter()
        .enumerate()
        .map(|(ion, level)| {
            true_q
                .iter()
                .enumerate()
                .map(|(run, q)| {
                    let wild = if ion == 3 { 3.0 * run as f64 } else { 0.0 };
                    level + q + wild + 0.02 * noise(ion * 3 + run)
                })
                .collect()
        })
        .collect();

    // When
    let result = fit(
        &run_major(&log2_values),
        &[1.0, 0.5, 0.25, 0.0],
        &[0; 4],
        3,
        1,
        &config(Some(1.5), Some(1.0)),
    );

    // Then
    for run in 0..3 {
        assert_abs_diff_eq!(
            result.quantity[run] - result.quantity[0],
            true_q[run],
            epsilon = 0.05
        );
    }
}

#[test]
fn test_derives_loss_scale_and_shrinkage_from_the_data() {
    // Given many precursors whose fragments scatter with a known noise and offset spread
    let (n_precursors, n_fragments, n_runs) = (400, 6, 4);
    let (noise_sd, offset_sd) = (0.2, 0.5);
    // a uniform variable on [-0.5, 0.5) has a standard deviation of 1 / sqrt(12)
    let uniform_to_sd = 12f64.sqrt();
    let true_q: Vec<Vec<f64>> = (0..n_precursors)
        .map(|p| {
            (0..n_runs)
                .map(|run| 12.0 + 2.0 * noise(1_000_000 + p * n_runs + run))
                .collect()
        })
        .collect();
    let mut log2_values = Vec::new();
    let mut precursor = Vec::new();
    for (p, q) in true_q.iter().enumerate() {
        for f in 0..n_fragments {
            let ion = p * n_fragments + f;
            let offset = offset_sd * uniform_to_sd * noise(2_000_000 + ion);
            log2_values.push(
                (0..n_runs)
                    .map(|run| {
                        q[run]
                            + offset
                            + noise_sd * uniform_to_sd * noise(3_000_000 + ion * n_runs + run)
                    })
                    .collect(),
            );
            precursor.push(p);
        }
    }

    // When fitted with automatic loss scale and shrinkage
    let result = fit(
        &run_major(&log2_values),
        &vec![1.0; precursor.len()],
        &precursor,
        n_runs,
        n_precursors,
        &FitConfig::DEFAULT,
    );

    // Then the first-sweep residuals carry noise and offsets, the loss scale follows them
    assert!(result.parameters.sigma > noise_sd);
    assert_abs_diff_eq!(
        result.parameters.tau,
        2.385 * result.parameters.sigma,
        epsilon = 1e-9
    );
    assert!(result.parameters.lambda > 0.0);
    let mut error = 0.0;
    for (p, q) in true_q.iter().enumerate() {
        let est = quantities(&result.quantity, p, n_runs, n_precursors);
        for run in 1..n_runs {
            error += ((est[run] - est[0]) - (q[run] - q[0])).abs();
        }
    }
    assert!(error / ((n_precursors * (n_runs - 1)) as f64) < 0.15);
}

#[test]
fn test_stops_at_convergence_with_exact_ratios() {
    // Given a clean precursor
    let log2_values = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.1, 1.1, 2.1],
        vec![-0.1, 0.9, 1.9],
    ];

    // When fitted with a tight tolerance and many allowed sweeps
    let config = FitConfig {
        tau: Some(1.0),
        lambda: Some(1.0),
        max_sweeps: 50,
        tolerance: 1e-6,
        ..FitConfig::DEFAULT
    };
    let result = fit(&run_major(&log2_values), &[1.0; 3], &[0; 3], 3, 1, &config);

    // Then
    for run in 0..3 {
        assert_abs_diff_eq!(
            result.quantity[run] - result.quantity[0],
            run as f64,
            epsilon = 1e-5
        );
    }
}

#[test]
fn test_noise_free_data_downweights_an_outlier() {
    // Given exact fragments over four runs and one outlier: most residuals are exactly zero,
    // the unweighted mean would be 4 / 6 log2 too high in run 1. (With two runs a free
    // fragment offset cannot tell an outlier in one run from a library offset.)
    let mut log2_values = vec![vec![0.0, 1.0, 2.0, 3.0]; 6];
    log2_values[5][1] += 4.0;

    // When
    let result = fit(
        &run_major(&log2_values),
        &[1.0; 6],
        &[0; 6],
        4,
        1,
        &FitConfig::DEFAULT,
    );

    // Then
    assert_abs_diff_eq!(result.quantity[1] - result.quantity[0], 1.0, epsilon = 0.1);
}

#[test]
fn test_validate_shapes_rejects_mismatches() {
    assert!(validate_shapes((2, 3), 3, &[0, 0, 1], 2).is_ok());
    assert!(validate_shapes((2, 3), 2, &[0, 0, 1], 2).is_err());
    assert!(validate_shapes((2, 3), 3, &[0, 1], 2).is_err());
    assert!(validate_shapes((2, 3), 3, &[0, 0, 2], 2).is_err());
}

#[test]
fn test_validate_rejects_invalid_parameters() {
    assert!(config(Some(0.0), None).validate().is_err());
    assert!(config(None, Some(-1.0)).validate().is_err());
    assert!(FitConfig {
        min_ions: 0,
        ..FitConfig::DEFAULT
    }
    .validate()
    .is_err());
    assert!(FitConfig::DEFAULT.validate().is_ok());
}

fn simulate_offsets_and_noise(
    offset_sd: f64,
    noise_sd: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>) {
    let (n_precursors, n_fragments, n_runs) = (1000, 10, 6);
    // a uniform variable on [-0.5, 0.5) has a standard deviation of 1 / sqrt(12)
    let uniform_to_sd = 12f64.sqrt();
    let true_q: Vec<Vec<f64>> = (0..n_precursors)
        .map(|p| {
            (0..n_runs)
                .map(|run| 15.0 + 4.0 * noise(4_000_000 + p * n_runs + run))
                .collect()
        })
        .collect();
    let mut log2_values = Vec::new();
    let mut precursor = Vec::new();
    for (p, q) in true_q.iter().enumerate() {
        for f in 0..n_fragments {
            let ion = p * n_fragments + f;
            let offset = offset_sd * uniform_to_sd * noise(5_000_000 + ion);
            log2_values.push(
                (0..n_runs)
                    .map(|run| {
                        let cell = ion * n_runs + run;
                        if noise(6_000_000 + cell) < -0.35 {
                            return f64::NAN;
                        }
                        let interference = if noise(7_000_000 + cell) > 0.45 {
                            3.0 + 4.0 * noise(8_000_000 + cell)
                        } else {
                            0.0
                        };
                        q[run]
                            + offset
                            + interference
                            + noise_sd * uniform_to_sd * noise(9_000_000 + cell)
                    })
                    .collect(),
            );
            precursor.push(p);
        }
    }
    (log2_values, precursor, true_q)
}

fn profile_error(quantity: &[f64], true_q: &[Vec<f64>]) -> f64 {
    let (n_precursors, n_runs) = (true_q.len(), true_q[0].len());
    let (mut error, mut count) = (0.0, 0usize);
    for (p, q) in true_q.iter().enumerate() {
        let est = quantities(quantity, p, n_runs, n_precursors);
        let defined: Vec<usize> = (0..n_runs).filter(|&run| est[run].is_finite()).collect();
        let est_mean = defined.iter().map(|&run| est[run]).sum::<f64>() / defined.len() as f64;
        let true_mean = defined.iter().map(|&run| q[run]).sum::<f64>() / defined.len() as f64;
        for &run in &defined {
            error += ((est[run] - est_mean) - (q[run] - true_mean)).abs();
            count += 1;
        }
    }
    error / count as f64
}

#[test]
fn test_residual_scale_excludes_the_library_offsets() {
    // Given the same noise, interference and missing cells, once without and once with
    // library offsets ten times larger than the noise
    let fit_scale = |offset_sd: f64| {
        let (log2_values, precursor, true_q) = simulate_offsets_and_noise(offset_sd, 0.1);
        fit(
            &run_major(&log2_values),
            &vec![1.0; precursor.len()],
            &precursor,
            6,
            true_q.len(),
            &FitConfig::DEFAULT,
        )
        .parameters
        .sigma
    };

    // When
    let (without_offsets, with_offsets) = (fit_scale(0.0), fit_scale(1.0));

    // Then the offsets do not count as noise
    assert!(
        with_offsets < 1.5 * without_offsets,
        "residual scale {with_offsets} with offsets, {without_offsets} without"
    );
}

#[test]
fn test_offsets_are_removed_with_unequal_library_shares() {
    // Given large library offsets, interference and missing cells, and unequal library shares
    let (log2_values, precursor, true_q) = simulate_offsets_and_noise(1.0, 0.1);
    let library: Vec<f32> = (0..precursor.len())
        .map(|ion| 0.05 + 0.95 * (noise(10_000_000 + ion) as f32 + 0.5))
        .collect();
    let library_sum_per_precursor: Vec<f64> = (0..true_q.len())
        .map(|p| {
            library[p * 10..(p + 1) * 10]
                .iter()
                .map(|&l| l as f64)
                .sum()
        })
        .collect();
    let scaled: Vec<Vec<f64>> = log2_values
        .iter()
        .enumerate()
        .map(|(ion, values)| {
            let share = library[ion] as f64 / library_sum_per_precursor[precursor[ion]];
            values.iter().map(|v| v + share.log2()).collect()
        })
        .collect();

    // When
    let result = fit(
        &run_major(&scaled),
        &library,
        &precursor,
        6,
        true_q.len(),
        &FitConfig::DEFAULT,
    );

    // Then the profiles are close to the truth: a missing cell of a fragment with a large
    // offset must not move its precursor's quantity in that run
    let error = profile_error(&result.quantity, &true_q);
    assert!(error < 0.07, "profile error {error}");
}

#[test]
fn test_variance_weights_follow_the_noise_of_the_cell_intensity() {
    // Given precursors over a wide abundance range with unequal library shares, whose cells
    // are noisy below an expected log2 intensity of 12 and precise above, whatever their share
    let (n_precursors, n_fragments, n_runs) = (1000, 10, 6);
    let true_q: Vec<Vec<f64>> = (0..n_precursors)
        .map(|p| {
            let level = 16.0 + 8.0 * noise(11_000_000 + p);
            (0..n_runs)
                .map(|run| level + 2.0 * noise(12_000_000 + p * n_runs + run))
                .collect()
        })
        .collect();
    let library: Vec<f32> = (0..n_precursors * n_fragments)
        .map(|ion| 2f32.powf(-6.0 * (noise(13_000_000 + ion) as f32 + 0.5)))
        .collect();
    let precursor: Vec<usize> = (0..n_precursors * n_fragments)
        .map(|ion| ion / n_fragments)
        .collect();
    let log2_values: Vec<Vec<f64>> = (0..n_precursors * n_fragments)
        .map(|ion| {
            let p = precursor[ion];
            let library_sum: f64 = library[p * n_fragments..(p + 1) * n_fragments]
                .iter()
                .map(|&l| l as f64)
                .sum();
            let log_share = (library[ion] as f64 / library_sum).log2();
            (0..n_runs)
                .map(|run| {
                    let expected = log_share + true_q[p][run];
                    let sd = if expected < 12.0 { 1.0 } else { 0.05 };
                    expected + sd * 12f64.sqrt() * noise(14_000_000 + ion * n_runs + run)
                })
                .collect()
        })
        .collect();
    let error = |max_sweeps| {
        let result = fit(
            &run_major(&log2_values),
            &library,
            &precursor,
            n_runs,
            n_precursors,
            &FitConfig {
                max_sweeps,
                ..FitConfig::DEFAULT
            },
        );
        profile_error(&result.quantity, &true_q)
    };

    // When
    let (first_sweep, fitted) = (error(0), error(FitConfig::DEFAULT.max_sweeps));

    // Then the noisy cells are downweighted below the library-share weights of the first sweep
    assert!(
        fitted < 0.85 * first_sweep,
        "profile error {fitted} fitted, {first_sweep} after the first sweep"
    );
}
