use super::network::{self, TrainConfig};
use super::ops::Backend;
use super::quantile::{self, norm_ppf, test_access};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn heavy_tailed(rng: &mut StdRng, n_rows: usize, n_features: usize) -> Vec<f32> {
    (0..n_rows * n_features)
        .map(|i| {
            let v: f32 = rng.random_range(-3.0..3.0);
            if i % n_features == 0 {
                v.exp()
            } else if i % n_features == 1 {
                v.round()
            } else {
                v
            }
        })
        .collect()
}

#[test]
fn test_norm_ppf_known_values() {
    // Given probabilities with known normal quantiles
    // When / Then
    assert!((norm_ppf(0.5)).abs() < 1e-15);
    assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-12);
    assert!((norm_ppf(1e-7) + 5.199_337_582_605_575).abs() < 1e-9);
    assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
    assert_eq!(norm_ppf(1.0), f64::INFINITY);
}

#[test]
fn test_quantile_fit_matches_numpy_linear_percentiles() {
    // Given the values 0..=100 in one feature
    let x: Vec<f32> = (0..=100).map(|v| v as f32).collect();

    // When fitting 5 quantiles
    let q = quantile::fit(&x, 101, 1, 5, None, 0).unwrap();

    // Then they are the 0/25/50/75/100th percentiles
    assert_eq!(q, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
}

#[test]
fn test_quantile_transform_matches_exact_scores() {
    // Given heavy-tailed, discrete and uniform features
    let mut rng = StdRng::seed_from_u64(1);
    let (n_rows, n_features, n_quantiles) = (20_000, 3, 1001);
    let x = heavy_tailed(&mut rng, n_rows, n_features);

    // When transforming with the lookup and with sklearn's exact formula
    let q = quantile::fit(&x, n_rows, n_features, n_quantiles, None, 0).unwrap();
    let fast = quantile::transform(&x, n_rows, n_features, &q, n_quantiles, false).unwrap();

    // Then the scores agree closely, and the extremes hit sklearn's clip values
    for f in 0..n_features {
        let column: Vec<f32> = (0..n_rows).map(|r| x[r * n_features + f]).collect();
        let exact = test_access::exact_scores(&column, &q[f * n_quantiles..(f + 1) * n_quantiles]);
        let mut sum = 0.0f64;
        for r in 0..n_rows {
            let diff = (fast[r * n_features + f] - exact[r]).abs();
            assert!(
                diff < 2e-2,
                "feature {f} row {r}: {} vs {}",
                fast[r * n_features + f],
                exact[r]
            );
            sum += diff as f64;
        }
        assert!(sum / (n_rows as f64) < 1e-3);
    }
    let max = fast.iter().cloned().fold(f32::MIN, f32::max);
    assert!((max - 5.199_337_6).abs() < 1e-4);
}

#[test]
fn test_radix_sort_orders_like_total_cmp() {
    // Given random floats of all magnitudes, with ties, signed zeros, infinities and NaNs
    let mut rng = StdRng::seed_from_u64(3);
    let mut x: Vec<f32> = (0..50_000)
        .map(|i| match i % 5 {
            0 => f32::from_bits(rng.random()),
            1 => rng.random_range(-3.0f32..3.0).round(),
            _ => rng.random_range(-1e3f32..1e3),
        })
        .collect();
    x.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
    ]);

    // When sorting with the radix sort of the quantile fit
    let sorted = test_access::radix_sort(&x);

    // Then the bits are those of a total_cmp sort
    let mut expected = x.clone();
    expected.sort_by(f32::total_cmp);
    let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&sorted), bits(&expected));
}

#[test]
#[serial_test::serial]
fn test_quantile_transform_is_identical_on_every_simd_backend() {
    // Given quantiles of heavy-tailed, discrete and uniform features fitted on a subsample, and
    // rows with values outside the fitted range, at the knots and NaN
    let mut rng = StdRng::seed_from_u64(4);
    let (n_rows, n_features, n_quantiles) = (10_007, 3, 101);
    let mut x = heavy_tailed(&mut rng, n_rows, n_features);
    let q = quantile::fit(&x, n_rows, n_features, n_quantiles, Some(500), 0).unwrap();
    x[..n_features].copy_from_slice(&[f32::NAN, 1e9, -1e9]);
    x[n_features..2 * n_features].copy_from_slice(&[q[0] as f32, q[150] as f32, q[252] as f32]);

    for mid_rank_bounds in [false, true] {
        // When transforming with every SIMD backend
        let transform = |backend: &str| {
            crate::simd::set_backend(backend).ok()?;
            quantile::transform(&x, n_rows, n_features, &q, n_quantiles, mid_rank_bounds).ok()
        };
        let reference = transform("scalar").unwrap();
        for backend in ["avx2", "avx512", "neon"] {
            // Then the scores are bit for bit those of the scalar code
            if let Some(scores) = transform(backend) {
                let same = scores
                    .iter()
                    .zip(&reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{backend}, mid_rank_bounds {mid_rank_bounds}");
            }
        }
    }
    crate::simd::clear_backend();
}

#[test]
fn test_quantile_transform_keeps_nan_and_rejects_shape_mismatch() {
    // Given fitted quantiles
    let x: Vec<f32> = (0..1000).map(|v| v as f32).collect();
    let q = quantile::fit(&x, 1000, 1, 11, None, 0).unwrap();

    // When transforming a NaN and a matrix of the wrong width
    let out = quantile::transform(&[f32::NAN], 1, 1, &q, 11, false).unwrap();
    let wrong = quantile::transform(&[0.0, 0.0], 1, 2, &q, 11, false);

    // Then the NaN stays NaN and the mismatch is an error
    assert!(out[0].is_nan());
    assert!(wrong.is_err());
}

#[test]
fn test_quantile_subsample_is_close_to_full_fit() {
    // Given many rows from a uniform distribution
    let mut rng = StdRng::seed_from_u64(2);
    let x: Vec<f32> = (0..200_000).map(|_| rng.random_range(0.0..1.0)).collect();

    // When fitting on all rows and on a subsample
    let full = quantile::fit(&x, 200_000, 1, 101, None, 0).unwrap();
    let sub = quantile::fit(&x, 200_000, 1, 101, Some(20_000), 0).unwrap();

    // Then the inner quantiles agree
    for (a, b) in full[1..100].iter().zip(&sub[1..100]) {
        assert!((a - b).abs() < 0.02);
    }
}

fn separable(rng: &mut StdRng, n_rows: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let y: Vec<f32> = (0..n_rows).map(|r| (r % 2) as f32).collect();
    let x = (0..n_rows * n_features)
        .map(|i| {
            let shift = if y[i / n_features] > 0.5 && i % n_features < 4 {
                1.5
            } else {
                0.0
            };
            rng.random_range(-1.0f32..1.0) + shift
        })
        .collect();
    (x, y)
}

fn config(backend: Backend) -> TrainConfig {
    TrainConfig {
        hidden_layers: vec![32, 16],
        epochs: 3,
        batch_size: 256,
        learning_rate: 0.005,
        weight_decay: 1e-5,
        seed: 7,
        backend,
        cosine_schedule: false,
        average_last_epoch: false,
        input_batch_norm: false,
    }
}

fn backends() -> Vec<Backend> {
    Backend::available_names()
        .into_iter()
        .map(|name| Backend::from_name(name).unwrap())
        .collect()
}

#[test]
fn test_network_learns_a_separable_problem_on_every_backend() {
    // Given two classes shifted apart in four features
    let mut rng = StdRng::seed_from_u64(3);
    let (n_rows, n_features) = (8192, 12);
    let (x, y) = separable(&mut rng, n_rows, n_features);

    for backend in backends() {
        // When training and predicting
        let params = network::train(&x, n_rows, n_features, &y, &config(backend), None).unwrap();
        let logits =
            network::predict(&x, n_rows, n_features, &[32, 16], &params, backend, false).unwrap();

        // Then nearly every row is classified correctly
        let correct = logits
            .iter()
            .zip(&y)
            .filter(|(l, y)| (**l > 0.0) == (**y > 0.5))
            .count();
        assert!(
            correct as f64 / n_rows as f64 > 0.95,
            "{backend:?}: {correct} correct"
        );
    }
}

#[test]
fn test_backends_predict_the_same_logits() {
    // Given one trained parameter vector
    let mut rng = StdRng::seed_from_u64(4);
    let (n_rows, n_features) = (1001, 12);
    let (x, y) = separable(&mut rng, n_rows, n_features);
    let params = network::train(&x, n_rows, n_features, &y, &config(Backend::Simd), None).unwrap();

    // When predicting with every backend
    let reference = network::predict(
        &x,
        n_rows,
        n_features,
        &[32, 16],
        &params,
        Backend::Simd,
        false,
    )
    .unwrap();
    for backend in backends() {
        let logits =
            network::predict(&x, n_rows, n_features, &[32, 16], &params, backend, false).unwrap();

        // Then the logits agree
        for (a, b) in logits.iter().zip(&reference) {
            assert!(
                (a - b).abs() < 1e-3 * (1.0 + b.abs()),
                "{backend:?}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_warm_start_continues_from_given_parameters() {
    // Given parameters from a first fit
    let mut rng = StdRng::seed_from_u64(5);
    let (n_rows, n_features) = (2048, 12);
    let (x, y) = separable(&mut rng, n_rows, n_features);
    let first = network::train(&x, n_rows, n_features, &y, &config(Backend::Simd), None).unwrap();

    // When training again from them, and with parameters of the wrong size
    let second = network::train(
        &x,
        n_rows,
        n_features,
        &y,
        &config(Backend::Simd),
        Some(first.clone()),
    )
    .unwrap();
    let wrong = network::train(
        &x,
        n_rows,
        n_features,
        &y,
        &config(Backend::Simd),
        Some(vec![0.0; 3]),
    );

    // Then the second fit moves the parameters, and the wrong size is an error
    assert_eq!(second.len(), first.len());
    assert_ne!(second, first);
    assert!(wrong.is_err());
}

#[test]
fn test_training_with_fewer_rows_than_a_batch() {
    // Given fewer rows than the batch size
    let mut rng = StdRng::seed_from_u64(6);
    let (x, y) = separable(&mut rng, 50, 12);

    // When training
    let params = network::train(&x, 50, 12, &y, &config(Backend::Simd), None);

    // Then it trains on one batch of all rows
    assert!(params.is_ok());
}

#[test]
fn test_cosine_schedule_and_averaging_change_the_fit_and_still_learn() {
    // Given a separable problem and the default (constant, last parameters) fit
    let mut rng = StdRng::seed_from_u64(8);
    let (n_rows, n_features) = (4096, 12);
    let (x, y) = separable(&mut rng, n_rows, n_features);
    let plain = network::train(&x, n_rows, n_features, &y, &config(Backend::Simd), None).unwrap();

    for (cosine_schedule, average_last_epoch) in [(true, false), (false, true), (true, true)] {
        // When training with the cosine schedule and/or the last-epoch average
        let smoothed_config = TrainConfig {
            cosine_schedule,
            average_last_epoch,
            ..config(Backend::Simd)
        };
        let params = network::train(&x, n_rows, n_features, &y, &smoothed_config, None).unwrap();
        let logits = network::predict(
            &x,
            n_rows,
            n_features,
            &[32, 16],
            &params,
            Backend::Simd,
            false,
        )
        .unwrap();

        // Then the parameters differ from the plain fit and the classes still separate
        assert_ne!(params, plain);
        let correct = logits
            .iter()
            .zip(&y)
            .filter(|(l, y)| (**l > 0.0) == (**y > 0.5))
            .count();
        assert!(correct as f64 / n_rows as f64 > 0.95, "{correct} correct");
    }
}

#[test]
fn test_average_of_one_minibatch_is_the_last_parameters() {
    // Given one epoch of one minibatch
    let mut rng = StdRng::seed_from_u64(9);
    let (x, y) = separable(&mut rng, 128, 12);
    let one_step = TrainConfig {
        epochs: 1,
        batch_size: 128,
        ..config(Backend::Simd)
    };

    // When averaging the last epoch
    let plain = network::train(&x, 128, 12, &y, &one_step, None).unwrap();
    let averaged = network::train(
        &x,
        128,
        12,
        &y,
        &TrainConfig {
            average_last_epoch: true,
            ..one_step
        },
        None,
    )
    .unwrap();

    // Then the mean of a single step is that step
    assert_eq!(plain, averaged);
}

fn batch_norm_config() -> TrainConfig {
    TrainConfig {
        input_batch_norm: true,
        ..config(Backend::Simd)
    }
}

fn accuracy(logits: &[f32], y: &[f32]) -> f64 {
    let correct = logits
        .iter()
        .zip(y)
        .filter(|(l, y)| (**l > 0.0) == (**y > 0.5))
        .count();
    correct as f64 / y.len() as f64
}

#[test]
fn test_input_batch_norm_learns_badly_scaled_features() {
    // Given separable features moved far off zero and scaled up, as tied quantile scores can be
    let mut rng = StdRng::seed_from_u64(10);
    let (n_rows, n_features) = (4096, 12);
    let (mut x, y) = separable(&mut rng, n_rows, n_features);
    for (i, v) in x.iter_mut().enumerate() {
        if i % n_features < 6 {
            *v = *v * 3.0 - 5.0;
        }
    }

    // When training with the input batch norm and predicting with its running statistics
    let params = network::train(&x, n_rows, n_features, &y, &batch_norm_config(), None).unwrap();
    let logits = network::predict(
        &x,
        n_rows,
        n_features,
        &[32, 16],
        &params,
        Backend::Simd,
        true,
    )
    .unwrap();

    // Then the classes separate
    assert!(accuracy(&logits, &y) > 0.95, "{}", accuracy(&logits, &y));
}

#[test]
fn test_input_batch_norm_running_statistics_follow_the_data() {
    // Given features with mean -5 and variance 9 (uniform(-1, 1) * 3 has variance 3)
    let mut rng = StdRng::seed_from_u64(11);
    let (n_rows, n_features) = (4096, 3);
    let x: Vec<f32> = (0..n_rows * n_features)
        .map(|_| rng.random_range(-1.0f32..1.0) * 3.0 - 5.0)
        .collect();
    let y: Vec<f32> = (0..n_rows).map(|r| (r % 2) as f32).collect();

    // When training
    let params = network::train(&x, n_rows, n_features, &y, &batch_norm_config(), None).unwrap();

    // Then the last 4 * n_features parameters are scale, shift, running mean and running variance,
    // and the running statistics are close to the data's
    let size_without = network::Layout::new(n_features, &[32, 16], false)
        .unwrap()
        .size;
    assert_eq!(params.len(), size_without + 4 * n_features);
    let running_mean = &params[size_without + 2 * n_features..size_without + 3 * n_features];
    let running_var = &params[size_without + 3 * n_features..];
    for p in 0..n_features {
        assert!(
            (running_mean[p] + 5.0).abs() < 0.3,
            "mean {}",
            running_mean[p]
        );
        assert!((running_var[p] - 3.0).abs() < 0.5, "var {}", running_var[p]);
    }
}

#[test]
fn test_predict_rejects_parameters_without_the_batch_norm_block() {
    // Given parameters trained without the input batch norm
    let mut rng = StdRng::seed_from_u64(12);
    let (x, y) = separable(&mut rng, 512, 12);
    let params = network::train(&x, 512, 12, &y, &config(Backend::Simd), None).unwrap();

    // When predicting with the batch norm
    let result = network::predict(&x, 512, 12, &[32, 16], &params, Backend::Simd, true);

    // Then the parameter count does not match
    assert!(result.is_err());
}

#[test]
fn test_mid_rank_bounds_maps_ties_at_the_lowest_quantile_to_their_middle() {
    // Given a feature that is 0 in 90 % of rows, then rising values
    let n_rows = 1000;
    let x: Vec<f32> = (0..n_rows)
        .map(|r| if r < 900 { 0.0 } else { (r - 899) as f32 })
        .collect();
    let q = quantile::fit(&x, n_rows, 1, 101, None, 0).unwrap();

    // When transforming with and without mid-ranked bounds
    let clamped = quantile::transform(&x, n_rows, 1, &q, 101, false).unwrap();
    let mid = quantile::transform(&x, n_rows, 1, &q, 101, true).unwrap();

    // Then sklearn's rule clamps the zeros to the clip value, the mid rank puts them at the
    // middle of their group (quantile 0.45, normal score about -0.13)
    assert!(clamped[0] < -5.0, "{}", clamped[0]);
    assert!((mid[0] - norm_ppf(0.45) as f32).abs() < 0.02, "{}", mid[0]);
    // and values above the tie group are unchanged
    for r in 905..995 {
        assert!(
            (clamped[r] - mid[r]).abs() < 1e-6,
            "row {r}: {} vs {}",
            clamped[r],
            mid[r]
        );
    }
}

#[test]
fn test_mid_rank_bounds_clips_values_beyond_the_fitted_range() {
    // Given quantiles of distinct values 1..=100
    let x: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let q = quantile::fit(&x, 100, 1, 101, None, 0).unwrap();

    // When transforming values below and above the training range
    let out = quantile::transform(&[-50.0, 500.0], 2, 1, &q, 101, true).unwrap();

    // Then they map to the clip values
    assert!(out[0] < -5.0 && out[1] > 5.0, "{out:?}");
}

#[test]
fn test_mid_rank_bounds_maps_ties_at_the_highest_quantile_to_their_middle() {
    // Given a feature at its maximum 6 in the top 20 % of rows
    let n_rows = 1000;
    let x: Vec<f32> = (0..n_rows)
        .map(|r| if r >= 800 { 6.0 } else { r as f32 / 200.0 })
        .collect();
    let q = quantile::fit(&x, n_rows, 1, 101, None, 0).unwrap();

    // When transforming with mid-ranked bounds
    let mid = quantile::transform(&[6.0], 1, 1, &q, 101, true).unwrap();

    // Then the maximum maps to the middle of its group as np.interp ranks it: the group starts
    // between knots 80 and 81 (rank 0.81 forward-backward mean with 1.0 gives 0.905)
    assert!((mid[0] - norm_ppf(0.905) as f32).abs() < 1e-4, "{}", mid[0]);
}

#[test]
fn test_batch_norm_folded_into_the_first_layer_predicts_like_normalized_input() {
    // Given trained batch norm parameters, with features far off zero relative to their spread
    let mut rng = StdRng::seed_from_u64(13);
    let (n_rows, n_features) = (2000, 12);
    let (mut x, y) = separable(&mut rng, n_rows, n_features);
    for (i, v) in x.iter_mut().enumerate() {
        if i % n_features < 6 {
            *v = *v * 0.1 + 100.0;
        }
    }
    let params = network::train(&x, n_rows, n_features, &y, &batch_norm_config(), None).unwrap();

    // When predicting with the batch norm, and without it on input normalized by hand
    let logits = network::predict(
        &x,
        n_rows,
        n_features,
        &[32, 16],
        &params,
        Backend::Simd,
        true,
    )
    .unwrap();
    let plain = network::Layout::new(n_features, &[32, 16], false)
        .unwrap()
        .size;
    let bn = |i: usize| &params[plain + i * n_features..plain + (i + 1) * n_features];
    let normalized: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let p = i % n_features;
            bn(0)[p] * (v - bn(2)[p]) / (bn(3)[p] + 1e-5).sqrt() + bn(1)[p]
        })
        .collect();
    let expected = network::predict(
        &normalized,
        n_rows,
        n_features,
        &[32, 16],
        &params[..plain],
        Backend::Simd,
        false,
    )
    .unwrap();

    // Then both agree
    for (a, e) in logits.iter().zip(&expected) {
        assert!((a - e).abs() < 1e-3 * (1.0 + e.abs()), "{a} vs {e}");
    }
}
