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
    let fast = quantile::transform(&x, n_rows, n_features, &q, n_quantiles).unwrap();

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
fn test_quantile_transform_keeps_nan_and_rejects_shape_mismatch() {
    // Given fitted quantiles
    let x: Vec<f32> = (0..1000).map(|v| v as f32).collect();
    let q = quantile::fit(&x, 1000, 1, 11, None, 0).unwrap();

    // When transforming a NaN and a matrix of the wrong width
    let out = quantile::transform(&[f32::NAN], 1, 1, &q, 11).unwrap();
    let wrong = quantile::transform(&[0.0, 0.0], 1, 2, &q, 11);

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
        let logits = network::predict(&x, n_rows, n_features, &[32, 16], &params, backend).unwrap();

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
    let reference =
        network::predict(&x, n_rows, n_features, &[32, 16], &params, Backend::Simd).unwrap();
    for backend in backends() {
        let logits = network::predict(&x, n_rows, n_features, &[32, 16], &params, backend).unwrap();

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
        let logits =
            network::predict(&x, n_rows, n_features, &[32, 16], &params, Backend::Simd).unwrap();

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
