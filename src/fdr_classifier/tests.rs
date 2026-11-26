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
    // When the normal quantile function is evaluated at them
    // Then it gives the known quantiles
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
    let (n_rows, n_features, n_quantiles) = (5_000, 3, 1001);
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
fn test_radix_sort_orders_like_total_cmp() {
    // Given random floats of all magnitudes, with ties, signed zeros, infinities and NaNs
    let mut rng = StdRng::seed_from_u64(3);
    let mut x: Vec<f32> = (0..10_000)
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
fn test_quantile_fit_rejects_invalid_input() {
    // Given a NaN value, an empty matrix, too few quantiles and an empty subsample
    let x = [0.0, 1.0, f32::NAN, 3.0];
    let cases = [
        quantile::fit(&x, 4, 1, 3, None, 0),
        quantile::fit(&[], 0, 1, 3, None, 0),
        quantile::fit(&x[..2], 2, 1, 1, None, 0),
        quantile::fit(&x[..2], 2, 1, 3, Some(0), 0),
    ];

    // When / Then every fit fails
    for (i, fitted) in cases.iter().enumerate() {
        assert!(fitted.is_err(), "case {i}");
    }
}

#[test]
fn test_quantile_subsample_is_close_to_full_fit() {
    // Given many rows from a uniform distribution
    let mut rng = StdRng::seed_from_u64(2);
    let n_rows = 50_000;
    let x: Vec<f32> = (0..n_rows).map(|_| rng.random_range(0.0..1.0)).collect();

    // When fitting on all rows, on a subsample and on a "subsample" of all rows
    let full = quantile::fit(&x, n_rows, 1, 101, None, 0).unwrap();
    let sub = quantile::fit(&x, n_rows, 1, 101, Some(10_000), 0).unwrap();
    let all = quantile::fit(&x, n_rows, 1, 101, Some(n_rows), 0).unwrap();

    // Then the subsample gives different but close inner quantiles, and the subsample of all rows
    // gives the full fit
    assert_ne!(sub, full);
    for (a, b) in full[1..100].iter().zip(&sub[1..100]) {
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
    }
    assert_eq!(all, full);
}
