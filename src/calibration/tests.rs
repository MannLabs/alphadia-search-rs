//! Unit tests for the LOESS calibration port.

use crate::calibration::estimator::CalibrationEstimator;
use crate::calibration::loess::{median, percentile, LoessRegression};

#[test]
fn test_percentile_linear() {
    let a: Vec<f32> = (0..=10).map(|i| i as f32).collect();
    assert!((percentile(&a, 0.0) - 0.0).abs() < 1e-5);
    assert!((percentile(&a, 100.0) - 10.0).abs() < 1e-5);
    assert!((percentile(&a, 50.0) - 5.0).abs() < 1e-5);
    // 25th percentile of 0..10 with linear interpolation = 2.5
    assert!((percentile(&a, 25.0) - 2.5).abs() < 1e-5);
}

#[test]
fn test_median() {
    assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-5);
    assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-5);
}

#[test]
fn test_fit_predict_linear() {
    // Perfectly linear data: y = 2x + 1. LOESS should reconstruct it closely.
    let x: Vec<f32> = (0..=100).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let mut model = LoessRegression::new(2, 2.0, 2);
    model.fit(&x, &y).expect("fit should succeed");

    let test_x: Vec<f32> = vec![10.0, 25.0, 50.0, 75.0, 90.0];
    let pred = model.predict(&test_x);
    for (xi, pi) in test_x.iter().zip(pred.iter()) {
        let expected = 2.0 * xi + 1.0;
        assert!(
            (pi - expected).abs() < 0.5,
            "x={xi}: predicted {pi}, expected {expected}"
        );
    }
}

#[test]
fn test_fit_predict_quadratic() {
    // y = 0.01 x^2 - x + 5, degree-2 kernels should follow it well.
    let x: Vec<f32> = (0..=200).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 0.01 * xi * xi - xi + 5.0).collect();

    let mut model = LoessRegression::new(6, 2.0, 2);
    model.fit(&x, &y).expect("fit should succeed");

    let test_x: Vec<f32> = vec![20.0, 60.0, 100.0, 140.0, 180.0];
    let pred = model.predict(&test_x);
    for (xi, pi) in test_x.iter().zip(pred.iter()) {
        let expected = 0.01 * xi * xi - xi + 5.0;
        assert!(
            (pi - expected).abs() < 1.0,
            "x={xi}: predicted {pi}, expected {expected}"
        );
    }
}

/// Deterministic pseudo-noise in `[-1, 1]` (no RNG, reproducible across runs).
fn pseudo_noise(i: usize) -> f32 {
    let s = (i as f32 * 12.9898).sin() * 43758.5;
    (s - s.floor()) * 2.0 - 1.0
}

#[test]
fn test_loess_recovers_smooth_ground_truth() {
    // Ground truth: a smooth non-linear function. LOESS should recover it from
    // noisy samples by averaging the noise out.
    let truth = |x: f32| (x * 0.5).sin() * 3.0 + 0.05 * x * x;

    let n = 2000;
    let x: Vec<f32> = (0..n).map(|i| 10.0 * i as f32 / (n - 1) as f32).collect(); // 0..10
    let noise_amp = 0.3;
    let y: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| truth(xi) + noise_amp * pseudo_noise(i))
        .collect();

    let mut model = LoessRegression::new(6, 2.0, 2);
    model.fit(&x, &y).expect("fit should succeed");
    let pred = model.predict(&x);

    // The fitted curve should track the ground truth well within the noise level.
    let errors: Vec<f32> = x
        .iter()
        .zip(&pred)
        .map(|(&xi, &p)| (p - truth(xi)).abs())
        .collect();
    let med = median(&errors);
    assert!(
        med < 0.1,
        "median error vs ground truth = {med} (noise amp {noise_amp})"
    );
}

#[test]
fn test_estimator_recovers_ppm_drift_ground_truth() {
    // Ground truth: a smoothly varying m/z drift (4 .. 12 ppm across the range).
    // The estimator should recover it from noisy observed values.
    let true_ppm = |mz: f32| 8e-6 - 4e-6 * (mz - 700.0) / 500.0;

    let n = 3000;
    let library: Vec<f32> = (0..n)
        .map(|i| 200.0 + 1000.0 * i as f32 / (n - 1) as f32) // 200..1200
        .collect();
    let observed: Vec<f32> = library
        .iter()
        .enumerate()
        .map(|(i, &mz)| mz * (1.0 + true_ppm(mz)) + 1e-3 * pseudo_noise(i))
        .collect();

    // fragment m/z config: 2 kernels, ppm transform
    let mut est = CalibrationEstimator::new(2, Some(1e6));
    est.fit(&library, &observed).expect("fit should succeed");
    let calibrated = est.predict(&library).expect("predict should succeed");

    // Recovered drift (ppm) should match the ground-truth drift within ~1 ppm.
    let errors: Vec<f32> = library
        .iter()
        .zip(&calibrated)
        .map(|(&mz, &c)| ((c - mz) / mz * 1e6 - true_ppm(mz) * 1e6).abs())
        .collect();
    let med = median(&errors);
    assert!(med < 1.0, "median ppm error vs ground truth = {med}");
}

#[test]
fn test_fit_too_few_points() {
    let mut model = LoessRegression::new(2, 2.0, 2);
    assert!(model.fit(&[1.0], &[1.0]).is_err());
}

#[test]
fn test_fit_requires_overlapping_kernels() {
    // Non-overlapping kernels leave gaps between their supports. Fitting a bimodal
    // distribution and predicting inside the gap used to yield NaN predictions.
    let mut x: Vec<f32> = (0..450).map(|i| 100.0 + i as f32 * 0.1).collect();
    x.extend((0..450).map(|i| 900.0 + i as f32 * 0.1));
    let y: Vec<f32> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

    assert!(LoessRegression::new(2, 1.0, 2).fit(&x, &y).is_err());

    // The production kernel size overlaps, so the gap stays covered.
    let mut model = LoessRegression::new(2, 2.0, 2);
    model.fit(&x, &y).expect("fit should succeed");
    assert!(model
        .predict(&[200.0, 500.0, 800.0])
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn test_fit_rejects_zero_kernels() {
    // Zero kernels used to divide by zero in kernel placement.
    let x: Vec<f32> = (0..=100).map(|i| i as f32).collect();
    let mut model = LoessRegression::new(0, 2.0, 2);
    assert!(model.fit(&x, &x).is_err());
}

#[test]
fn test_fit_rejects_shorter_target() {
    let mut est = CalibrationEstimator::new(2, None);
    assert!(est.fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    assert!(!est.is_fitted());
}

#[test]
fn test_fit_rejects_longer_target() {
    // The dangerous direction: a longer target used to fit successfully on the
    // truncated prefix and report success.
    let mut est = CalibrationEstimator::new(2, None);
    assert!(est.fit(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
    assert!(!est.is_fitted());
}

#[test]
fn test_unfitted_estimator_errors_instead_of_panicking() {
    let est = CalibrationEstimator::new(2, None);
    let x = [1.0, 2.0, 3.0];
    assert!(est.predict(&x).is_err());
    assert!(est.deviation(&x, &x).is_err());
    // `ci` deliberately reports 0.0 when unfitted (the Python wrapper relies on it).
    assert_eq!(est.ci(&x, &x, 0.95), Ok(0.0));
}

#[test]
fn test_fitted_estimator_rejects_length_mismatch() {
    let library: Vec<f32> = (200..=1200).map(|i| i as f32).collect();
    let observed: Vec<f32> = library.iter().map(|&mz| mz * (1.0 + 5e-6)).collect();
    let mut est = CalibrationEstimator::new(2, Some(1e6));
    est.fit(&library, &observed).expect("fit should succeed");

    assert!(est.deviation(&library, &observed[..500]).is_err());
    assert!(est.ci(&library, &observed[..500], 0.95).is_err());
}

#[test]
fn test_ci_rejects_out_of_range() {
    let library: Vec<f32> = (200..=1200).map(|i| i as f32).collect();
    let observed: Vec<f32> = library.iter().map(|&mz| mz * (1.0 + 5e-6)).collect();
    let mut est = CalibrationEstimator::new(2, Some(1e6));
    est.fit(&library, &observed).expect("fit should succeed");

    assert!(est.ci(&library, &observed, -0.1).is_err());
    assert!(est.ci(&library, &observed, 1.5).is_err());
    assert!(est.ci(&library, &observed, 0.95).is_ok());
}

#[test]
fn test_estimator_ppm_transform() {
    // Fragment m/z uses a ppm transform. Build data where observed = library
    // shifted by a constant ppm, and check the estimator is fitted and metrics
    // are finite.
    let library: Vec<f32> = (200..=1200).map(|i| i as f32).collect();
    let observed: Vec<f32> = library.iter().map(|&mz| mz * (1.0 + 5e-6)).collect();

    // fragment m/z config: 2 kernels, ppm transform
    let mut est = CalibrationEstimator::new(2, Some(1e6));
    est.fit(&library, &observed).expect("fit should succeed");
    assert!(est.is_fitted());
    assert_eq!(est.transform_deviation(), Some(1e6));

    let (bias, variance) = est.metrics().expect("metrics present");
    assert!(bias.is_finite() && variance.is_finite());

    // Calibrated values should be much closer to observed than the library was.
    let dev = est
        .deviation(&library, &observed)
        .expect("deviation should succeed");
    let calibrated = est.predict(&library).expect("predict should succeed");
    let raw_err: f32 = median(
        &observed
            .iter()
            .zip(&library)
            .map(|(o, l)| (o - l).abs())
            .collect::<Vec<_>>(),
    );
    let cal_err: f32 = median(
        &observed
            .iter()
            .zip(&calibrated)
            .map(|(o, c)| (o - c).abs())
            .collect::<Vec<_>>(),
    );
    assert!(cal_err < raw_err, "calibration should reduce error");
    // Residual (ppm) should be small; the floor is the f32 storage resolution on
    // m/z values in the hundreds (~0.1 ppm per value).
    let med_res = median(&dev.residual.iter().map(|v| v.abs()).collect::<Vec<_>>());
    assert!(med_res < 0.5, "median residual ppm = {med_res}");
}
