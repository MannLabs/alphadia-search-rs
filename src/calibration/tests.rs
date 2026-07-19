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

#[test]
fn test_fit_too_few_points() {
    let mut model = LoessRegression::new(2, 2.0, 2);
    assert!(model.fit(&[1.0], &[1.0]).is_err());
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

    let (acc, prec) = est.metrics().expect("metrics present");
    assert!(acc.is_finite() && prec.is_finite());

    // Calibrated values should be much closer to observed than the library was.
    let dev = est.deviation(&library, &observed);
    let calibrated = est.predict(&library);
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
