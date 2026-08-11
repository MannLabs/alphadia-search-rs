//! Calibration estimator: wraps a [`LoessRegression`] and models the deviation of
//! an input property (e.g. `mz_library`) from an observed target (e.g.
//! `mz_observed`). Port of `alphadia/calibration/estimator.py::CalibrationEstimator`.
//!
//! This is a thin calibration wrapper: it takes its two configuration values
//! (`n_kernels` and the deviation `transform`) directly from the caller — the
//! per-property configuration lives on the Python side, not here.

use crate::calibration::loess::{median_abs, percentiles, LoessRegression};

/// Shared LOESS hyper-parameters for all calibration estimators.
const KERNEL_SIZE: f32 = 2.0;
const POLYNOMIAL_DEGREE: usize = 2;

/// Deviation columns produced by [`CalibrationEstimator::deviation`].
pub struct Deviation {
    pub observed: Vec<f32>,
    pub calibrated: Vec<f32>,
    pub residual: Vec<f32>,
}

/// A single calibration estimator for one property.
pub struct CalibrationEstimator {
    n_kernels: usize,
    transform_deviation: Option<f32>,
    model: LoessRegression,
    is_fitted: bool,
    metrics: Option<(f32, f32)>, // (median_bias, median_variance)
}

impl CalibrationEstimator {
    /// Create an unfitted estimator.
    ///
    /// * `n_kernels` — number of LOESS kernels.
    /// * `transform_deviation` — deviation transform factor (e.g. `1e6` for ppm),
    ///   or `None` for absolute deviations.
    pub fn new(n_kernels: usize, transform_deviation: Option<f32>) -> Self {
        Self {
            n_kernels,
            transform_deviation,
            model: LoessRegression::new(n_kernels, KERNEL_SIZE, POLYNOMIAL_DEGREE),
            is_fitted: false,
            metrics: None,
        }
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    pub fn transform_deviation(&self) -> Option<f32> {
        self.transform_deviation
    }

    /// `(median_bias, median_variance)` if fitted.
    pub fn metrics(&self) -> Option<(f32, f32)> {
        self.metrics
    }

    /// Fit the underlying LOESS model and cache metrics. Leaves the estimator
    /// unfitted (and returns `Err`) on degenerate input.
    pub fn fit(&mut self, input: &[f32], target: &[f32]) -> Result<(), String> {
        validate_pair(input, target)?;

        // Calibration corrects a near-identity mapping (target ≈ input, e.g.
        // observed ≈ library m/z). Fitting the *residual* `target - input` keeps the
        // LOESS target small, so the f32 core resolves the tiny correction; the
        // identity baseline is added back in `predict`. This reparametrization is
        // exact (it does not change the fitted function) and is a calibration-domain
        // concern, deliberately kept out of the general LOESS core.
        let residual: Vec<f32> = target.iter().zip(input).map(|(t, i)| t - i).collect();

        let mut model = LoessRegression::new(self.n_kernels, KERNEL_SIZE, POLYNOMIAL_DEGREE);
        model.fit(input, &residual)?;
        self.model = model;
        self.is_fitted = true;

        let dev = self.deviation(input, target)?;
        self.metrics = Some((median_abs(&dev.calibrated), median_abs(&dev.residual)));
        Ok(())
    }

    /// Predict calibrated values from input values (baseline + fitted residual).
    ///
    /// Returns `Err` if the estimator is not fitted: the underlying model has empty
    /// parameter vectors until `fit` succeeds, and using them would panic.
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.require_fitted()?;
        let fitted_residual = self.model.predict(input);
        Ok(input
            .iter()
            .zip(&fitted_residual)
            .map(|(i, r)| i + r)
            .collect())
    }

    /// Compute observed / calibrated / residual deviations (applying the ppm
    /// transform where configured), for plotting and metrics.
    pub fn deviation(&self, input: &[f32], target: &[f32]) -> Result<Deviation, String> {
        validate_pair(input, target)?;
        let calibrated_values = self.predict(input)?;
        let transform = self.transform_deviation;

        let mut observed = Vec::with_capacity(input.len());
        let mut calibrated = Vec::with_capacity(input.len());
        let mut residual = Vec::with_capacity(input.len());

        for ((&uncal, &tgt), &cal_value) in input.iter().zip(target).zip(&calibrated_values) {
            let mut obs = tgt - uncal;
            let mut cal = cal_value - uncal;
            if let Some(t) = transform {
                obs = obs / uncal * t;
                cal = cal / uncal * t;
            }
            observed.push(obs);
            calibrated.push(cal);
            residual.push(obs - cal);
        }

        Ok(Deviation {
            observed,
            calibrated,
            residual,
        })
    }

    /// Residual deviation at the given confidence interval (e.g. `0.95`), matching
    /// the Python `ci` method: mean of the absolute central-interval endpoints.
    ///
    /// Returns `Ok(0.0)` when unfitted (the Python wrapper relies on this).
    pub fn ci(&self, input: &[f32], target: &[f32], ci: f32) -> Result<f32, String> {
        if !(0.0..=1.0).contains(&ci) {
            return Err(format!("confidence interval must be in [0, 1], got {ci}"));
        }
        if !self.is_fitted {
            return Ok(0.0);
        }
        if input.is_empty() {
            return Err("cannot compute a confidence interval on empty input".to_string());
        }
        let dev = self.deviation(input, target)?;
        let lo = 100.0 * (1.0 - ci) / 2.0;
        let hi = 100.0 * (1.0 + ci) / 2.0;
        let bounds = percentiles(&dev.residual, &[lo, hi]);
        Ok((bounds[0].abs() + bounds[1].abs()) / 2.0)
    }

    fn require_fitted(&self) -> Result<(), String> {
        if self.is_fitted {
            Ok(())
        } else {
            Err("estimator is not fitted".to_string())
        }
    }
}

/// Reject mismatched input/target lengths at the public entry points.
///
/// Without this a shorter `target` panics on indexing, and — worse — a longer one
/// silently fits on the truncated prefix and reports success.
fn validate_pair(input: &[f32], target: &[f32]) -> Result<(), String> {
    if input.len() != target.len() {
        return Err(format!(
            "input and target must have the same length, got {} and {}",
            input.len(),
            target.len()
        ));
    }
    Ok(())
}
