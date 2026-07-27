//! Calibration estimator: wraps a [`LoessRegression`] and models the deviation of
//! an input property (e.g. `mz_library`) from an observed target (e.g.
//! `mz_observed`). Port of `alphadia/calibration/estimator.py::CalibrationEstimator`.
//!
//! This is a thin calibration wrapper: it takes its two configuration values
//! (`n_kernels` and the deviation `transform`) directly from the caller — the
//! per-property configuration lives on the Python side, not here.

use crate::calibration::loess::{median, percentile, LoessRegression};

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
        // Calibration corrects a near-identity mapping (target ≈ input, e.g.
        // observed ≈ library m/z). Fitting the *residual* `target - input` keeps the
        // LOESS target small, so the f32 core resolves the tiny correction; the
        // identity baseline is added back in `predict`. This reparametrization is
        // exact (it does not change the fitted function) and is a calibration-domain
        // concern, deliberately kept out of the general LOESS core.
        let residual: Vec<f32> = (0..input.len()).map(|i| target[i] - input[i]).collect();

        let mut model = LoessRegression::new(self.n_kernels, KERNEL_SIZE, POLYNOMIAL_DEGREE);
        model.fit(input, &residual)?;
        self.model = model;
        self.is_fitted = true;

        let dev = self.deviation_with(input, target);
        let median_bias = median(&abs(&dev.calibrated));
        let median_variance = median(&abs(&dev.residual));
        self.metrics = Some((median_bias, median_variance));
        Ok(())
    }

    /// Predict calibrated values from input values (baseline + fitted residual).
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let fitted_residual = self.model.predict(input);
        (0..input.len())
            .map(|i| input[i] + fitted_residual[i])
            .collect()
    }

    /// Compute observed / calibrated / residual deviations (applying the ppm
    /// transform where configured), for plotting and metrics.
    pub fn deviation(&self, input: &[f32], target: &[f32]) -> Deviation {
        self.deviation_with(input, target)
    }

    fn deviation_with(&self, input: &[f32], target: &[f32]) -> Deviation {
        let calibrated_values = self.predict(input);
        let transform = self.transform_deviation;

        let mut observed = Vec::with_capacity(input.len());
        let mut calibrated = Vec::with_capacity(input.len());
        let mut residual = Vec::with_capacity(input.len());

        for i in 0..input.len() {
            let uncal = input[i];
            let mut obs = target[i] - uncal;
            let mut cal = calibrated_values[i] - uncal;
            if let Some(t) = transform {
                obs = obs / uncal * t;
                cal = cal / uncal * t;
            }
            observed.push(obs);
            calibrated.push(cal);
            residual.push(obs - cal);
        }

        Deviation {
            observed,
            calibrated,
            residual,
        }
    }

    /// Residual deviation at the given confidence interval (e.g. `0.95`), matching
    /// the Python `ci` method: mean of the absolute central-interval endpoints.
    pub fn ci(&self, input: &[f32], target: &[f32], ci: f32) -> f32 {
        if !self.is_fitted {
            return 0.0;
        }
        let dev = self.deviation_with(input, target);
        let lo = 100.0 * (1.0 - ci) / 2.0;
        let hi = 100.0 * (1.0 + ci) / 2.0;
        let p_lo = percentile(&dev.residual, lo);
        let p_hi = percentile(&dev.residual, hi);
        (p_lo.abs() + p_hi.abs()) / 2.0
    }
}

fn abs(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| v.abs()).collect()
}
