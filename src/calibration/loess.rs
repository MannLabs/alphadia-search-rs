//! LOESS-style local polynomial regression — the general, domain-agnostic numeric
//! core of calibration (it knows nothing about m/z, RT, ppm or calibration).
//!
//! Pure-f32 port of `alphadia/calibration/models.py::LOESSRegression`. Only the
//! single-feature (1-D) path is supported, which is the only path used in
//! production. The model fits `n_kernels` local weighted polynomials of degree
//! `polynomial_degree` and blends them with tricubic kernels.
//!
//! Two general conditioning choices keep the fit accurate in f32: the design
//! matrix is centered and scaled so its entries stay O(1), and each per-kernel
//! weighted system is solved by QR (which works on the design's condition number
//! rather than squaring it, as the normal equations would). The target is also
//! mean-centered. Callers that fit a small signal riding on a large baseline (as
//! calibration does) should subtract that baseline themselves before fitting.

/// Small epsilon added inside the tricubic kernel (matches the Python `1e-6`).
const TRICUBIC_EPSILON: f32 = 1e-6;

/// A fitted (or fittable) LOESS regression model.
#[derive(Clone, Debug)]
pub struct LoessRegression {
    n_kernels: usize,
    kernel_size: f32,
    polynomial_degree: usize,
    // Fitted parameters (empty until `fit` succeeds).
    scale_mean: Vec<f32>,
    scale_max: Vec<f32>,
    /// `beta[k]` holds the `polynomial_degree + 1` coefficients of kernel `k`, in
    /// the centered/scaled design coordinate `u = (x - center) / scale`.
    beta: Vec<Vec<f32>>,
    /// Centering/scaling applied to the polynomial design matrix (not the tricubic
    /// kernel). A pure conditioning transform that keeps the design entries O(1) so
    /// the f32 QR solve stays accurate; it does not change the fitted function.
    design_center: f32,
    design_scale: f32,
    /// Mean of the target `y`, subtracted before fitting and added back at predict
    /// (standard response centering, improves conditioning of the constant term).
    target_center: f32,
}

impl LoessRegression {
    /// Create an unfitted model. `uniform` kernel placement is not used in
    /// production and is therefore not implemented (density placement only).
    pub fn new(n_kernels: usize, kernel_size: f32, polynomial_degree: usize) -> Self {
        Self {
            n_kernels,
            kernel_size,
            polynomial_degree,
            scale_mean: Vec::new(),
            scale_max: Vec::new(),
            beta: Vec::new(),
            design_center: 0.0,
            design_scale: 1.0,
            target_center: 0.0,
        }
    }

    /// Fit the model on the training data.
    ///
    /// Returns `Err` on degenerate input (too few points, singular kernel system)
    /// which mirrors the Python behaviour of leaving the estimator unfitted.
    pub fn fit(&mut self, x: &[f32], y: &[f32]) -> Result<(), String> {
        let n = x.len();
        if n != y.len() {
            return Err("x and y must have the same length".to_string());
        }
        if n < 2 {
            return Err("At least two datapoints required for fitting.".to_string());
        }
        // Zero kernels would divide by zero in `kernel_indices_density` (`m / n_kernels`).
        if self.n_kernels == 0 {
            return Err("At least one kernel required for fitting.".to_string());
        }

        // === sanity checks: reduce n_kernels then polynomial_degree if needed ===
        let mut n_kernels = self.n_kernels;
        let mut poly_degree = self.polynomial_degree;

        let degrees_freedom = (1 + poly_degree) * n_kernels;
        if n < degrees_freedom {
            n_kernels = (n / (1 + poly_degree)).max(1);
        }
        let degrees_freedom = (1 + poly_degree) * n_kernels;
        if n < degrees_freedom {
            poly_degree = n - 1;
        }

        // === remove outliers using the 0.1 / 99.9 percentiles of x ===
        let p_low = percentile(x, 0.1);
        let p_high = percentile(x, 99.9);
        let mut xf: Vec<f32> = Vec::new();
        let mut yf: Vec<f32> = Vec::new();
        for i in 0..n {
            if p_low < x[i] && x[i] < p_high {
                xf.push(x[i]);
                yf.push(y[i]);
            }
        }
        let m = xf.len();
        if m == 0 {
            return Err("No datapoints left after outlier removal.".to_string());
        }

        // sorted x used for kernel placement and scaling
        let mut x_sorted = xf.clone();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // === density kernel indices ===
        let kernel_indices = kernel_indices_density(m, n_kernels, self.kernel_size);

        // === per-kernel scaling (mean and max absolute deviation) ===
        let mut scale_mean = vec![0.0f32; n_kernels];
        let mut scale_max = vec![0.0f32; n_kernels];
        for k in 0..n_kernels {
            let (start, end) = kernel_indices[k];
            if end <= start {
                return Err("Empty kernel encountered during fitting.".to_string());
            }
            let slice = &x_sorted[start..end];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            let max_abs = slice
                .iter()
                .map(|v| (v - mean).abs())
                .fold(0.0f32, f32::max);
            scale_mean[k] = mean;
            scale_max[k] = max_abs;
        }

        // === design-matrix conditioning (f32 numerical stability) ===
        // Center and scale x used to build the polynomial design matrix so that
        // its entries stay O(1) even for large m/z values. This does not affect
        // the fitted function, only the numerical conditioning of the solve.
        let design_center = xf.iter().sum::<f32>() / m as f32;
        let x_min = xf.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = xf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let half_range = (x_max - x_min) / 2.0;
        let design_scale = if half_range > 0.0 && half_range.is_finite() {
            half_range
        } else {
            1.0
        };

        // === center the design coordinate and the target ===
        let u_scaled: Vec<f32> = xf
            .iter()
            .map(|&xi| (xi - design_center) / design_scale)
            .collect();
        let target_center = yf.iter().sum::<f32>() / m as f32;
        let tc: Vec<f32> = yf.iter().map(|&yi| yi - target_center).collect();

        // === weighted least squares per kernel (f32 QR) ===
        let w = weight_matrix(&xf, &scale_mean, &scale_max, n_kernels);
        let d = poly_degree + 1;

        let mut beta = vec![vec![0.0f32; d]; n_kernels];
        for k in 0..n_kernels {
            let wk: Vec<f32> = w.iter().map(|row| row[k]).collect();
            match fit_kernel_qr(&u_scaled, &wk, &tc, poly_degree) {
                Some(coefs) => beta[k] = coefs,
                None => return Err("Rank-deficient kernel system during fitting.".to_string()),
            }
        }

        self.n_kernels = n_kernels;
        self.polynomial_degree = poly_degree;
        self.scale_mean = scale_mean;
        self.scale_max = scale_max;
        self.beta = beta;
        self.design_center = design_center;
        self.design_scale = design_scale;
        self.target_center = target_center;
        Ok(())
    }

    /// Predict calibrated values for `x`. Must be called after a successful `fit`.
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let w = weight_matrix(x, &self.scale_mean, &self.scale_max, self.n_kernels);
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let u = (xi - self.design_center) / self.design_scale;
                let xrow = poly_features(u, self.polynomial_degree);
                let mut acc = 0.0f32;
                for k in 0..self.n_kernels {
                    let pred_k: f32 = xrow
                        .iter()
                        .zip(self.beta[k].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    acc += pred_k * w[i][k];
                }
                // weights are row-normalized (sum to 1), so the target mean is added
                // back exactly once.
                acc + self.target_center
            })
            .collect()
    }
}

/// Polynomial features `[1, x, x^2, ..., x^degree]` (matches sklearn
/// `PolynomialFeatures(degree, include_bias=True)` for a single feature).
fn poly_features(x: f32, degree: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(degree + 1);
    let mut p = 1.0f32;
    for _ in 0..=degree {
        out.push(p);
        p *= x;
    }
    out
}

/// Solve the weighted least-squares system for one kernel in f32 via QR (modified
/// Gram-Schmidt on the `sqrt(weight)`-scaled polynomial design).
///
/// QR operates on the design's condition number directly, unlike the normal
/// equations `XᵀWX` which square it — so the ppm-scale calibration signal survives
/// in f32. `u` are the centered/scaled design coordinates, `wk` the kernel weights,
/// `tc` the centered target (fitted deviation). Returns the `degree + 1`
/// coefficients, or `None` if the system is rank-deficient.
fn fit_kernel_qr(u: &[f32], wk: &[f32], tc: &[f32], degree: usize) -> Option<Vec<f32>> {
    let m = u.len();
    let d = degree + 1;

    // sqrt(weight)-scaled design columns cols[j][i] = sqrt(w_i) * u_i^j, rhs c_i
    let sw: Vec<f32> = wk.iter().map(|v| v.sqrt()).collect();
    let mut cols: Vec<Vec<f32>> = vec![vec![0.0f32; m]; d];
    for i in 0..m {
        let mut p = sw[i];
        for col in cols.iter_mut() {
            col[i] = p;
            p *= u[i];
        }
    }
    let c: Vec<f32> = (0..m).map(|i| sw[i] * tc[i]).collect();

    // modified Gram-Schmidt QR
    let mut q: Vec<Vec<f32>> = vec![vec![0.0f32; m]; d];
    let mut r = vec![vec![0.0f32; d]; d];
    for j in 0..d {
        let mut v = cols[j].clone();
        for i in 0..j {
            let rij: f32 = (0..m).map(|t| q[i][t] * v[t]).sum();
            r[i][j] = rij;
            for t in 0..m {
                v[t] -= rij * q[i][t];
            }
        }
        let norm: f32 = v.iter().map(|z| z * z).sum::<f32>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return None;
        }
        r[j][j] = norm;
        for t in 0..m {
            q[j][t] = v[t] / norm;
        }
    }

    // qtc[j] = q_j . c, then back-substitute R beta = qtc
    let mut qtc = vec![0.0f32; d];
    for j in 0..d {
        qtc[j] = (0..m).map(|t| q[j][t] * c[t]).sum();
    }
    let mut beta = vec![0.0f32; d];
    for i in (0..d).rev() {
        let mut s = qtc[i];
        for cc in (i + 1)..d {
            s -= r[i][cc] * beta[cc];
        }
        beta[i] = s / r[i][i];
    }
    Some(beta)
}

/// Density-based kernel index intervals over `m` sorted datapoints.
fn kernel_indices_density(m: usize, n_kernels: usize, kernel_size: f32) -> Vec<(usize, usize)> {
    let interval_size = m / n_kernels;
    // interval_extension = (interval_size * kernel_size - interval_size) // 2
    let ext = ((interval_size as f32 * kernel_size - interval_size as f32) / 2.0).floor() as i64;

    let mut out = Vec::with_capacity(n_kernels);
    for k in 0..n_kernels {
        let start = (k * interval_size) as i64;
        let end = start + interval_size as i64;
        let start = (start - ext).max(0) as usize;
        let end = (end + ext).min(m as i64) as usize;
        out.push((start, end));
    }
    out
}

/// Row-normalized tricubic weight matrix of shape `[x.len()][n_kernels]`.
fn weight_matrix(
    x: &[f32],
    scale_mean: &[f32],
    scale_max: &[f32],
    n_kernels: usize,
) -> Vec<Vec<f32>> {
    let mut w = Vec::with_capacity(x.len());
    for &xi in x {
        let mut row: Vec<f32> = (0..n_kernels)
            .map(|k| (xi - scale_mean[k]) / scale_max[k])
            .collect();

        // apply the (edge-aware) tricubic kernel column-wise
        if n_kernels == 1 {
            row[0] = 1.0;
        } else if n_kernels == 2 {
            row[0] = left_open_tricubic(row[0]);
            row[1] = right_open_tricubic(row[1]);
        } else {
            row[0] = left_open_tricubic(row[0]);
            for item in row.iter_mut().take(n_kernels - 1).skip(1) {
                *item = tricubic(*item);
            }
            row[n_kernels - 1] = right_open_tricubic(row[n_kernels - 1]);
        }

        // row-normalize
        let s: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s;
        }
        w.push(row);
    }
    w
}

/// Tricubic weight kernel: `(1 - |x|^3)^3 + eps` for `|x| <= 1`, else `0`.
fn tricubic(x: f32) -> f32 {
    if x.abs() <= 1.0 {
        let a = x.abs();
        (1.0 - a * a * a).powi(3) + TRICUBIC_EPSILON
    } else {
        0.0
    }
}

/// Tricubic kernel that assigns weight `1` to values left of the center (`x < 0`).
fn left_open_tricubic(x: f32) -> f32 {
    if x < 0.0 {
        1.0
    } else {
        tricubic(x)
    }
}

/// Tricubic kernel that assigns weight `1` to values right of the center (`x > 0`).
fn right_open_tricubic(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        tricubic(x)
    }
}

/// Linear-interpolated percentile (matches `numpy.percentile` default method).
pub fn percentile(a: &[f32], q: f32) -> f32 {
    let mut sorted = a.to_vec();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = q / 100.0 * (n - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Median via the 50th percentile.
pub fn median(a: &[f32]) -> f32 {
    percentile(a, 50.0)
}
