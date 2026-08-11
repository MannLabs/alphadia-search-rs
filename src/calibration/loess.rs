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
//!
//! Summation order is load-bearing: the calibration port is verified bit-for-bit
//! against the Python implementation, so refactors here must preserve the order in
//! which terms are accumulated, not merely the mathematical result.

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

        let (n_kernels, poly_degree) = adjust_capacity(n, self.n_kernels, self.polynomial_degree);

        let (xf, yf) = remove_outliers(x, y);
        if xf.is_empty() {
            return Err("No datapoints left after outlier removal.".to_string());
        }

        // Kernels are placed over the sorted x so each covers an equal number of
        // datapoints; the fit itself runs on the unsorted data.
        let x_sorted = sorted_copy(&xf);
        let kernel_indices = kernel_indices_density(xf.len(), n_kernels, self.kernel_size);
        let (scale_mean, scale_max) = kernel_scales(&x_sorted, &kernel_indices)?;

        let (design_center, design_scale) = design_conditioning(&xf);
        let u_scaled: Vec<f32> = xf
            .iter()
            .map(|&xi| (xi - design_center) / design_scale)
            .collect();
        let target_center = mean(&yf);
        let target_centered: Vec<f32> = yf.iter().map(|&yi| yi - target_center).collect();

        let weights = WeightMatrix::new(&xf, &scale_mean, &scale_max, n_kernels);
        let beta = solve_kernels(&u_scaled, &target_centered, &weights, poly_degree)?;

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
    ///
    /// The per-row weight and polynomial buffers are allocated once and reused, so
    /// prediction over a full precursor table allocates only the output vector.
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.n_kernels];
        let mut powers = vec![0.0f32; self.polynomial_degree + 1];

        x.iter()
            .map(|&xi| {
                kernel_weights(xi, &self.scale_mean, &self.scale_max, &mut weights);
                fill_poly_features((xi - self.design_center) / self.design_scale, &mut powers);

                let mut acc = 0.0f32;
                for (beta_k, &weight) in self.beta.iter().zip(&weights) {
                    let pred_k: f32 = powers.iter().zip(beta_k).map(|(a, b)| a * b).sum();
                    acc += pred_k * weight;
                }
                // weights are row-normalized (sum to 1), so the target mean is added
                // back exactly once.
                acc + self.target_center
            })
            .collect()
    }
}

/// Reduce `n_kernels`, then the polynomial degree, until the model has no more
/// degrees of freedom than `n` datapoints.
fn adjust_capacity(n: usize, n_kernels: usize, poly_degree: usize) -> (usize, usize) {
    let mut n_kernels = n_kernels;
    let mut poly_degree = poly_degree;

    if n < (1 + poly_degree) * n_kernels {
        n_kernels = (n / (1 + poly_degree)).max(1);
    }
    if n < (1 + poly_degree) * n_kernels {
        poly_degree = n - 1;
    }
    (n_kernels, poly_degree)
}

/// Drop points outside the 0.1 / 99.9 percentiles of `x`, keeping `y` aligned.
fn remove_outliers(x: &[f32], y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let bounds = percentiles(x, &[0.1, 99.9]);
    let (p_low, p_high) = (bounds[0], bounds[1]);

    let mut xf = Vec::new();
    let mut yf = Vec::new();
    for (&xi, &yi) in x.iter().zip(y) {
        if p_low < xi && xi < p_high {
            xf.push(xi);
            yf.push(yi);
        }
    }
    (xf, yf)
}

/// Center and half-extent of every kernel, taken from its slice of the sorted x.
///
/// `scale_max` is the largest absolute deviation from the slice mean, i.e. the
/// distance at which the tricubic kernel reaches zero.
fn kernel_scales(
    x_sorted: &[f32],
    kernel_indices: &[(usize, usize)],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut scale_mean = Vec::with_capacity(kernel_indices.len());
    let mut scale_max = Vec::with_capacity(kernel_indices.len());

    for &(start, end) in kernel_indices {
        if end <= start {
            return Err("Empty kernel encountered during fitting.".to_string());
        }
        let slice = &x_sorted[start..end];
        let slice_mean = mean(slice);
        scale_mean.push(slice_mean);
        scale_max.push(
            slice
                .iter()
                .map(|v| (v - slice_mean).abs())
                .fold(0.0f32, f32::max),
        );
    }
    Ok((scale_mean, scale_max))
}

/// Centering and scaling that keep the polynomial design matrix entries O(1).
///
/// A pure conditioning transform for the f32 solve: it does not change the fitted
/// function. Degenerate ranges fall back to a unit scale.
fn design_conditioning(x: &[f32]) -> (f32, f32) {
    let center = mean(x);
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let half_range = (max - min) / 2.0;
    let scale = if half_range > 0.0 && half_range.is_finite() {
        half_range
    } else {
        1.0
    };
    (center, scale)
}

/// Solve every kernel's weighted least-squares system, reusing one column buffer.
fn solve_kernels(
    u_scaled: &[f32],
    target_centered: &[f32],
    weights: &WeightMatrix,
    poly_degree: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let mut beta = Vec::with_capacity(weights.n_kernels);
    let mut column = Vec::with_capacity(u_scaled.len());

    for k in 0..weights.n_kernels {
        weights.write_column(k, &mut column);
        match fit_kernel_qr(u_scaled, &column, target_centered, poly_degree) {
            Some(coefs) => beta.push(coefs),
            None => return Err("Rank-deficient kernel system during fitting.".to_string()),
        }
    }
    Ok(beta)
}

/// Polynomial features `[1, x, x^2, ..., x^degree]` written into `out`, whose
/// length sets the degree (matches sklearn `PolynomialFeatures(include_bias=True)`
/// for a single feature).
fn fill_poly_features(x: f32, out: &mut [f32]) {
    let mut power = 1.0f32;
    for slot in out.iter_mut() {
        *slot = power;
        power *= x;
    }
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

/// Row-normalized tricubic kernel weights for many x values, stored flat as
/// `n_rows × n_kernels`.
///
/// Row-major with an explicit stride: the previous `Vec<Vec<f32>>` allocated once
/// per row, and again per kernel when fitting transposed a column out of it.
struct WeightMatrix {
    values: Vec<f32>,
    n_kernels: usize,
}

impl WeightMatrix {
    fn new(x: &[f32], scale_mean: &[f32], scale_max: &[f32], n_kernels: usize) -> Self {
        let mut values = vec![0.0f32; x.len() * n_kernels];
        for (row, &xi) in x.iter().enumerate() {
            let start = row * n_kernels;
            kernel_weights(
                xi,
                scale_mean,
                scale_max,
                &mut values[start..start + n_kernels],
            );
        }
        Self { values, n_kernels }
    }

    /// Copy kernel `k`'s weights for every row into `out`, which is reused across
    /// kernels by the caller.
    fn write_column(&self, k: usize, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.values.iter().skip(k).step_by(self.n_kernels));
    }
}

/// Row-normalized tricubic weights of a single `x` against all kernels, written
/// into `out` (whose length is the kernel count).
fn kernel_weights(xi: f32, scale_mean: &[f32], scale_max: &[f32], out: &mut [f32]) {
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = (xi - scale_mean[k]) / scale_max[k];
    }
    apply_tricubic_kernels(out);
    normalize_row(out);
}

/// Apply the edge-aware tricubic kernel to a row of scaled distances in place.
///
/// The outer flanks of the first and last kernels are one-padded so the model
/// extrapolates flat beyond the fitted range instead of dropping to zero weight.
fn apply_tricubic_kernels(row: &mut [f32]) {
    let n_kernels = row.len();
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
}

/// Scale a weight row to sum to 1.
///
/// Deliberately unguarded against a zero or non-finite sum: the Python original
/// divides by `np.sum(w, axis=1)` unguarded too, and the port is verified against
/// it bit-for-bit. A guard here would change the numbers, so it belongs in both
/// implementations or neither.
fn normalize_row(row: &mut [f32]) {
    let sum: f32 = row.iter().sum();
    for weight in row.iter_mut() {
        *weight /= sum;
    }
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

fn mean(a: &[f32]) -> f32 {
    a.iter().sum::<f32>() / a.len() as f32
}

/// Sorted copy of `a`, ordering NaN as equal (the comparator numpy's sort-based
/// percentile effectively uses on well-formed input).
fn sorted_copy(a: &[f32]) -> Vec<f32> {
    let mut sorted = a.to_vec();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Linear-interpolated percentile of an already-sorted slice.
fn percentile_of_sorted(sorted: &[f32], q: f32) -> f32 {
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

/// Linear-interpolated percentile (matches `numpy.percentile` default method).
pub fn percentile(a: &[f32], q: f32) -> f32 {
    percentile_of_sorted(&sorted_copy(a), q)
}

/// Several percentiles of the same data, sorting once instead of once per call.
pub fn percentiles(a: &[f32], qs: &[f32]) -> Vec<f32> {
    let sorted = sorted_copy(a);
    qs.iter()
        .map(|&q| percentile_of_sorted(&sorted, q))
        .collect()
}

/// Median via the 50th percentile.
pub fn median(a: &[f32]) -> f32 {
    percentile(a, 50.0)
}

/// Median of the absolute values, in a single allocation.
pub fn median_abs(a: &[f32]) -> f32 {
    let mut absolute: Vec<f32> = a.iter().map(|v| v.abs()).collect();
    absolute.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    percentile_of_sorted(&absolute, 50.0)
}
