//! Internal f32 / f64 parity experiment for the core numeric kernel of LOESS: the
//! per-kernel **weighted local polynomial least-squares** solve.
//!
//! This is a domain-agnostic numerics test (no m/z, RT or ppm). The hard case for
//! f32 is fitting a small local signal that rides on a large baseline — then the
//! fitted correction is a tiny relative perturbation that naive f32 arithmetic
//! cannot resolve. The experiment compares three ways of solving the same weighted
//! polynomial system against an f64 reference and reports the dimensionless
//! relative error of the fitted curve:
//!
//!   1. `NaiveF32` — normal equations `A = XᵀWX` solved by Gaussian elimination in
//!      f32. Forming `XᵀWX` squares the condition number, so f32 loses precision.
//!   2. `QrF32` — QR (modified Gram-Schmidt) in f32: works on the design's
//!      condition number, not its square. Better, but still fits the raw target.
//!   3. `ImprovedF32` — the production strategy: fit the residual from the baseline
//!      (`y - x`) *and* solve by QR, entirely in f32. Matches the f64 reference to
//!      the f32 rounding floor.
//!
//! The solvers here are self-contained reimplementations so the comparison isolates
//! exactly the solve precision, independent of the production code path.

const DEGREE: usize = 2;
const N_COEF: usize = DEGREE + 1;

/// A weighted-least-squares problem that is hard for f32: a small local signal
/// `s(x)` riding on a large baseline `x`. Returned in f64 (the "truth").
///
/// The baseline dominates the target magnitude while the signal is what the fit
/// must recover — exactly the regime where forming the normal equations in f32
/// discards the signal.
fn make_dataset() -> (Vec<f64>, Vec<f64>) {
    let n = 4000;
    let baseline_lo = 300.0;
    let baseline_hi = 1200.0;
    let signal_scale = 1e-5; // signal is ~1e-5 of the baseline: the hard case

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = baseline_lo + (baseline_hi - baseline_lo) * (i as f64) / ((n - 1) as f64);
        // small, smoothly varying local signal plus deterministic pseudo-noise
        let signal = signal_scale * (1.0 - 0.4 * (xi - 750.0) / 450.0);
        let s = (i as f64 * 12.9898).sin() * 43758.5453;
        let noise = ((s - s.floor()) * 2.0 - 1.0) * 1e-3;
        y.push(xi * (1.0 + signal) + noise);
        x.push(xi);
    }
    (x, y)
}

// ----- tricubic kernel weights (self-contained copies of the LOESS weighting) -----

fn tricubic(x: f64) -> f64 {
    if x.abs() <= 1.0 {
        let a = x.abs();
        (1.0 - a * a * a).powi(3) + 1e-6
    } else {
        0.0
    }
}

/// Two-kernel, row-normalized weights with open (weight-1) outer edges, matching the
/// production `n_kernels == 2` case. Skewed weights of this kind are what make the
/// per-kernel systems ill-conditioned.
fn weights_two_kernels(x: &[f64]) -> Vec<[f64; 2]> {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = sorted.len();
    let half = m / 2;
    let mean0 = sorted[..half].iter().sum::<f64>() / half as f64;
    let max0 = sorted[..half]
        .iter()
        .map(|v| (v - mean0).abs())
        .fold(0.0, f64::max);
    let mean1 = sorted[half..].iter().sum::<f64>() / (m - half) as f64;
    let max1 = sorted[half..]
        .iter()
        .map(|v| (v - mean1).abs())
        .fold(0.0, f64::max);

    x.iter()
        .map(|&xi| {
            let z0 = (xi - mean0) / max0;
            let z1 = (xi - mean1) / max1;
            // kernel 0 is open to the left, kernel 1 open to the right
            let w0 = if z0 < 0.0 { 1.0 } else { tricubic(z0) };
            let w1 = if z1 > 0.0 { 1.0 } else { tricubic(z1) };
            let s = w0 + w1;
            [w0 / s, w1 / s]
        })
        .collect()
}

// ----- naive normal-equations solve (Gaussian elimination), f32 and f64 -----

macro_rules! impl_naive {
    ($name:ident, $t:ty) => {
        /// Weighted polynomial fit for one kernel via the normal equations `A=XᵀWX`.
        fn $name(u: &[$t], w: &[$t], yc: &[$t]) -> [$t; N_COEF] {
            let mut a = [[0 as $t; N_COEF]; N_COEF];
            let mut b = [0 as $t; N_COEF];
            for i in 0..u.len() {
                let wi = w[i];
                if wi == 0 as $t {
                    continue;
                }
                let mut phi = [1 as $t; N_COEF];
                for d in 1..N_COEF {
                    phi[d] = phi[d - 1] * u[i];
                }
                for r in 0..N_COEF {
                    let wr = wi * phi[r];
                    for c in 0..N_COEF {
                        a[r][c] += wr * phi[c];
                    }
                    b[r] += wr * yc[i];
                }
            }
            // Gaussian elimination with partial pivoting.
            for i in 0..N_COEF {
                let mut piv = i;
                for r in (i + 1)..N_COEF {
                    if a[r][i].abs() > a[piv][i].abs() {
                        piv = r;
                    }
                }
                a.swap(i, piv);
                b.swap(i, piv);
                let diag = a[i][i];
                for r in (i + 1)..N_COEF {
                    let f = a[r][i] / diag;
                    for c in i..N_COEF {
                        a[r][c] -= f * a[i][c];
                    }
                    b[r] -= f * b[i];
                }
            }
            let mut beta = [0 as $t; N_COEF];
            for i in (0..N_COEF).rev() {
                let mut s = b[i];
                for c in (i + 1)..N_COEF {
                    s -= a[i][c] * beta[c];
                }
                beta[i] = s / a[i][i];
            }
            beta
        }
    };
}

impl_naive!(fit_naive_f64, f64);
impl_naive!(fit_naive_f32, f32);

// ----- improved f32 solve: QR via modified Gram-Schmidt -----

/// Weighted polynomial fit for one kernel in pure f32, solved by QR (modified
/// Gram-Schmidt) on the `sqrt(weight)`-scaled design. QR works on the design's
/// condition number directly rather than its square, so it stays accurate in f32.
fn fit_qr_f32(u: &[f32], w: &[f32], yc: &[f32]) -> [f32; N_COEF] {
    let m = u.len();
    let sw: Vec<f32> = w.iter().map(|v| v.sqrt()).collect();
    let mut cols = [(); N_COEF].map(|_| vec![0.0f32; m]);
    for i in 0..m {
        let mut p = sw[i];
        for col in cols.iter_mut() {
            col[i] = p;
            p *= u[i];
        }
    }
    let c: Vec<f32> = (0..m).map(|i| sw[i] * yc[i]).collect();

    let mut q = [(); N_COEF].map(|_| vec![0.0f32; m]);
    let mut r = [[0.0f32; N_COEF]; N_COEF];
    for j in 0..N_COEF {
        let mut v = cols[j].clone();
        for i in 0..j {
            let rij: f32 = (0..m).map(|t| q[i][t] * v[t]).sum();
            r[i][j] = rij;
            for t in 0..m {
                v[t] -= rij * q[i][t];
            }
        }
        let norm: f32 = v.iter().map(|z| z * z).sum::<f32>().sqrt();
        r[j][j] = norm;
        for t in 0..m {
            q[j][t] = v[t] / norm;
        }
    }
    let mut qtc = [0.0f32; N_COEF];
    for j in 0..N_COEF {
        qtc[j] = (0..m).map(|t| q[j][t] * c[t]).sum();
    }
    let mut beta = [0.0f32; N_COEF];
    for i in (0..N_COEF).rev() {
        let mut s = qtc[i];
        for c in (i + 1)..N_COEF {
            s -= r[i][c] * beta[c];
        }
        beta[i] = s / r[i][i];
    }
    beta
}

// ----- prediction + comparison harness -----

fn phi_f64(u: f64) -> [f64; N_COEF] {
    let mut p = [1.0; N_COEF];
    for d in 1..N_COEF {
        p[d] = p[d - 1] * u;
    }
    p
}

/// Which precision, which solver, and whether to fit the residual from the baseline
/// (`y - x`) rather than the raw target `y`.
#[derive(Clone, Copy)]
enum Method {
    NaiveF64,
    NaiveF32,
    QrF32,
    /// The production strategy: fit the residual (small target) and solve by QR.
    ImprovedF32,
}

/// Fit the blended 2-kernel curve and return predictions at every `x`.
fn fit_predict(x: &[f64], y: &[f64], method: Method) -> Vec<f64> {
    let m = x.len();
    let center = x.iter().sum::<f64>() / m as f64;
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scale = (xmax - xmin) / 2.0;
    let u: Vec<f64> = x.iter().map(|&xi| (xi - center) / scale).collect();
    let weights = weights_two_kernels(x);

    let fit_residual = matches!(method, Method::ImprovedF32);
    let target: Vec<f64> = if fit_residual {
        (0..m).map(|i| y[i] - x[i]).collect()
    } else {
        y.to_vec()
    };
    let target_mean = target.iter().sum::<f64>() / m as f64;
    let tc: Vec<f64> = target.iter().map(|&t| t - target_mean).collect();

    let mut beta = [[0.0f64; N_COEF]; 2];
    for (k, bk) in beta.iter_mut().enumerate() {
        let w: Vec<f64> = weights.iter().map(|wk| wk[k]).collect();
        *bk = match method {
            Method::NaiveF64 => fit_naive_f64(&u, &w, &tc),
            Method::NaiveF32 => solve_f32(&u, &w, &tc, false),
            Method::QrF32 => solve_f32(&u, &w, &tc, true),
            Method::ImprovedF32 => solve_f32(&u, &w, &tc, true),
        };
    }

    // Evaluate in the method's own precision so the comparison reflects the true
    // end-to-end pipeline (f32 methods also incur the f32 output rounding floor).
    let f32_eval = !matches!(method, Method::NaiveF64);
    (0..m)
        .map(|i| {
            let baseline = target_mean + if fit_residual { x[i] } else { 0.0 };
            if f32_eval {
                let ui = u[i] as f32;
                let phi = [1.0f32, ui, ui * ui];
                let mut acc = 0.0f32;
                for k in 0..2 {
                    let pk: f32 = (0..N_COEF).map(|d| phi[d] * beta[k][d] as f32).sum();
                    acc += pk * weights[i][k] as f32;
                }
                (acc + baseline as f32) as f64
            } else {
                let phi = phi_f64(u[i]);
                let mut acc = 0.0;
                for k in 0..2 {
                    let pk: f64 = (0..N_COEF).map(|d| phi[d] * beta[k][d]).sum();
                    acc += pk * weights[i][k];
                }
                acc + baseline
            }
        })
        .collect()
}

/// Run one f32 kernel solve (QR or naive normal equations) and widen to f64.
fn solve_f32(u: &[f64], w: &[f64], tc: &[f64], qr: bool) -> [f64; N_COEF] {
    let uf: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    let wf: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    let yf: Vec<f32> = tc.iter().map(|&v| v as f32).collect();
    let b = if qr {
        fit_qr_f32(&uf, &wf, &yf)
    } else {
        fit_naive_f32(&uf, &wf, &yf)
    };
    std::array::from_fn(|i| b[i] as f64)
}

/// Median and max **relative** error (dimensionless) between a method's fitted curve
/// and the f64 normal-equations reference.
fn relative_error(method: Method) -> (f64, f64) {
    let (x, y) = make_dataset();
    let reference = fit_predict(&x, &y, Method::NaiveF64);
    let candidate = fit_predict(&x, &y, method);
    let mut rel: Vec<f64> = (0..x.len())
        .map(|i| (candidate[i] - reference[i]).abs() / reference[i].abs())
        .collect();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (rel[rel.len() / 2], *rel.last().unwrap())
}

#[test]
fn naive_f32_loses_precision_vs_f64() {
    // The naive f32 normal-equations solve differs from the f64 solve by a
    // non-trivial relative amount on the ill-conditioned weighted problem.
    let (median, max) = relative_error(Method::NaiveF32);
    eprintln!("naive_f32     vs f64:  median rel = {median:.2e}, max rel = {max:.2e}");
    assert!(
        median > 2e-7,
        "expected naive f32 to lose >2e-7 relative, got {median:.2e}"
    );
}

#[test]
fn improved_f32_matches_f64() {
    // The improved pure-f32 algorithm (fit residual + QR solve) matches the f64
    // reference to the f32 rounding floor (~1e-7 relative, 24-bit mantissa).
    let (median, max) = relative_error(Method::ImprovedF32);
    eprintln!("improved_f32  vs f64:  median rel = {median:.2e}, max rel = {max:.2e}");
    assert!(
        median < 1e-7,
        "expected improved f32 within 1e-7 relative, got {median:.2e}"
    );
}

#[test]
fn improved_f32_beats_naive_f32() {
    // Attribute the two independent f32 fixes for the record (QR alone vs QR + fit
    // the residual), then assert the combined algorithm is far closer to f64.
    let (naive, _) = relative_error(Method::NaiveF32);
    let (qr_only, _) = relative_error(Method::QrF32);
    let (improved, _) = relative_error(Method::ImprovedF32);
    eprintln!(
        "naive={naive:.2e}  qr_only={qr_only:.2e}  improved={improved:.2e}  (factor {:.0}x)",
        naive / improved.max(1e-30)
    );
    assert!(
        improved * 5.0 < naive,
        "improved f32 ({improved:.2e}) not >=5x better than naive f32 ({naive:.2e})"
    );
}
