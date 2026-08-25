//! Whittaker-Henderson smoothing of the consensus elution profile.
//!
//! The consensus profile is the median of at most a few dozen apex-normalised fragment
//! traces over eleven or so cycles, so it is noisy — and every strategy in this module
//! leans on it. [`boundary`](super::boundary) needs its curvature, which amplifies noise
//! twice over; [`projection`](super::projection) uses it as a matched filter, where
//! template noise propagates straight into the amplitude; [`emg`](super::emg) fits three
//! parameters to it.
//!
//! Smoothing solves the penalised least-squares problem
//!
//! ```text
//! minimise  Σ_c (z_c - y_c)²  +  λ Σ (D₃ z)²        ⇒  (I + λ D₃ᵀD₃) z = y
//! ```
//!
//! where `D₃` is the third divided-difference operator on the *actual* retention times.
//! Two properties make this the right smoother here rather than, say, Savitzky-Golay:
//!
//! * `D₃` is built from divided differences, so uneven cycle spacing — routine in DIA once
//!   acquisition hiccups or a precursor is enumerated in two isolation windows — is handled
//!   exactly instead of being assumed away.
//! * `D₃` annihilates every polynomial of degree two or less, so constant, linear and
//!   quadratic structure passes through untouched. The smoother is therefore close to
//!   area preserving, which is what lets it run *before* integration without biasing the
//!   result.
//!
//! Retention time is rescaled to unit window width before solving, which makes `λ`
//! dimensionless and transferable between runs with different cycle times.

/// Third-order differences need four points.
const MIN_POINTS_FOR_SMOOTHING: usize = 4;

/// Half-bandwidth of `I + λ D₃ᵀD₃`: a third difference couples four consecutive cycles.
const BANDWIDTH: usize = 3;

/// Smooth `values` sampled at `rt`, in place. A non-positive `lambda`, too few points or a
/// degenerate retention time span leave the input untouched.
pub fn whittaker_henderson(rt: &[f32], values: &mut [f32], lambda: f32) {
    let n = values.len();
    if lambda <= 0.0 || n < MIN_POINTS_FOR_SMOOTHING || rt.len() != n {
        return;
    }

    let span = rt[n - 1] - rt[0];
    if !span.is_finite() || span <= 0.0 {
        return;
    }

    // Unit window width, so that lambda does not depend on the cycle time of the run.
    let scaled_rt: Vec<f64> = rt.iter().map(|&t| ((t - rt[0]) / span) as f64).collect();

    let Some(bands) = penalty_bands(&scaled_rt, lambda as f64) else {
        return;
    };

    let rhs: Vec<f64> = values.iter().map(|&value| value as f64).collect();
    let Some(solution) = solve_banded(bands, &rhs) else {
        return;
    };

    // The profile is an intensity ratio and cannot be negative; the solve can undershoot
    // slightly next to a sharp peak.
    for (target, smoothed) in values.iter_mut().zip(solution.iter()) {
        *target = smoothed.max(0.0) as f32;
    }
}

/// Bands of the symmetric matrix `I + λ D₃ᵀD₃`, as `[diagonal, first, second, third]`
/// off-diagonals. Entry `k` of band `b` is the matrix element `(k, k + b)`.
fn penalty_bands(rt: &[f64], lambda: f64) -> Option<[Vec<f64>; BANDWIDTH + 1]> {
    let n = rt.len();
    let mut bands: [Vec<f64>; BANDWIDTH + 1] =
        [vec![1.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];

    // One row of D₃ per group of four consecutive cycles, accumulated as a rank-one update
    // of DᵀD so the full operator is never materialised.
    for row in 0..n - BANDWIDTH {
        let h0 = rt[row + 1] - rt[row];
        let h1 = rt[row + 2] - rt[row + 1];
        let h2 = rt[row + 3] - rt[row + 2];
        if h0 <= 0.0 || h1 <= 0.0 || h2 <= 0.0 {
            return None;
        }

        let coefficients = [
            -1.0 / (h0 * (h0 + h1) * (h0 + h1 + h2)),
            1.0 / (h0 * h1 * (h1 + h2)),
            -1.0 / ((h0 + h1) * h1 * h2),
            1.0 / ((h0 + h1 + h2) * (h1 + h2) * h2),
        ];

        for a in 0..=BANDWIDTH {
            for b in a..=BANDWIDTH {
                bands[b - a][row + a] += lambda * coefficients[a] * coefficients[b];
            }
        }
    }

    Some(bands)
}

/// Solve a symmetric positive definite banded system by `LDLᵀ` factorisation.
///
/// The matrix is small — one candidate window — but this runs per precursor, so the
/// factorisation stays banded and allocation free beyond the two result vectors.
fn solve_banded(mut bands: [Vec<f64>; BANDWIDTH + 1], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();

    // LDLᵀ in place: `bands[0]` becomes D, `bands[1..]` become the sub-diagonals of L.
    for col in 0..n {
        let pivot = bands[0][col];
        if !pivot.is_finite() || pivot <= 0.0 {
            return None;
        }

        for offset in 1..=BANDWIDTH.min(n - 1 - col) {
            let factor = bands[offset][col] / pivot;
            for inner in offset..=BANDWIDTH.min(n - 1 - col) {
                let value = bands[inner][col] * factor;
                bands[inner - offset][col + offset] -= value;
            }
            bands[offset][col] = factor;
        }
    }

    // Forward substitution with L.
    let mut solution = rhs.to_vec();
    for col in 0..n {
        for offset in 1..=BANDWIDTH.min(n - 1 - col) {
            solution[col + offset] -= bands[offset][col] * solution[col];
        }
    }

    // Divide by D, then back-substitute with Lᵀ.
    for col in 0..n {
        solution[col] /= bands[0][col];
    }
    for col in (0..n).rev() {
        for offset in 1..=BANDWIDTH.min(n - 1 - col) {
            solution[col] -= bands[offset][col] * solution[col + offset];
        }
    }

    if solution.iter().all(|value| value.is_finite()) {
        Some(solution)
    } else {
        None
    }
}
