//! Numerical building blocks of the dropout-curve fit.

use std::cell::Cell;

/// Below this argument the log normal CDF is computed from its asymptotic series, because the
/// CDF itself underflows further out.
const LOG_NDTR_ASYMPTOTIC_BELOW: f64 = -20.0;
/// Above this argument `log(1 - x)` is approximated by `-x`.
const LOG_NDTR_LINEAR_ABOVE: f64 = 6.0;

const NELDER_MEAD_REFLECTION: f64 = 1.0;
const NELDER_MEAD_EXPANSION: f64 = 2.0;
const NELDER_MEAD_CONTRACTION: f64 = 0.5;
const NELDER_MEAD_SHRINK: f64 = 0.5;
const NELDER_MEAD_NONZERO_STEP: f64 = 0.05;
const NELDER_MEAD_ZERO_STEP: f64 = 0.00025;
const NELDER_MEAD_X_TOLERANCE: f64 = 1e-4;
const NELDER_MEAD_F_TOLERANCE: f64 = 1e-4;
const NELDER_MEAD_EVALUATIONS_PER_DIMENSION: usize = 200;

/// Standard normal CDF, without cancellation in either tail.
pub fn normal_cdf(a: f64) -> f64 {
    let x = a * std::f64::consts::FRAC_1_SQRT_2;
    if x < -std::f64::consts::FRAC_1_SQRT_2 {
        0.5 * libm::erfc(-x)
    } else if x > std::f64::consts::FRAC_1_SQRT_2 {
        1.0 - 0.5 * libm::erfc(x)
    } else {
        0.5 + 0.5 * libm::erf(x)
    }
}

/// Log of the standard normal CDF, accurate in both tails.
pub fn log_normal_cdf(a: f64) -> f64 {
    if a > LOG_NDTR_LINEAR_ABOVE {
        return -normal_cdf(-a);
    }
    if a > LOG_NDTR_ASYMPTOTIC_BELOW {
        return normal_cdf(a).ln();
    }
    // log Phi(a) = -a^2 / 2 - log(-a) - log(sqrt(2 pi)) + log(1 - 1/a^2 + 3/a^4 - 15/a^6 ...)
    let log_leading = -0.5 * a * a - (-a).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let inverse_square = 1.0 / (a * a);
    let mut series: f64 = 1.0;
    let mut term: f64 = 1.0;
    let mut k = 1.0;
    loop {
        term *= -(2.0 * k - 1.0) * inverse_square;
        if term.abs() <= f64::EPSILON * series.abs() {
            break;
        }
        series += term;
        k += 1.0;
    }
    log_leading + series.ln()
}

pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Quantile with linear interpolation between the order statistics, reordering the non-empty
/// slice.
pub fn quantile(values: &mut [f64], probability: f64) -> f64 {
    let position = (values.len() - 1) as f64 * probability;
    let below = position.floor() as usize;
    let (_, &mut a, above) = values.select_nth_unstable_by(below, f64::total_cmp);
    match above.iter().copied().min_by(f64::total_cmp) {
        Some(b) => a + (position - below as f64) * (b - a),
        None => a,
    }
}

/// Unconstrained Nelder-Mead minimisation with the standard coefficients, an initial simplex
/// of 5 % steps (0.00025 for zero coordinates), stopping when the simplex spans at most 1e-4
/// in every coordinate and in function value, or after 200 evaluations per dimension.
pub fn nelder_mead<const N: usize>(
    mut function: impl FnMut(&[f64; N]) -> f64,
    x0: [f64; N],
) -> [f64; N] {
    let evaluations = Cell::new(0);
    let mut evaluate = |x: [f64; N]| {
        evaluations.set(evaluations.get() + 1);
        (x, function(&x))
    };
    let mut simplex: Vec<([f64; N], f64)> = (0..=N)
        .map(|k| {
            let mut vertex = x0;
            if k > 0 {
                vertex[k - 1] = if x0[k - 1] != 0.0 {
                    (1.0 + NELDER_MEAD_NONZERO_STEP) * x0[k - 1]
                } else {
                    NELDER_MEAD_ZERO_STEP
                };
            }
            evaluate(vertex)
        })
        .collect();
    let max_evaluations = N * NELDER_MEAD_EVALUATIONS_PER_DIMENSION;
    let along = |from: &[f64; N], to: &[f64; N], t: f64| -> [f64; N] {
        std::array::from_fn(|k| from[k] + t * (to[k] - from[k]))
    };
    loop {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (best, best_value) = simplex[0];
        let x_spread = simplex[1..]
            .iter()
            .flat_map(|(vertex, _)| (0..N).map(move |k| (vertex[k] - best[k]).abs()))
            .fold(0.0, f64::max);
        let f_spread = simplex[1..]
            .iter()
            .map(|(_, value)| (value - best_value).abs())
            .fold(0.0, f64::max);
        if (x_spread <= NELDER_MEAD_X_TOLERANCE && f_spread <= NELDER_MEAD_F_TOLERANCE)
            || evaluations.get() >= max_evaluations
        {
            return best;
        }

        let (worst, worst_value) = simplex[N];
        let second_worst_value = simplex[N - 1].1;
        let centroid: [f64; N] =
            std::array::from_fn(|k| simplex[..N].iter().map(|(v, _)| v[k]).sum::<f64>() / N as f64);
        let reflected = evaluate(along(&centroid, &worst, -NELDER_MEAD_REFLECTION));
        if reflected.1 < best_value {
            let expanded = evaluate(along(
                &centroid,
                &worst,
                -NELDER_MEAD_REFLECTION * NELDER_MEAD_EXPANSION,
            ));
            simplex[N] = if expanded.1 < reflected.1 {
                expanded
            } else {
                reflected
            };
            continue;
        }
        if reflected.1 < second_worst_value {
            simplex[N] = reflected;
            continue;
        }
        let contracted = if reflected.1 < worst_value {
            evaluate(along(
                &centroid,
                &worst,
                -NELDER_MEAD_REFLECTION * NELDER_MEAD_CONTRACTION,
            ))
        } else {
            evaluate(along(&centroid, &worst, NELDER_MEAD_CONTRACTION))
        };
        if contracted.1 < reflected.1.min(worst_value) {
            simplex[N] = contracted;
            continue;
        }
        for vertex in simplex.iter_mut().skip(1) {
            *vertex = evaluate(along(&best, &vertex.0, NELDER_MEAD_SHRINK));
        }
    }
}
