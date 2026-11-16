//! Numerical building blocks of the censored fit, with the operation order of the array
//! reference implementation it was validated against, so that both agree to rounding.

/// Block size below which a pairwise sum adds sequentially with eight accumulators.
const PAIRWISE_BLOCK_SIZE: usize = 128;
const PAIRWISE_UNROLL: usize = 8;
/// Above this length the two halves of a pairwise sum are added in parallel; the split points
/// do not depend on it, so the result is the same.
const PARALLEL_PAIRWISE_LENGTH: usize = 1 << 16;
/// Below this argument the log normal CDF is computed from its asymptotic series.
const LOG_NDTR_ASYMPTOTIC_BELOW: f64 = -20.0;
/// Above this argument `log(1 - x)` is approximated by `-x`.
const LOG_NDTR_LINEAR_ABOVE: f64 = 6.0;
const LOGIT_DIRECT_BELOW: f64 = 0.3;
const LOGIT_DIRECT_ABOVE: f64 = 0.65;

const NELDER_MEAD_REFLECTION: f64 = 1.0;
const NELDER_MEAD_EXPANSION: f64 = 2.0;
const NELDER_MEAD_CONTRACTION: f64 = 0.5;
const NELDER_MEAD_SHRINK: f64 = 0.5;
const NELDER_MEAD_NONZERO_STEP: f64 = 0.05;
const NELDER_MEAD_ZERO_STEP: f64 = 0.00025;
const NELDER_MEAD_X_TOLERANCE: f64 = 1e-4;
const NELDER_MEAD_F_TOLERANCE: f64 = 1e-4;
const NELDER_MEAD_EVALUATIONS_PER_DIMENSION: usize = 200;

/// Sum with eight accumulators per block of 128 and a recursive split above, the summation
/// of contiguous float arrays in the reference implementation.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < PAIRWISE_UNROLL {
        let mut sum = 0.0;
        for &value in values {
            sum += value;
        }
        sum
    } else if n <= PAIRWISE_BLOCK_SIZE {
        let mut r = [0.0; PAIRWISE_UNROLL];
        r.copy_from_slice(&values[..PAIRWISE_UNROLL]);
        let blocked = n - n % PAIRWISE_UNROLL;
        for block in values[PAIRWISE_UNROLL..blocked].chunks_exact(PAIRWISE_UNROLL) {
            for (accumulator, &value) in r.iter_mut().zip(block) {
                *accumulator += value;
            }
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &value in &values[blocked..] {
            sum += value;
        }
        sum
    } else {
        let mut half = n / 2;
        half -= half % PAIRWISE_UNROLL;
        let (left, right) = values.split_at(half);
        if n > PARALLEL_PAIRWISE_LENGTH {
            let (a, b) = rayon::join(|| pairwise_sum(left), || pairwise_sum(right));
            a + b
        } else {
            pairwise_sum(left) + pairwise_sum(right)
        }
    }
}

/// Sum of a reduction: the identity plus the pairwise sum.
pub fn array_sum(values: &[f64]) -> f64 {
    0.0 + pairwise_sum(values)
}

pub fn array_mean(values: &[f64]) -> f64 {
    array_sum(values) / values.len() as f64
}

/// Population variance.
pub fn array_variance(values: &[f64]) -> f64 {
    let mean = array_mean(values);
    let squares: Vec<f64> = values.iter().map(|&v| (v - mean) * (v - mean)).collect();
    array_sum(&squares) / values.len() as f64
}

/// NaN to 0 and infinities to the largest finite values.
pub fn nan_to_num(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value == f64::INFINITY {
        f64::MAX
    } else if value == f64::NEG_INFINITY {
        f64::MIN
    } else {
        value
    }
}

/// Clip that keeps NaN.
pub fn clip(value: f64, low: f64, high: f64) -> f64 {
    if value.is_nan() {
        value
    } else {
        value.max(low).min(high)
    }
}

/// Median of a non-empty slice, reordering it.
pub fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (_, &mut upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return upper;
    }
    let lower = values[..n / 2]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (lower + upper) / 2.0
}

/// Quantile with linear interpolation between the order statistics (sample quantile type 7),
/// reordering the non-empty slice.
pub fn quantile(values: &mut [f64], probability: f64) -> f64 {
    let n = values.len();
    let virtual_index = (n - 1) as f64 * probability;
    if virtual_index >= (n - 1) as f64 {
        return *values.select_nth_unstable_by(n - 1, f64::total_cmp).1;
    }
    let previous_index = virtual_index.floor();
    let gamma = virtual_index - previous_index;
    let previous_index = previous_index as usize;
    let (_, &mut a, rest) = values.select_nth_unstable_by(previous_index, f64::total_cmp);
    let b = rest.iter().copied().fold(f64::INFINITY, f64::min);
    let difference = b - a;
    if gamma >= 0.5 {
        b - difference * (1.0 - gamma)
    } else {
        a + difference * gamma
    }
}

/// Piecewise-linear interpolation, constant beyond the ends; `x` is sorted ascending.
pub fn interpolate(value: f64, x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if value.is_nan() {
        return value;
    }
    if n == 1 || value < x[0] {
        return y[0];
    }
    if value > x[n - 1] {
        return y[n - 1];
    }
    let j = x.partition_point(|&m| m <= value) - 1;
    if j == n - 1 || x[j] == value {
        return y[j];
    }
    let slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
    let result = slope * (value - x[j]) + y[j];
    if result.is_nan() {
        let result = slope * (value - x[j + 1]) + y[j + 1];
        if result.is_nan() && y[j] == y[j + 1] {
            return y[j];
        }
        return result;
    }
    result
}

fn normal_cdf(a: f64) -> f64 {
    let x = a * std::f64::consts::FRAC_1_SQRT_2;
    let z = x.abs();
    if z < std::f64::consts::FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(x)
    } else {
        let y = 0.5 * libm::erfc(z);
        if x > 0.0 {
            1.0 - y
        } else {
            y
        }
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
    let log_lhs = -0.5 * a * a - (-a).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut last_total: f64 = 0.0;
    let mut right_hand_side: f64 = 1.0;
    let mut numerator = 1.0;
    let mut denominator_factor = 1.0;
    let denominator_step = 1.0 / (a * a);
    let mut sign = 1.0;
    let mut i = 0.0;
    while (last_total - right_hand_side).abs() > f64::EPSILON {
        i += 1.0;
        last_total = right_hand_side;
        sign = -sign;
        denominator_factor *= denominator_step;
        numerator *= 2.0 * i - 1.0;
        right_hand_side += sign * numerator * denominator_factor;
    }
    log_lhs + right_hand_side.ln()
}

pub fn logit(p: f64) -> f64 {
    if !(LOGIT_DIRECT_BELOW..=LOGIT_DIRECT_ABOVE).contains(&p) {
        (p / (1.0 - p)).ln()
    } else {
        let s = 2.0 * (p - 0.5);
        s.ln_1p() - (-s).ln_1p()
    }
}

pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn nan_aware_max(values: impl Iterator<Item = f64>) -> f64 {
    let mut result = f64::NEG_INFINITY;
    for value in values {
        if value.is_nan() {
            return value;
        }
        result = result.max(value);
    }
    result
}

/// Stable ascending order with NaN last.
fn sort_simplex<const N: usize>(simplex: &mut [[f64; N]], values: &mut [f64]) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (values[a], values[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => x.partial_cmp(&y).unwrap(),
        }
    });
    let sorted_simplex: Vec<[f64; N]> = order.iter().map(|&i| simplex[i]).collect();
    let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    simplex.copy_from_slice(&sorted_simplex);
    values.copy_from_slice(&sorted_values);
}

/// Counts evaluations and refuses those beyond the budget.
struct Budget<F> {
    function: F,
    calls: usize,
    max_calls: usize,
}

impl<F> Budget<F> {
    fn call<const N: usize>(&mut self, x: &[f64; N]) -> Option<f64>
    where
        F: FnMut(&[f64; N]) -> f64,
    {
        if self.calls >= self.max_calls {
            return None;
        }
        self.calls += 1;
        Some((self.function)(x))
    }
}

fn combine<const N: usize>(a: f64, x: &[f64; N], b: f64, y: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|k| a * x[k] - b * y[k])
}

/// One iteration of the simplex search; `None` when the evaluation budget ran out.
fn nelder_mead_step<const N: usize, F: FnMut(&[f64; N]) -> f64>(
    budget: &mut Budget<F>,
    simplex: &mut [[f64; N]],
    values: &mut [f64],
) -> Option<()> {
    let (rho, chi, psi, sigma) = (
        NELDER_MEAD_REFLECTION,
        NELDER_MEAD_EXPANSION,
        NELDER_MEAD_CONTRACTION,
        NELDER_MEAD_SHRINK,
    );
    let mut sum = [0.0; N];
    for vertex in &simplex[..N] {
        for k in 0..N {
            sum[k] += vertex[k];
        }
    }
    let xbar: [f64; N] = std::array::from_fn(|k| sum[k] / N as f64);
    let worst = simplex[N];
    let xr = combine(1.0 + rho, &xbar, rho, &worst);
    let fxr = budget.call(&xr)?;
    if fxr < values[0] {
        let xe = combine(1.0 + rho * chi, &xbar, rho * chi, &worst);
        let fxe = budget.call(&xe)?;
        if fxe < fxr {
            simplex[N] = xe;
            values[N] = fxe;
        } else {
            simplex[N] = xr;
            values[N] = fxr;
        }
    } else if fxr < values[N - 1] {
        simplex[N] = xr;
        values[N] = fxr;
    } else {
        let shrink = if fxr < values[N] {
            let xc = combine(1.0 + psi * rho, &xbar, psi * rho, &worst);
            let fxc = budget.call(&xc)?;
            if fxc <= fxr {
                simplex[N] = xc;
                values[N] = fxc;
                false
            } else {
                true
            }
        } else {
            let xcc: [f64; N] = std::array::from_fn(|k| (1.0 - psi) * xbar[k] + psi * worst[k]);
            let fxcc = budget.call(&xcc)?;
            if fxcc < values[N] {
                simplex[N] = xcc;
                values[N] = fxcc;
                false
            } else {
                true
            }
        };
        if shrink {
            let best = simplex[0];
            for j in 1..=N {
                simplex[j] = std::array::from_fn(|k| best[k] + sigma * (simplex[j][k] - best[k]));
                values[j] = budget.call(&simplex[j])?;
            }
        }
    }
    Some(())
}

/// Unconstrained Nelder-Mead minimisation with the standard coefficients, an initial simplex
/// of 5 % steps (0.00025 for zero coordinates), stopping when the simplex spans at most 1e-4
/// in every coordinate and in function value, or after 200 evaluations per dimension.
pub fn nelder_mead<const N: usize>(
    function: impl FnMut(&[f64; N]) -> f64,
    x0: [f64; N],
) -> [f64; N] {
    let max_calls = N * NELDER_MEAD_EVALUATIONS_PER_DIMENSION;
    let mut budget = Budget {
        function,
        calls: 0,
        max_calls,
    };
    let mut simplex = vec![x0; N + 1];
    for k in 0..N {
        simplex[k + 1][k] = if x0[k] != 0.0 {
            (1.0 + NELDER_MEAD_NONZERO_STEP) * x0[k]
        } else {
            NELDER_MEAD_ZERO_STEP
        };
    }
    let mut values = vec![f64::INFINITY; N + 1];
    for k in 0..=N {
        match budget.call(&simplex[k]) {
            Some(value) => values[k] = value,
            None => break,
        }
    }
    sort_simplex(&mut simplex, &mut values);
    let mut iterations = 1;
    while budget.calls < max_calls && iterations < max_calls {
        let best = simplex[0];
        let x_spread = nan_aware_max(
            simplex[1..]
                .iter()
                .flat_map(|vertex| (0..N).map(move |k| (vertex[k] - best[k]).abs())),
        );
        let f_spread = nan_aware_max(values[1..].iter().map(|&v| (values[0] - v).abs()));
        if x_spread <= NELDER_MEAD_X_TOLERANCE && f_spread <= NELDER_MEAD_F_TOLERANCE {
            break;
        }
        let completed = nelder_mead_step(&mut budget, &mut simplex, &mut values).is_some();
        if completed {
            iterations += 1;
        }
        sort_simplex(&mut simplex, &mut values);
    }
    simplex[0]
}
