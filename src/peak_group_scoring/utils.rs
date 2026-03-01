use crate::constants::FragmentType;
use numpy::ndarray::{Array1, Array2};
use std::f32;

/// Filter out rows that contain only zeros
/// Returns a new Array2 with only non-zero rows
pub fn filter_non_zero(array: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = array.dim();

    // Find indices of rows that have at least one non-zero value
    let non_zero_rows: Vec<usize> = (0..rows)
        .filter(|&i| {
            let row = array.row(i);
            row.iter().any(|&val| val != 0.0)
        })
        .collect();

    if non_zero_rows.is_empty() {
        // Return empty array with same number of columns if all rows are zero
        Array2::zeros((0, cols))
    } else {
        // Create new array with only non-zero rows
        let mut filtered = Array2::zeros((non_zero_rows.len(), cols));
        for (new_idx, &old_idx) in non_zero_rows.iter().enumerate() {
            filtered.row_mut(new_idx).assign(&array.row(old_idx));
        }
        filtered
    }
}

/// Calculate the median along axis 0 (first axis) of a 2D array
/// Works with any input array - caller can filter using filter_non_zero if needed
/// Returns zeros for all columns if array has no rows
/// Similar to np.median(array, axis=0) in NumPy
pub fn median_axis_0(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();

    // If no rows exist, return zeros
    if rows == 0 {
        return vec![0.0; cols];
    }

    let mut result = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut column_values: Vec<f32> = Vec::with_capacity(rows);
        for row in 0..rows {
            column_values.push(array[[row, col]]);
        }

        // Sort the column values to find median
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if rows % 2 == 0 {
            // Even number of elements: average of two middle values
            let mid = rows / 2;
            (column_values[mid - 1] + column_values[mid]) / 2.0
        } else {
            // Odd number of elements: middle value
            column_values[rows / 2]
        };

        result.push(median);
    }

    result
}

/// Calculate normalized intensity profiles from dense array.
/// Similar to normalize_profiles in Python
pub fn normalize_profiles(intensity_slice: &Array2<f32>, center_dilations: usize) -> Array2<f32> {
    let (rows, cols) = intensity_slice.dim();
    let center_idx = cols / 2;

    // Calculate mean center intensity for each row
    let mut center_intensity = Vec::with_capacity(rows);

    for i in 0..rows {
        let start_idx = center_idx.saturating_sub(center_dilations);
        let end_idx = std::cmp::min(center_idx + center_dilations + 1, cols);

        let mut sum = 0.0;
        let mut count = 0;

        for j in start_idx..end_idx {
            sum += intensity_slice[[i, j]];
            count += 1;
        }

        let mean_intensity = if count > 0 { sum / count as f32 } else { 0.0 };
        center_intensity.push(mean_intensity);
    }

    // Create normalized output array, initialized to zeros
    let mut normalized = Array2::zeros((rows, cols));

    // Only normalize profiles where center intensity > 0
    for i in 0..rows {
        if center_intensity[i] > 0.0 {
            for j in 0..cols {
                normalized[[i, j]] = intensity_slice[[i, j]] / center_intensity[i];
            }
        }
    }

    normalized
}

/// Calculate correlation between median profile and each row of dense_xic
/// Returns zero where no safe correlation can be calculated
pub fn correlation_axis_0(median_profile: &[f32], dense_xic: &Array2<f32>) -> Vec<f32> {
    let (rows, _cols) = dense_xic.dim();
    let mut correlations = Vec::with_capacity(rows);

    for row in 0..rows {
        let row_data: Vec<f32> = dense_xic.row(row).to_vec();
        let correlation = calculate_correlation_safe(median_profile, &row_data);
        correlations.push(correlation);
    }

    correlations
}

/// Calculate correlation between two arrays safely
/// Returns 0.0 if correlation cannot be calculated safely
pub fn calculate_correlation_safe(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    // Check for all zeros or constant values
    let x_sum: f32 = x.iter().sum();
    let y_sum: f32 = y.iter().sum();

    if x_sum == 0.0 || y_sum == 0.0 {
        return 0.0;
    }

    // Check for constant values (zero variance)
    let x_mean = x_sum / x.len() as f32;
    let y_mean = y_sum / y.len() as f32;

    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    let mut covariance = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;

        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
        covariance += x_diff * y_diff;
    }

    // Check for zero variance (constant values)
    if x_variance == 0.0 || y_variance == 0.0 {
        return 0.0;
    }

    // Calculate correlation coefficient
    let correlation = covariance / (f32::sqrt(x_variance) * f32::sqrt(y_variance));

    // Check for NaN or infinite values
    if correlation.is_nan() || correlation.is_infinite() {
        panic!("correlation.is_nan() || correlation.is_infinite()");
    }

    // Clamp to valid range [-1, 1]
    correlation.clamp(-1.0, 1.0)
}

/// Calculate correlation between two f32 slices
/// Returns 0.0 if correlation cannot be calculated safely
pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    calculate_correlation_safe(x, y)
}

/// Naive hyperscore: straightforward loop with direct factorial computation.
pub fn calculate_hyperscore_naive(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
) -> f32 {
    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut sum_b = 0.0f32;
    let mut sum_y = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }
        if fragment_types[i] == FragmentType::B {
            n_b += 1;
            sum_b += fragment_intensities[i];
        } else if fragment_types[i] == FragmentType::Y {
            n_y += 1;
            sum_y += fragment_intensities[i];
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    let mut fb = 1.0f64;
    for k in 2..=n_b {
        fb *= k as f64;
    }
    let mut fy = 1.0f64;
    for k in 2..=n_y {
        fy *= k as f64;
    }

    let hs = fb.ln() as f32
        + fy.ln() as f32
        + if sum_b > 0.0 { sum_b.ln() } else { 0.0 }
        + if sum_y > 0.0 { sum_y.ln() } else { 0.0 };

    if hs.is_finite() {
        hs
    } else {
        0.0
    }
}

/// Calculate hyperscore with optional per-fragment weights
pub fn calculate_hyperscore_weighted(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    if fragment_types.len() != fragment_intensities.len()
        || fragment_types.len() != matched_mask.len()
    {
        return 0.0;
    }

    if let Some(w) = weights {
        if w.len() != fragment_types.len() {
            return 0.0;
        }
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut weighted_sum_b = 0.0f32;
    let mut weighted_sum_y = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }

        let weight = weights.map(|w| w[i]).unwrap_or(1.0);
        let weighted_intensity = fragment_intensities[i] * weight;

        match fragment_types[i] {
            FragmentType::B => {
                // b-ion
                n_b += 1;
                weighted_sum_b += weighted_intensity;
            }
            FragmentType::Y => {
                // y-ion
                n_y += 1;
                weighted_sum_y += weighted_intensity;
            }
            _ => {
                // Other fragment types not used in hyperscore
            }
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    // Calculate factorial using gamma function: n! = Γ(n+1)
    let factorial_b = if n_b > 0 {
        gamma_ln(n_b as f32 + 1.0)
    } else {
        0.0
    };
    let factorial_y = if n_y > 0 {
        gamma_ln(n_y as f32 + 1.0)
    } else {
        0.0
    };

    // Calculate hyperscore: log(Nb! * Ny! * weighted_sum_b * weighted_sum_y)
    // Don't use .max(0.0) on ln() as it can make valid small values become 0
    let ln_sum_b = if weighted_sum_b > 0.0 {
        weighted_sum_b.ln()
    } else {
        0.0
    };
    let ln_sum_y = if weighted_sum_y > 0.0 {
        weighted_sum_y.ln()
    } else {
        0.0
    };

    let hyperscore = factorial_b + factorial_y + ln_sum_b + ln_sum_y;

    if hyperscore.is_finite() {
        hyperscore
    } else {
        0.0
    }
}

/// Calculate standard hyperscore similar to X! Tandem and MSFragger
///
/// This is a wrapper around calculate_hyperscore_weighted with no weights
pub fn calculate_hyperscore(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
) -> f32 {
    calculate_hyperscore_weighted(fragment_types, fragment_intensities, matched_mask, None)
}

/// Natural logarithm of gamma function using Stirling's approximation
/// For factorial calculation: ln(n!) = ln(Γ(n+1))
fn gamma_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    if (x - 1.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(1)) = ln(0!) = ln(1) = 0
    }

    if (x - 2.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(2)) = ln(1!) = ln(1) = 0
    }

    // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    let ln_2pi = 1.837_877_f32;
    (x - 0.5) * x.ln() - x + 0.5 * ln_2pi
}

/// ln(n!) computed by iterative multiplication then a single ln call.
/// Exact for all representable n! in f64, then cast to f32.
fn ln_factorial_scalar(n: u32) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let mut product = 1.0_f64;
    for i in 2..=n {
        product *= i as f64;
    }
    product.ln() as f32
}

/// ln(n!) from a precomputed lookup table.
/// Covers n=0..64; panics on n>64 (fragments never exceed ~48).
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn ln_factorial_lut(n: u32) -> f32 {
    const LN_FACTORIAL: [f32; 65] = [
        0.0, 0.0, 0.693_147, 1.791_759, 3.178_054, 4.787_492, 6.579_251, 8.525_162, 10.604_602,
        12.801_827, 15.104_413, 17.502_308, 19.987_214, 22.552_164, 25.191_221, 27.899_271,
        30.671_86, 33.505_074, 36.395_445, 39.339_884, 42.335_617, 45.380_14, 48.471_18, 51.606_72,
        54.784_73, 58.003_605, 61.261_7, 64.557_54, 67.889_74, 71.257_04, 74.658_24, 78.092_22,
        81.557_96, 85.054_47, 88.580_83, 92.136_17, 95.719_67, 99.330_61, 102.968_18, 106.631_76,
        110.320_64, 114.034_21, 117.771_88, 121.533_08, 125.317_27, 129.123_95, 132.952_57,
        136.802_72, 140.673_92, 144.565_74, 148.477_77, 152.409_59, 156.360_77, 160.330_91,
        164.319_61, 168.326_47, 172.351_1, 176.393_12, 180.452_14, 184.527_8, 188.619_74,
        192.727_64, 196.851_14, 200.989_92, 205.143_64,
    ];
    LN_FACTORIAL[n as usize]
}

/// Hyperscore using iterative scalar factorial (no Stirling approximation).
pub fn calculate_hyperscore_scalar(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    hyperscore_core(
        fragment_types,
        fragment_intensities,
        matched_mask,
        weights,
        ln_factorial_scalar,
    )
}

/// Hyperscore using precomputed lookup table for ln(n!).
pub fn calculate_hyperscore_lut(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    hyperscore_core(
        fragment_types,
        fragment_intensities,
        matched_mask,
        weights,
        ln_factorial_lut,
    )
}

/// Shared hyperscore logic parameterized by the ln(n!) function.
fn hyperscore_core(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
    ln_factorial_fn: fn(u32) -> f32,
) -> f32 {
    if fragment_types.len() != fragment_intensities.len()
        || fragment_types.len() != matched_mask.len()
    {
        return 0.0;
    }

    if let Some(w) = weights {
        if w.len() != fragment_types.len() {
            return 0.0;
        }
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut weighted_sum_b = 0.0f32;
    let mut weighted_sum_y = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }

        let weight = weights.map(|w| w[i]).unwrap_or(1.0);
        let weighted_intensity = fragment_intensities[i] * weight;

        match fragment_types[i] {
            FragmentType::B => {
                n_b += 1;
                weighted_sum_b += weighted_intensity;
            }
            FragmentType::Y => {
                n_y += 1;
                weighted_sum_y += weighted_intensity;
            }
            _ => {}
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    let factorial_b = if n_b > 0 { ln_factorial_fn(n_b) } else { 0.0 };
    let factorial_y = if n_y > 0 { ln_factorial_fn(n_y) } else { 0.0 };

    let ln_sum_b = if weighted_sum_b > 0.0 {
        weighted_sum_b.ln()
    } else {
        0.0
    };
    let ln_sum_y = if weighted_sum_y > 0.0 {
        weighted_sum_y.ln()
    } else {
        0.0
    };

    let hyperscore = factorial_b + factorial_y + ln_sum_b + ln_sum_y;

    if hyperscore.is_finite() {
        hyperscore
    } else {
        0.0
    }
}

/// Calculate longest continuous b and y ion series scores
/// Returns (longest_b_series, longest_y_series) based on fragment_number values
/// Handles fragment numbers in any order by sorting internally
pub fn calculate_longest_ion_series(
    fragment_types: &[u8],
    fragment_numbers: &[u8],
    matched_mask: &[bool],
) -> (u8, u8) {
    if fragment_types.len() != matched_mask.len() || fragment_types.len() != fragment_numbers.len()
    {
        return (0, 0);
    }

    // Collect matched b and y ions with their fragment numbers
    let mut b_ions: Vec<u8> = Vec::new();
    let mut y_ions: Vec<u8> = Vec::new();

    for i in 0..fragment_types.len() {
        if matched_mask[i] {
            match fragment_types[i] {
                FragmentType::B => b_ions.push(fragment_numbers[i]),
                FragmentType::Y => y_ions.push(fragment_numbers[i]),
                _ => {}
            }
        }
    }

    // Helper function to find longest continuous sequence
    let find_longest_sequence = |mut ions: Vec<u8>| -> u8 {
        if ions.is_empty() {
            return 0;
        }

        ions.sort_unstable();

        let mut max_length = 1u8;
        let mut current_length = 1u8;

        for i in 1..ions.len() {
            if ions[i] == ions[i - 1] + 1 {
                current_length += 1;
                max_length = max_length.max(current_length);
            } else {
                current_length = 1;
            }
        }

        max_length
    };

    let longest_b = find_longest_sequence(b_ions);
    let longest_y = find_longest_sequence(y_ions);

    (longest_b, longest_y)
}

/// Branchless hyperscore with LUT factorial.
///
/// Eliminates branches in the inner loop by using arithmetic masking
/// and splits the weights/no-weights paths to avoid per-element Option checks.
pub fn calculate_hyperscore_branchless(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    let n = fragment_types.len();
    if n != fragment_intensities.len() || n != matched_mask.len() {
        return 0.0;
    }
    if let Some(w) = weights {
        if w.len() != n {
            return 0.0;
        }
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut sum_b = 0.0f32;
    let mut sum_y = 0.0f32;

    match weights {
        None => {
            for i in 0..n {
                let active = (matched_mask[i] as u32) & ((fragment_intensities[i] > 0.0) as u32);
                let is_b = (fragment_types[i] == FragmentType::B) as u32;
                let is_y = (fragment_types[i] == FragmentType::Y) as u32;
                let ab = active & is_b;
                let ay = active & is_y;
                n_b += ab;
                n_y += ay;
                sum_b += fragment_intensities[i] * ab as f32;
                sum_y += fragment_intensities[i] * ay as f32;
            }
        }
        Some(w) => {
            for i in 0..n {
                let active = (matched_mask[i] as u32) & ((fragment_intensities[i] > 0.0) as u32);
                let is_b = (fragment_types[i] == FragmentType::B) as u32;
                let is_y = (fragment_types[i] == FragmentType::Y) as u32;
                let ab = active & is_b;
                let ay = active & is_y;
                let wi = fragment_intensities[i] * w[i];
                n_b += ab;
                n_y += ay;
                sum_b += wi * ab as f32;
                sum_y += wi * ay as f32;
            }
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    let factorial_b = if n_b > 0 { ln_factorial_lut(n_b) } else { 0.0 };
    let factorial_y = if n_y > 0 { ln_factorial_lut(n_y) } else { 0.0 };
    let ln_sum_b = if sum_b > 0.0 { sum_b.ln() } else { 0.0 };
    let ln_sum_y = if sum_y > 0.0 { sum_y.ln() } else { 0.0 };

    let hs = factorial_b + factorial_y + ln_sum_b + ln_sum_y;
    if hs.is_finite() {
        hs
    } else {
        0.0
    }
}

/// NEON SIMD hyperscore with branchless mask operations and LUT factorial.
///
/// Processes 8 fragments per iteration using NEON u8 comparison for type
/// classification, sign-extending widening to u32 masks, and vectorized
/// f32 accumulation.
#[cfg(target_arch = "aarch64")]
pub fn calculate_hyperscore_neon(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    use std::arch::aarch64::*;

    let n = fragment_types.len();
    if n != fragment_intensities.len() || n != matched_mask.len() {
        return 0.0;
    }
    if let Some(w) = weights {
        if w.len() != n {
            return 0.0;
        }
    }

    // Sign-extend u8 comparison mask (0x00/0xFF) to full-width u32 mask
    // (0x00000000/0xFFFFFFFF). vmovl_u8 zero-extends which only gives
    // 0x000000FF, breaking AND-mask logic. Sign-extending via s8->s16->s32
    // propagates all-ones correctly.
    #[inline(always)]
    unsafe fn widen_mask_lo(mask_u8: uint8x8_t) -> uint32x4_t {
        use std::arch::aarch64::*;
        let s8 = vreinterpret_s8_u8(mask_u8);
        let s16 = vmovl_s8(s8);
        vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(s16)))
    }

    #[inline(always)]
    unsafe fn widen_mask_hi(mask_u8: uint8x8_t) -> uint32x4_t {
        use std::arch::aarch64::*;
        let s8 = vreinterpret_s8_u8(mask_u8);
        let s16 = vmovl_s8(s8);
        vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(s16)))
    }

    unsafe {
        let b_ref = vdup_n_u8(FragmentType::B);
        let y_ref = vdup_n_u8(FragmentType::Y);
        let zero_u8 = vdup_n_u8(0);
        let zero_f = vdupq_n_f32(0.0);
        let one_u = vdupq_n_u32(1);

        let mut sb0 = vdupq_n_f32(0.0);
        let mut sb1 = vdupq_n_f32(0.0);
        let mut sy0 = vdupq_n_f32(0.0);
        let mut sy1 = vdupq_n_f32(0.0);
        let mut cb = vdupq_n_u32(0);
        let mut cy = vdupq_n_u32(0);

        let type_ptr = fragment_types.as_ptr();
        let int_ptr = fragment_intensities.as_ptr();
        let mask_ptr = matched_mask.as_ptr() as *const u8;

        // 2x-unrolled: process 8 elements per iteration (one full u8x8 register)
        let block8 = n & !7;
        let mut j = 0;
        while j < block8 {
            let t8 = vld1_u8(type_ptr.add(j));
            let m8 = vld1_u8(mask_ptr.add(j));

            let is_b = vceq_u8(t8, b_ref);
            let is_y = vceq_u8(t8, y_ref);
            let matched = vcgt_u8(m8, zero_u8);

            let ib_lo = widen_mask_lo(is_b);
            let ib_hi = widen_mask_hi(is_b);
            let iy_lo = widen_mask_lo(is_y);
            let iy_hi = widen_mask_hi(is_y);
            let mt_lo = widen_mask_lo(matched);
            let mt_hi = widen_mask_hi(matched);

            let i_lo = vld1q_f32(int_ptr.add(j));
            let i_hi = vld1q_f32(int_ptr.add(j + 4));

            let gt_lo = vcgtq_f32(i_lo, zero_f);
            let gt_hi = vcgtq_f32(i_hi, zero_f);

            let act_lo = vandq_u32(mt_lo, gt_lo);
            let act_hi = vandq_u32(mt_hi, gt_hi);

            let ab_lo = vandq_u32(act_lo, ib_lo);
            let ab_hi = vandq_u32(act_hi, ib_hi);
            let ay_lo = vandq_u32(act_lo, iy_lo);
            let ay_hi = vandq_u32(act_hi, iy_hi);

            let (w_lo, w_hi) = match weights {
                Some(w) => (
                    vmulq_f32(i_lo, vld1q_f32(w.as_ptr().add(j))),
                    vmulq_f32(i_hi, vld1q_f32(w.as_ptr().add(j + 4))),
                ),
                None => (i_lo, i_hi),
            };

            sb0 = vaddq_f32(sb0, vbslq_f32(ab_lo, w_lo, zero_f));
            sb1 = vaddq_f32(sb1, vbslq_f32(ab_hi, w_hi, zero_f));
            sy0 = vaddq_f32(sy0, vbslq_f32(ay_lo, w_lo, zero_f));
            sy1 = vaddq_f32(sy1, vbslq_f32(ay_hi, w_hi, zero_f));

            cb = vaddq_u32(
                cb,
                vaddq_u32(vandq_u32(ab_lo, one_u), vandq_u32(ab_hi, one_u)),
            );
            cy = vaddq_u32(
                cy,
                vaddq_u32(vandq_u32(ay_lo, one_u), vandq_u32(ay_hi, one_u)),
            );

            j += 8;
        }

        // Merge the two accumulator pairs
        sb0 = vaddq_f32(sb0, sb1);
        sy0 = vaddq_f32(sy0, sy1);

        // Handle a remaining 4-element chunk
        if j + 4 <= n {
            let mut tbuf = [0u8; 8];
            std::ptr::copy_nonoverlapping(type_ptr.add(j), tbuf.as_mut_ptr(), 4);
            let t4 = vld1_u8(tbuf.as_ptr());

            let mut mbuf = [0u8; 8];
            std::ptr::copy_nonoverlapping(mask_ptr.add(j), mbuf.as_mut_ptr(), 4);
            let m4 = vld1_u8(mbuf.as_ptr());

            let is_b4 = widen_mask_lo(vceq_u8(t4, b_ref));
            let is_y4 = widen_mask_lo(vceq_u8(t4, y_ref));
            let mt4 = widen_mask_lo(vcgt_u8(m4, zero_u8));

            let i4 = vld1q_f32(int_ptr.add(j));
            let gt4 = vcgtq_f32(i4, zero_f);
            let act4 = vandq_u32(mt4, gt4);
            let ab4 = vandq_u32(act4, is_b4);
            let ay4 = vandq_u32(act4, is_y4);

            let w4 = match weights {
                Some(w) => vmulq_f32(i4, vld1q_f32(w.as_ptr().add(j))),
                None => i4,
            };

            sb0 = vaddq_f32(sb0, vbslq_f32(ab4, w4, zero_f));
            sy0 = vaddq_f32(sy0, vbslq_f32(ay4, w4, zero_f));
            cb = vaddq_u32(cb, vandq_u32(ab4, one_u));
            cy = vaddq_u32(cy, vandq_u32(ay4, one_u));

            j += 4;
        }

        // Horizontal reductions
        let mut n_b = vaddvq_u32(cb);
        let mut n_y = vaddvq_u32(cy);
        let mut sum_b = vaddvq_f32(sb0);
        let mut sum_y = vaddvq_f32(sy0);

        // Scalar tail (0-3 remaining elements)
        while j < n {
            let ft = *type_ptr.add(j);
            let fi = *int_ptr.add(j);
            let fm = *mask_ptr.add(j) != 0;
            if fm && fi > 0.0 {
                let w = weights.map(|w| *w.as_ptr().add(j)).unwrap_or(1.0);
                let wi = fi * w;
                if ft == FragmentType::B {
                    n_b += 1;
                    sum_b += wi;
                } else if ft == FragmentType::Y {
                    n_y += 1;
                    sum_y += wi;
                }
            }
            j += 1;
        }

        if n_b == 0 && n_y == 0 {
            return 0.0;
        }

        let factorial_b = if n_b > 0 { ln_factorial_lut(n_b) } else { 0.0 };
        let factorial_y = if n_y > 0 { ln_factorial_lut(n_y) } else { 0.0 };
        let ln_sum_b = if sum_b > 0.0 { sum_b.ln() } else { 0.0 };
        let ln_sum_y = if sum_y > 0.0 { sum_y.ln() } else { 0.0 };

        let hs = factorial_b + factorial_y + ln_sum_b + ln_sum_y;
        if hs.is_finite() {
            hs
        } else {
            0.0
        }
    }
}

/// Calculate hyperscore with inverse mass error weighting
///
/// Similar to standard hyperscore but weights each matched fragment by 1/(|mass_error| + 0.1)
/// Excludes fragments with zero observed intensity (sum across all cycles)
///
/// hyperscore = log(Nb! * Ny! * sum(Ib,i * w_i) * sum(Iy,i * w_i))
/// where w_i = 1/(|mass_error_i| + 0.1)
pub fn calculate_hyperscore_inverse_mass_error(
    fragment_types: &[u8],
    fragment_intensities: &[f32], // Observed intensities (sum across cycles)
    matched_mask: &[bool],
    mass_errors: &[f32], // Mass errors in ppm
) -> f32 {
    if fragment_types.len() != mass_errors.len() {
        return 0.0;
    }

    // Calculate inverse mass error weights: 1/(|mass_error| + 0.1)
    let weights: Vec<f32> = mass_errors
        .iter()
        .map(|&error| 1.0 / (error.abs() + 0.1))
        .collect();

    calculate_hyperscore_weighted(
        fragment_types,
        fragment_intensities,
        matched_mask,
        Some(&weights),
    )
}

/// Calculate total intensity for a specific ion series
///
/// Sums all observed intensities for fragments of the specified type
/// that have a matched intensity (intensity > 0 and matched_mask = true)
pub fn intensity_ion_series(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    target_fragment_type: u8,
) -> f32 {
    let n_fragments = fragment_types.len();
    if n_fragments != fragment_intensities.len() || n_fragments != matched_mask.len() {
        return 0.0;
    }

    let mut total_intensity = 0.0;

    for i in 0..n_fragments {
        if matched_mask[i]
            && fragment_intensities[i] > 0.0
            && fragment_types[i] == target_fragment_type
        {
            total_intensity += fragment_intensities[i];
        }
    }

    total_intensity
}

/// Calculate dot product between two slices of equal length
///
/// Returns the sum of element-wise products: sum(a_i * b_i)
/// Returns 0.0 if slices have different lengths or are empty
pub fn calculate_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Calculate Full Width at Half Maximum (FWHM) for retention time from an XIC profile
///
/// Finds the maximum peak in the XIC slice and calculates the FWHM by finding points
/// where the intensity is half of the maximum. The slice should be centered at the maximum
/// and have an odd number of elements.
///
/// Parameters:
/// - xic_profile: Intensity profile (median profile across fragments)
/// - cycle_start_idx: Starting cycle index in the RT array
/// - cycle_stop_idx: Ending cycle index in the RT array (exclusive)
/// - rt_values: Array of retention time values
///
/// Returns:
/// - FWHM in retention time units, or 0.0 if cannot be calculated
pub fn calculate_fwhm_rt(
    xic_profile: &[f32],
    cycle_start_idx: usize,
    rt_values: &Array1<f32>,
) -> f32 {
    if xic_profile.is_empty() {
        return 0.0;
    }

    let half_size = xic_profile.len() / 2;
    let center_intensity = xic_profile[half_size];

    for i in 0..half_size {
        let mean_intensity = (xic_profile[half_size - i] + xic_profile[half_size + i]) / 2.0;

        if mean_intensity <= center_intensity / 2.0 {
            let left_rt = rt_values[cycle_start_idx + half_size - i];
            let right_rt = rt_values[cycle_start_idx + half_size + i];
            return right_rt - left_rt;
        }
    }

    0.0
}
