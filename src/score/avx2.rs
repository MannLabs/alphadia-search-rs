//! AVX2 + FMA ports of the NEON score kernels.

#[cfg(target_arch = "x86_64")]
use numpy::ndarray::{Array1, Array2};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const LANES: usize = 8;

/// # Safety
/// The caller must ensure the CPU supports the `avx2` and `fma` target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axis_log_dot_product_avx2(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);
    let simd_width_cols = (n_cols / LANES) * LANES;
    let one = _mm256_set1_ps(1.0);

    for i in 0..n_rows {
        let weight_vec = _mm256_set1_ps(weights[i]);

        let mut j = 0;
        while j < simd_width_cols {
            let data_vec = _mm256_loadu_ps(array.as_ptr().add(i * n_cols + j));
            let val_plus_one = _mm256_add_ps(data_vec, one);
            let log_approx = fast_log_approx_avx2(val_plus_one);
            let current = _mm256_loadu_ps(result.as_ptr().add(j));
            let new_result = _mm256_fmadd_ps(log_approx, weight_vec, current);
            _mm256_storeu_ps(result.as_mut_ptr().add(j), new_result);
            j += LANES;
        }

        let weight = weights[i];
        for j in simd_width_cols..n_cols {
            let val = (array[[i, j]] + 1.0).ln();
            result[j] += val * weight;
        }
    }

    result
}

/// AVX2 fast logarithm approximation matching the NEON one:
/// IEEE-754 split, with ln(1+y) approximated by y where y is the mantissa in [0, 1).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_log_approx_avx2(x: __m256) -> __m256 {
    const LN2_F32: f32 = std::f32::consts::LN_2;

    let x_bits = _mm256_castps_si256(x);

    // ((x_bits >> 23) & 0xFF) - 127
    let exp_mask = _mm256_set1_epi32(0xFF);
    let bias = _mm256_set1_epi32(127);
    let shifted = _mm256_srli_epi32::<23>(x_bits);
    let masked = _mm256_and_si256(shifted, exp_mask);
    let exponent = _mm256_sub_epi32(masked, bias);

    let exponent_f32 = _mm256_cvtepi32_ps(exponent);
    let ln2_vec = _mm256_set1_ps(LN2_F32);
    let exponent_part = _mm256_mul_ps(exponent_f32, ln2_vec);

    // Mantissa reinterpreted as float in [1, 2), then subtract 1.0 → y in [0, 1).
    let mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
    let exponent_127 = _mm256_set1_epi32(127 << 23);
    let mantissa_bits = _mm256_and_si256(x_bits, mantissa_mask);
    let mantissa_with_exp = _mm256_or_si256(mantissa_bits, exponent_127);
    let y = _mm256_sub_ps(_mm256_castsi256_ps(mantissa_with_exp), _mm256_set1_ps(1.0));

    _mm256_add_ps(exponent_part, y)
}

/// # Safety
/// The caller must ensure the CPU supports the `avx2` and `fma` target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axis_sqrt_dot_product_avx2(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);
    let simd_width_cols = (n_cols / LANES) * LANES;
    let zero = _mm256_setzero_ps();
    let epsilon = _mm256_set1_ps(1e-10);

    for i in 0..n_rows {
        let weight_vec = _mm256_set1_ps(weights[i]);

        let mut j = 0;
        while j < simd_width_cols {
            let data_vec = _mm256_loadu_ps(array.as_ptr().add(i * n_cols + j));
            let data_pos = _mm256_max_ps(data_vec, zero);

            // x * rsqrt(x + eps), masked to 0 for x <= eps.
            let x_safe = _mm256_add_ps(data_pos, epsilon);
            let rsqrt_est = _mm256_rsqrt_ps(x_safe);
            let sqrt_val = _mm256_mul_ps(data_pos, rsqrt_est);
            // Mask: lanes where x <= eps → zero (NEON uses vcgeq_f32(eps, x)).
            let near_zero = _mm256_cmp_ps::<_CMP_GE_OQ>(epsilon, data_pos);
            let sqrt_approx = _mm256_andnot_ps(near_zero, sqrt_val);

            let current = _mm256_loadu_ps(result.as_ptr().add(j));
            let new_result = _mm256_fmadd_ps(sqrt_approx, weight_vec, current);
            _mm256_storeu_ps(result.as_mut_ptr().add(j), new_result);
            j += LANES;
        }

        let weight = weights[i];
        for j in simd_width_cols..n_cols {
            let val = (array[[i, j]].max(0.0)).sqrt();
            result[j] += val * weight;
        }
    }

    result
}
