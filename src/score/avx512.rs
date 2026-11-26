//! AVX-512F ports of the NEON score kernels.

#[cfg(target_arch = "x86_64")]
use numpy::ndarray::{Array1, Array2};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const LANES: usize = 16;

/// # Safety
/// The caller must ensure the CPU supports the `avx512f` target feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn axis_log_dot_product_avx512(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);
    let simd_width_cols = (n_cols / LANES) * LANES;
    let one = _mm512_set1_ps(1.0);

    for i in 0..n_rows {
        let weight_vec = _mm512_set1_ps(weights[i]);

        let mut j = 0;
        while j < simd_width_cols {
            let data_vec = _mm512_loadu_ps(array.as_ptr().add(i * n_cols + j));
            let val_plus_one = _mm512_add_ps(data_vec, one);
            let log_approx = fast_log_approx_avx512(val_plus_one);
            let current = _mm512_loadu_ps(result.as_ptr().add(j));
            let new_result = _mm512_fmadd_ps(log_approx, weight_vec, current);
            _mm512_storeu_ps(result.as_mut_ptr().add(j), new_result);
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

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn fast_log_approx_avx512(x: __m512) -> __m512 {
    const LN2_F32: f32 = std::f32::consts::LN_2;

    let x_bits = _mm512_castps_si512(x);

    let exp_mask = _mm512_set1_epi32(0xFF);
    let bias = _mm512_set1_epi32(127);
    let shifted = _mm512_srli_epi32::<23>(x_bits);
    let masked = _mm512_and_si512(shifted, exp_mask);
    let exponent = _mm512_sub_epi32(masked, bias);

    let exponent_f32 = _mm512_cvtepi32_ps(exponent);
    let ln2_vec = _mm512_set1_ps(LN2_F32);
    let exponent_part = _mm512_mul_ps(exponent_f32, ln2_vec);

    let mantissa_mask = _mm512_set1_epi32(0x7FFFFF);
    let exponent_127 = _mm512_set1_epi32(127 << 23);
    let mantissa_bits = _mm512_and_si512(x_bits, mantissa_mask);
    let mantissa_with_exp = _mm512_or_si512(mantissa_bits, exponent_127);
    let y = _mm512_sub_ps(_mm512_castsi512_ps(mantissa_with_exp), _mm512_set1_ps(1.0));

    _mm512_add_ps(exponent_part, y)
}

/// # Safety
/// The caller must ensure the CPU supports the `avx512f` target feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn axis_sqrt_dot_product_avx512(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);
    let simd_width_cols = (n_cols / LANES) * LANES;
    let zero = _mm512_setzero_ps();
    let epsilon = _mm512_set1_ps(1e-10);

    for i in 0..n_rows {
        let weight_vec = _mm512_set1_ps(weights[i]);

        let mut j = 0;
        while j < simd_width_cols {
            let data_vec = _mm512_loadu_ps(array.as_ptr().add(i * n_cols + j));
            let data_pos = _mm512_max_ps(data_vec, zero);

            let x_safe = _mm512_add_ps(data_pos, epsilon);
            let rsqrt_est = _mm512_rsqrt14_ps(x_safe);
            let sqrt_val = _mm512_mul_ps(data_pos, rsqrt_est);
            // Mask: lanes where x <= eps → zero.
            let near_zero = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(epsilon, data_pos);
            let sqrt_approx = _mm512_mask_blend_ps(near_zero, sqrt_val, zero);

            let current = _mm512_loadu_ps(result.as_ptr().add(j));
            let new_result = _mm512_fmadd_ps(sqrt_approx, weight_vec, current);
            _mm512_storeu_ps(result.as_mut_ptr().add(j), new_result);
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
