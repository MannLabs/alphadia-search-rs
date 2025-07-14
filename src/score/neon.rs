#[cfg(target_arch = "aarch64")]
use numpy::ndarray::{Array1, Array2};

/// NEON-optimized implementation of log-dot-product operation for aarch64
#[cfg(target_arch = "aarch64")]
pub fn axis_log_dot_product_neon(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let (n_rows, n_cols) = array.dim();

    // Check that the number of rows matches the number of weights
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);

    // Process SIMD blocks of 4 elements
    const SIMD_WIDTH: usize = 4;
    let simd_width_cols = (n_cols / SIMD_WIDTH) * SIMD_WIDTH;

    // Process each row, then column in blocks of 4
    for i in 0..n_rows {
        let weight = weights[i];
        let weight_vec = unsafe { vdupq_n_f32(weight) };

        let mut j = 0;
        while j < simd_width_cols {
            // Load 4 elements
            let data_vec = unsafe { vld1q_f32(array.as_ptr().add(i * n_cols + j)) };

            // Add 1.0 to avoid log(0)
            let one_vec = unsafe { vdupq_n_f32(1.0) };
            let val_plus_one = unsafe { vaddq_f32(data_vec, one_vec) };

            // Fast log approximation for NEON
            let log_approx = unsafe { fast_log_approx_neon(val_plus_one) };

            // Multiply log by weight
            let weighted_log = unsafe { vmulq_f32(log_approx, weight_vec) };

            // Load current results
            let current_result = unsafe { vld1q_f32(result.as_ptr().add(j)) };

            // Add to results
            let new_result = unsafe { vaddq_f32(current_result, weighted_log) };

            // Store results
            unsafe { vst1q_f32(result.as_mut_ptr().add(j), new_result) };

            j += SIMD_WIDTH;
        }

        // Handle remaining elements
        for j in simd_width_cols..n_cols {
            let val = (array[[i, j]] + 1.0).ln();
            result[j] += val * weight;
        }
    }

    result
}

/// NEON fast logarithm approximation
#[cfg(target_arch = "aarch64")]
unsafe fn fast_log_approx_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        uint32x4_t, vaddq_f32, vandq_u32, vcvtq_f32_u32, vdupq_n_f32, vdupq_n_u32, vmulq_f32,
        vorrq_u32, vreinterpretq_f32_u32, vreinterpretq_u32_f32, vshrq_n_u32, vsubq_f32, vsubq_u32,
    };

    // Constants for the approximation
    const LN2_F32: f32 = std::f32::consts::LN_2;

    // IEEE-754 floating-point bit structure: sign(1) | exponent(8) | mantissa(23)
    // ln(2^e * 1.m) = e*ln(2) + ln(1.m)

    // Get bits of x
    let x_bits: uint32x4_t = vreinterpretq_u32_f32(x);

    // Extract exponent: ((x_bits >> 23) & 0xFF) - 127
    let exp_mask = vdupq_n_u32(0xFF);
    let bias = vdupq_n_u32(127);
    let exponent = vsubq_u32(vandq_u32(vshrq_n_u32(x_bits, 23), exp_mask), bias);

    // Convert exponent to float and multiply by ln(2)
    let exponent_f32 = vcvtq_f32_u32(exponent);
    let ln2_vec = vdupq_n_f32(LN2_F32);
    let exponent_part = vmulq_f32(exponent_f32, ln2_vec);

    // For the mantissa part, we'll use a simple approximation
    // Extract mantissa bits and create a float between 1.0 and 2.0
    let mantissa_mask = vdupq_n_u32(0x7FFFFF);
    let exponent_127 = vdupq_n_u32(127 << 23);

    // Isolate mantissa bits and OR with exponent 127 (creates a float in [1,2))
    let mantissa_with_exp = vorrq_u32(vandq_u32(x_bits, mantissa_mask), exponent_127);

    // Convert to float and subtract 1.0 to get a value in [0,1)
    let y = vsubq_f32(vreinterpretq_f32_u32(mantissa_with_exp), vdupq_n_f32(1.0));

    // Simple approximation for ln(1+y) ≈ y for speed
    // Better approximations could use: y - y²/2 + y³/3, etc
    // Combine exponent and mantissa parts
    vaddq_f32(exponent_part, y)
}

/// NEON-optimized implementation of sqrt-dot-product operation for aarch64
#[cfg(target_arch = "aarch64")]
pub fn axis_sqrt_dot_product_neon(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmulq_f32, vst1q_f32};

    let (n_rows, n_cols) = array.dim();

    // Check that the number of rows matches the number of weights
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result: Array1<f32> = Array1::zeros(n_cols);

    // Process SIMD blocks of 4 elements
    const SIMD_WIDTH: usize = 4;
    let simd_width_cols = (n_cols / SIMD_WIDTH) * SIMD_WIDTH;

    // Process each row, then column in blocks of 4
    for i in 0..n_rows {
        let weight = weights[i];
        let weight_vec = unsafe { vdupq_n_f32(weight) };

        let mut j = 0;
        while j < simd_width_cols {
            // Load 4 elements
            let data_vec = unsafe { vld1q_f32(array.as_ptr().add(i * n_cols + j)) };

            // Ensure values are not negative for sqrt
            let zero_vec = unsafe { vdupq_n_f32(0.0) };
            let data_pos = unsafe { vmaxq_f32(data_vec, zero_vec) };

            // Fast square root approximation for NEON
            let sqrt_approx = unsafe { fast_sqrt_approx_neon(data_pos) };

            // Multiply sqrt by weight
            let weighted_sqrt = unsafe { vmulq_f32(sqrt_approx, weight_vec) };

            // Load current results
            let current_result = unsafe { vld1q_f32(result.as_ptr().add(j)) };

            // Add to results
            let new_result = unsafe { vaddq_f32(current_result, weighted_sqrt) };

            // Store results
            unsafe { vst1q_f32(result.as_mut_ptr().add(j), new_result) };

            j += SIMD_WIDTH;
        }

        // Handle remaining elements
        for j in simd_width_cols..n_cols {
            let val = (array[[i, j]].max(0.0)).sqrt();
            result[j] += val * weight;
        }
    }

    result
}

/// NEON fast square root approximation
#[cfg(target_arch = "aarch64")]
unsafe fn fast_sqrt_approx_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_f32, vbslq_f32, vcgeq_f32, vdupq_n_f32, vmulq_f32, vrsqrteq_f32,
    };

    // Add a small epsilon to prevent division by zero
    let epsilon = vdupq_n_f32(1e-10);
    let x_safe = vaddq_f32(x, epsilon);

    // Check which values are zero or near-zero
    let zero = vdupq_n_f32(0.0);
    let is_zero_mask = vcgeq_f32(epsilon, x); // true if x ≤ epsilon

    // For normal values: sqrt(x) = x * rsqrt(x)
    let rsqrt_estimate = vrsqrteq_f32(x_safe);
    let sqrt_result = vmulq_f32(x, rsqrt_estimate);

    // Use the mask to select: if x is zero/near-zero, use zero, otherwise use sqrt result
    vbslq_f32(is_zero_mask, zero, sqrt_result)
}
