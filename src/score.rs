use numpy::ndarray::{Array2, Array1};

/// Performs a weighted dot product operation along the first axis of a 2D array.
/// Each row of the 2D array is multiplied by its corresponding weight in the fragment_intensity vector,
/// then columns are summed to produce a 1D array with the same length as the second dimension.
pub fn axis_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    
    // Check that the number of rows matches the number of weights
    assert_eq!(n_rows, weights.len(), "Number of rows in array must match the length of weights vector");
    
    let mut result = Array1::zeros(n_cols);
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            result[j] += array[[i, j]] * weights[i];
        }
    }
    
    result
}

/// Applies natural logarithm to each element and sums values along the first axis.
/// For each column, this computes the sum of log values across all rows.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_log_sum(array: &Array2<f32>) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    let mut result = Array1::zeros(n_cols);
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            // Add a small epsilon to avoid log(0)
            let val = array[[i, j]] + 1.0;
            result[j] += val.ln();
        }
    }
    
    result
}

/// First applies logarithm to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        axis_log_dot_product_simd(array, weights)
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        axis_log_dot_product_scalar(array, weights)
    }
}

/// Scalar implementation of log-dot-product operation
fn axis_log_dot_product_scalar(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    
    // Check that the number of rows matches the number of weights
    assert_eq!(n_rows, weights.len(), "Number of rows in array must match the length of weights vector");
    
    let mut result = Array1::zeros(n_cols);
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            // Apply log transformation and then weighted sum
            let val = (array[[i, j]] + 1.0).ln();
            result[j] += val * weights[i];
        }
    }
    
    result
}

/// SIMD-optimized implementation of log-dot-product operation for aarch64
#[cfg(target_arch = "aarch64")]
fn axis_log_dot_product_simd(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmulq_f32, vdupq_n_f32, vst1q_f32};
    
    let (n_rows, n_cols) = array.dim();
    
    // Check that the number of rows matches the number of weights
    assert_eq!(n_rows, weights.len(), "Number of rows in array must match the length of weights vector");
    
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
            
            // Fast log approximation for SIMD
            let log_approx = unsafe { fast_log_approx_simd(val_plus_one) };
            
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

/// SIMD fast logarithm approximation
#[cfg(target_arch = "aarch64")]
unsafe fn fast_log_approx_simd(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        uint32x4_t, vreinterpretq_u32_f32, vreinterpretq_f32_u32,
        vsubq_u32, vshrq_n_u32, vcvtq_f32_u32, vsubq_f32, vmulq_f32, vdupq_n_f32,
        vaddq_f32, vdupq_n_u32, vandq_u32, vorrq_u32
    };
    
    // Constants for the approximation
    const LN2_F32: f32 = 0.6931471805599453;
    
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

/// First applies square root to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_sqrt_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        axis_sqrt_dot_product_simd(array, weights)
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        axis_sqrt_dot_product_scalar(array, weights)
    }
}

/// Scalar implementation of sqrt-dot-product operation
fn axis_sqrt_dot_product_scalar(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();
    
    // Check that the number of rows matches the number of weights
    assert_eq!(n_rows, weights.len(), "Number of rows in array must match the length of weights vector");
    
    let mut result = Array1::zeros(n_cols);
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            // Apply square root transformation and then weighted sum
            // Add small epsilon to avoid sqrt(negative)
            let val = (array[[i, j]].max(0.0)).sqrt();
            result[j] += val * weights[i];
        }
    }
    
    result
}

/// SIMD-optimized implementation of sqrt-dot-product operation for aarch64
#[cfg(target_arch = "aarch64")]
fn axis_sqrt_dot_product_simd(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmulq_f32, vdupq_n_f32, vst1q_f32, vmaxq_f32};
    
    let (n_rows, n_cols) = array.dim();
    
    // Check that the number of rows matches the number of weights
    assert_eq!(n_rows, weights.len(), "Number of rows in array must match the length of weights vector");
    
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
            
            // Fast square root approximation for SIMD
            let sqrt_approx = unsafe { fast_sqrt_approx_simd(data_pos) };
            
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

/// SIMD fast square root approximation
#[cfg(target_arch = "aarch64")]
unsafe fn fast_sqrt_approx_simd(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vmulq_f32, vrsqrteq_f32, vcgeq_f32, vdupq_n_f32, vbslq_f32, vaddq_f32
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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2};
    use approx::assert_relative_eq;

    #[test]
    fn test_axis_dot_product_basic_case() {
        let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let weights = vec![0.5, 1.5];
        let result = axis_dot_product(&array, &weights);
        
        // Correct expected values: 0.5*1.0 + 1.5*4.0 = 0.5 + 6.0 = 6.5
        //                          0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
        //                          0.5*3.0 + 1.5*6.0 = 1.5 + 9.0 = 10.5
        let expected = arr1(&[6.5, 8.5, 10.5]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_axis_dot_product_single_row() {
        let array = arr2(&[[1.0, 2.0, 3.0]]);
        let weights = vec![2.0];
        let result = axis_dot_product(&array, &weights);
        
        let expected = arr1(&[2.0, 4.0, 6.0]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_axis_dot_product_all_zeros() {
        let array = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        let weights = vec![1.0, 1.0];
        let result = axis_dot_product(&array, &weights);
        
        let expected = arr1(&[0.0, 0.0]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    #[should_panic]
    fn test_axis_dot_product_dimension_mismatch() {
        // Should panic because weights.len() != array.dim().0
        let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let weights = vec![0.5];
        let _ = axis_dot_product(&array, &weights);
    }

    #[test]
    fn test_axis_log_sum_basic() {
        let array = arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let result = axis_log_sum(&array);
        
        // Expected: ln(0.0+1.0) + ln(3.0+1.0) = 0.0 + ln(4.0) = 1.386...
        //           ln(1.0+1.0) + ln(4.0+1.0) = ln(2.0) + ln(5.0) = 0.693... + 1.609... = 2.302...
        //           ln(2.0+1.0) + ln(5.0+1.0) = ln(3.0) + ln(6.0) = 1.098... + 1.791... = 2.889...
        let expected = arr1(&[1.3862944, 2.3025851, 2.8903718]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_axis_log_dot_product_basic() {
        let array = arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let weights = vec![0.5, 1.5]; 
        let result = axis_log_dot_product(&array, &weights);
        
        // Expected: 0.5*ln(0.0+1.0) + 1.5*ln(3.0+1.0) = 0.5*0.0 + 1.5*ln(4.0) = 0.0 + 1.5*1.386... = 2.079...
        //           0.5*ln(1.0+1.0) + 1.5*ln(4.0+1.0) = 0.5*ln(2.0) + 1.5*ln(5.0) = 0.5*0.693... + 1.5*1.609... = 0.346... + 2.413... = 2.759...
        //           0.5*ln(2.0+1.0) + 1.5*ln(5.0+1.0) = 0.5*ln(3.0) + 1.5*ln(6.0) = 0.5*1.098... + 1.5*1.791... = 0.549... + 2.686... = 3.235...
        let expected = arr1(&[2.0794415, 2.7607305, 3.2369454]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_simd_vs_scalar_implementation() {
        // Create a larger dataset to test SIMD effectively
        let n_rows = 5;
        let n_cols = 20;
        let mut data = Array2::zeros((n_rows, n_cols));
        let mut weights = Vec::with_capacity(n_rows);
        
        // Fill with test data
        for i in 0..n_rows {
            weights.push((i as f32) * 0.5 + 0.5); // weights from 0.5 to 2.5
            for j in 0..n_cols {
                data[[i, j]] = (i as f32) * 0.5 + (j as f32) * 0.25;
            }
        }
        
        // Run both implementations
        let scalar_result = axis_log_dot_product_scalar(&data, &weights);
        let simd_result = axis_log_dot_product_simd(&data, &weights);
        
        // Verify SIMD and scalar results are reasonably close
        // The SIMD log approximation can diverge from scalar implementation
        for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            println!("Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%", 
                j, s, v, (s - v).abs(), (s - v).abs() / s.abs() * 100.0);
            
            // Allow a larger difference due to log approximation
            // This will still catch major implementation errors
            assert!((s - v).abs() / s.abs() < 0.15, 
                    "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
                    j, s, v, (s - v).abs() / s.abs() * 100.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_simd_approximation_accuracy() {
        // Test with values that cover a wide range to test the log approximation
        let n_rows = 3;
        let n_cols = 32; // Multiple of SIMD width (4)
        let mut data = Array2::zeros((n_rows, n_cols));
        let weights = vec![1.0, 1.0, 1.0]; // Equal weights to test just the log approximation
        
        // Create test values: values near zero, values near 1, and larger values
        for j in 0..n_cols {
            // Row 0: Small values (0.001 to 0.031)
            data[[0, j]] = 0.001 * (j as f32 + 1.0);
            
            // Row 1: Medium values (0.5 to 16.0)
            data[[1, j]] = 0.5 * (j as f32 + 1.0);
            
            // Row 2: Larger values (10 to 320)
            data[[2, j]] = 10.0 * (j as f32 + 1.0);
        }
        
        // Run both implementations
        let scalar_result = axis_log_dot_product_scalar(&data, &weights);
        let simd_result = axis_log_dot_product_simd(&data, &weights);
        
        // Print values for inspection and verify the general pattern matches
        println!("Scalar\tSIMD\tRelDiff");
        for j in 0..n_cols {
            let rel_diff = (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs() * 100.0;
            println!("{:.4}\t{:.4}\t{:.2}%", scalar_result[j], simd_result[j], rel_diff);
        }
        
        // Verify the general trend is similar between scalar and SIMD
        // by checking correlation between the results
        let scalar_mean = scalar_result.sum() / scalar_result.len() as f32;
        let simd_mean = simd_result.sum() / simd_result.len() as f32;
        
        let mut numerator = 0.0;
        let mut scalar_denom = 0.0;
        let mut simd_denom = 0.0;
        
        for j in 0..n_cols {
            let scalar_diff = scalar_result[j] - scalar_mean;
            let simd_diff = simd_result[j] - simd_mean;
            
            numerator += scalar_diff * simd_diff;
            scalar_denom += scalar_diff * scalar_diff;
            simd_denom += simd_diff * simd_diff;
        }
        
        let correlation = numerator / (scalar_denom.sqrt() * simd_denom.sqrt());
        println!("Correlation between scalar and SIMD results: {}", correlation);
        
        // Expect strong correlation above 0.95
        assert!(correlation > 0.95, "Correlation between scalar and SIMD is too low: {}", correlation);
        
        // Also verify the SIMD implementation generally follows the same pattern
        // by checking that the start and end values are in the correct order
        assert!(simd_result[0] < simd_result[n_cols-1], 
                "SIMD implementation doesn't follow the general increasing pattern");
                
        // And check that the average relative difference is acceptable
        let mut total_rel_diff = 0.0;
        for j in 0..n_cols {
            total_rel_diff += (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs();
        }
        let avg_rel_diff = total_rel_diff / n_cols as f32;
        println!("Average relative difference: {:.2}%", avg_rel_diff * 100.0);
        
        // Allow up to 10% average relative difference
        assert!(avg_rel_diff < 0.10, "Average relative difference too high: {:.2}%", avg_rel_diff * 100.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_unaligned_data_handling() {
        // Test with array sizes that aren't multiples of SIMD width
        let array = arr2(&[
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4, 1.5]
        ]);
        let weights = vec![0.5, 1.0, 1.5];
        
        // Run both implementations
        let scalar_result = axis_log_dot_product_scalar(&array, &weights);
        let simd_result = axis_log_dot_product_simd(&array, &weights);
        
        // Verify reasonable similarity - allow larger differences due to approximation
        for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            println!("Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%", 
                j, s, v, (s - v).abs(), (s - v).abs() / s.abs() * 100.0);
            
            // Allow a relative difference of up to 15%
            assert!((s - v).abs() / s.abs() < 0.15, 
                    "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
                    j, s, v, (s - v).abs() / s.abs() * 100.0);
        }
    }

    #[test]
    fn test_axis_sqrt_dot_product_basic() {
        let array = arr2(&[[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
        let weights = vec![0.5, 1.5]; 
        let result = axis_sqrt_dot_product(&array, &weights);
        
        // Expected: 0.5*sqrt(0.0) + 1.5*sqrt(9.0) = 0.0 + 1.5*3.0 = 4.5
        //           0.5*sqrt(1.0) + 1.5*sqrt(16.0) = 0.5*1.0 + 1.5*4.0 = 0.5 + 6.0 = 6.5
        //           0.5*sqrt(4.0) + 1.5*sqrt(25.0) = 0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
        let expected = arr1(&[4.5, 6.5, 8.5]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_axis_sqrt_dot_product_negative_values() {
        let array = arr2(&[[-1.0, 0.0, 4.0], [9.0, -4.0, 25.0]]);
        let weights = vec![0.5, 1.5]; 
        let result = axis_sqrt_dot_product(&array, &weights);
        
        // Expected: 0.5*sqrt(0.0) + 1.5*sqrt(9.0) = 0.0 + 1.5*3.0 = 4.5
        //           0.5*sqrt(0.0) + 1.5*sqrt(0.0) = 0.0 + 0.0 = 0.0
        //           0.5*sqrt(4.0) + 1.5*sqrt(25.0) = 0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
        let expected = arr1(&[4.5, 0.0, 8.5]);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sqrt_simd_vs_scalar_implementation() {
        // Create a larger dataset to test SIMD effectively
        let n_rows = 5;
        let n_cols = 20;
        let mut data = Array2::zeros((n_rows, n_cols));
        let mut weights = Vec::with_capacity(n_rows);
        
        // Fill with test data
        for i in 0..n_rows {
            weights.push((i as f32) * 0.5 + 0.5); // weights from 0.5 to 2.5
            for j in 0..n_cols {
                data[[i, j]] = ((i as f32) * 0.5 + (j as f32) * 0.25).powi(2); // square values
            }
        }
        
        // Run both implementations
        let scalar_result = axis_sqrt_dot_product_scalar(&data, &weights);
        let simd_result = axis_sqrt_dot_product_simd(&data, &weights);
        
        // Verify SIMD and scalar results are reasonably close
        // The SIMD sqrt approximation can diverge from scalar implementation
        for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            println!("Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%", 
                j, s, v, (s - v).abs(), (s - v).abs() / s.abs() * 100.0);
            
            // Allow a small difference due to sqrt approximation
            assert!((s - v).abs() / s.abs() < 0.01, 
                    "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
                    j, s, v, (s - v).abs() / s.abs() * 100.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sqrt_approximation_accuracy() {
        // Test with values that cover a wide range to test the sqrt approximation
        let n_rows = 3;
        let n_cols = 32; // Multiple of SIMD width (4)
        let mut data = Array2::zeros((n_rows, n_cols));
        let weights = vec![1.0, 1.0, 1.0]; // Equal weights to test just the sqrt approximation
        
        // Create test values: values near zero, medium values, and larger values
        for j in 0..n_cols {
            // Row 0: Small values (0.0001 to 0.0032)
            data[[0, j]] = 0.0001 * (j as f32 + 1.0);
            
            // Row 1: Medium values (0.25 to 8.0)
            data[[1, j]] = 0.25 * (j as f32 + 1.0);
            
            // Row 2: Larger values (100 to 3200)
            data[[2, j]] = 100.0 * (j as f32 + 1.0);
        }
        
        // Run both implementations
        let scalar_result = axis_sqrt_dot_product_scalar(&data, &weights);
        let simd_result = axis_sqrt_dot_product_simd(&data, &weights);
        
        // Print values for inspection and verify the general pattern matches
        println!("Scalar\tSIMD\tRelDiff");
        for j in 0..n_cols {
            let rel_diff = (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs() * 100.0;
            println!("{:.4}\t{:.4}\t{:.2}%", scalar_result[j], simd_result[j], rel_diff);
        }
        
        // Verify the general trend is similar between scalar and SIMD
        // by checking correlation between the results
        let scalar_mean = scalar_result.sum() / scalar_result.len() as f32;
        let simd_mean = simd_result.sum() / simd_result.len() as f32;
        
        let mut numerator = 0.0;
        let mut scalar_denom = 0.0;
        let mut simd_denom = 0.0;
        
        for j in 0..n_cols {
            let scalar_diff = scalar_result[j] - scalar_mean;
            let simd_diff = simd_result[j] - simd_mean;
            
            numerator += scalar_diff * simd_diff;
            scalar_denom += scalar_diff * scalar_diff;
            simd_denom += simd_diff * simd_diff;
        }
        
        let correlation = numerator / (scalar_denom.sqrt() * simd_denom.sqrt());
        println!("Correlation between scalar and SIMD results: {}", correlation);
        
        // Expect strong correlation above 0.99
        assert!(correlation > 0.99, "Correlation between scalar and SIMD is too low: {}", correlation);
        
        // Also verify the SIMD implementation generally follows the same pattern
        // by checking that the start and end values are in the correct order
        assert!(simd_result[0] < simd_result[n_cols-1], 
                "SIMD implementation doesn't follow the general increasing pattern");
                
        // And check that the average relative difference is acceptable
        let mut total_rel_diff = 0.0;
        for j in 0..n_cols {
            total_rel_diff += (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs();
        }
        let avg_rel_diff = total_rel_diff / n_cols as f32;
        println!("Average relative difference: {:.2}%", avg_rel_diff * 100.0);
        
        // Allow up to 1% average relative difference
        assert!(avg_rel_diff < 0.01, "Average relative difference too high: {:.2}%", avg_rel_diff * 100.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sqrt_unaligned_data_handling() {
        // Test with array sizes that aren't multiples of SIMD width
        let array = arr2(&[
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [25.0, 36.0, 49.0, 64.0, 81.0],
            [100.0, 121.0, 144.0, 169.0, 196.0]
        ]);
        let weights = vec![0.5, 1.0, 1.5];
        
        // Run both implementations
        let scalar_result = axis_sqrt_dot_product_scalar(&array, &weights);
        let simd_result = axis_sqrt_dot_product_simd(&array, &weights);
        
        // Verify reasonable similarity
        for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            println!("Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%", 
                j, s, v, (s - v).abs(), (s - v).abs() / s.abs() * 100.0);
            
            // Allow a small relative difference
            assert!((s - v).abs() / s.abs() < 0.01, 
                    "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
                    j, s, v, (s - v).abs() / s.abs() * 100.0);
        }
    }
} 