#[cfg(target_arch = "aarch64")]
use super::neon::{axis_log_dot_product_neon, axis_sqrt_dot_product_neon};
#[cfg(target_arch = "aarch64")]
use super::scalar::{axis_log_dot_product_scalar, axis_sqrt_dot_product_scalar};
#[cfg(target_arch = "aarch64")]
use numpy::ndarray::Array2;

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
    let simd_result = axis_log_dot_product_neon(&data, &weights);

    // Verify SIMD and scalar results are reasonably close
    // The SIMD log approximation can diverge from scalar implementation
    for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        println!(
            "Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs(),
            (s - v).abs() / s.abs() * 100.0
        );

        // Allow a larger difference due to log approximation
        // This will still catch major implementation errors
        assert!(
            (s - v).abs() / s.abs() < 0.15,
            "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs() / s.abs() * 100.0
        );
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
    let simd_result = axis_log_dot_product_neon(&data, &weights);

    // Print values for inspection and verify the general pattern matches
    println!("Scalar\tSIMD\tRelDiff");
    for j in 0..n_cols {
        let rel_diff = (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs() * 100.0;
        println!(
            "{:.4}\t{:.4}\t{:.2}%",
            scalar_result[j], simd_result[j], rel_diff
        );
    }

    // Verify the general trend is similar between scalar and SIMD
    // by checking correlation between the results
    let scalar_mean = scalar_result.sum() / scalar_result.len() as f32;
    let simd_mean = simd_result.sum() / simd_result.len() as f32;

    let mut numerator = 0.0;
    let mut scalar_denom: f32 = 0.0;
    let mut simd_denom: f32 = 0.0;

    for j in 0..n_cols {
        let scalar_diff = scalar_result[j] - scalar_mean;
        let simd_diff = simd_result[j] - simd_mean;

        numerator += scalar_diff * simd_diff;
        scalar_denom += scalar_diff * scalar_diff;
        simd_denom += simd_diff * simd_diff;
    }

    let correlation = numerator / (scalar_denom.sqrt() * simd_denom.sqrt());
    println!(
        "Correlation between scalar and SIMD results: {}",
        correlation
    );

    // Expect strong correlation above 0.95
    assert!(
        correlation > 0.95,
        "Correlation between scalar and SIMD is too low: {}",
        correlation
    );

    // Also verify the SIMD implementation generally follows the same pattern
    // by checking that the start and end values are in the correct order
    assert!(
        simd_result[0] < simd_result[n_cols - 1],
        "SIMD implementation doesn't follow the general increasing pattern"
    );

    // And check that the average relative difference is acceptable
    let mut total_rel_diff = 0.0;
    for j in 0..n_cols {
        total_rel_diff += (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs();
    }
    let avg_rel_diff = total_rel_diff / n_cols as f32;
    println!("Average relative difference: {:.2}%", avg_rel_diff * 100.0);

    // Allow up to 10% average relative difference
    assert!(
        avg_rel_diff < 0.10,
        "Average relative difference too high: {:.2}%",
        avg_rel_diff * 100.0
    );
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_unaligned_data_handling() {
    use numpy::ndarray::arr2;

    // Test with array sizes that aren't multiples of SIMD width
    let array = arr2(&[
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
    ]);
    let weights = vec![0.5, 1.0, 1.5];

    // Run both implementations
    let scalar_result = axis_log_dot_product_scalar(&array, &weights);
    let simd_result = axis_log_dot_product_neon(&array, &weights);

    // Verify reasonable similarity - allow larger differences due to approximation
    for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        println!(
            "Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs(),
            (s - v).abs() / s.abs() * 100.0
        );

        // Allow a relative difference of up to 15%
        assert!(
            (s - v).abs() / s.abs() < 0.15,
            "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs() / s.abs() * 100.0
        );
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
    let simd_result = axis_sqrt_dot_product_neon(&data, &weights);

    // Verify SIMD and scalar results are reasonably close
    // The SIMD sqrt approximation can diverge from scalar implementation
    for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        println!(
            "Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs(),
            (s - v).abs() / s.abs() * 100.0
        );

        // Allow 1% relative error due to SIMD sqrt approximation
        // SIMD uses fast reciprocal square root approximation which introduces small errors
        assert!(
            (s - v).abs() / s.abs() < 0.01,
            "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs() / s.abs() * 100.0
        );
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
    let simd_result = axis_sqrt_dot_product_neon(&data, &weights);

    // Print values for inspection and verify the general pattern matches
    println!("Scalar\tSIMD\tRelDiff");
    for j in 0..n_cols {
        let rel_diff = (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs() * 100.0;
        println!(
            "{:.4}\t{:.4}\t{:.2}%",
            scalar_result[j], simd_result[j], rel_diff
        );
    }

    // Verify the general trend is similar between scalar and SIMD
    // by checking correlation between the results
    let scalar_mean = scalar_result.sum() / scalar_result.len() as f32;
    let simd_mean = simd_result.sum() / simd_result.len() as f32;

    let mut numerator = 0.0;
    let mut scalar_denom: f32 = 0.0;
    let mut simd_denom: f32 = 0.0;

    for j in 0..n_cols {
        let scalar_diff = scalar_result[j] - scalar_mean;
        let simd_diff = simd_result[j] - simd_mean;

        numerator += scalar_diff * simd_diff;
        scalar_denom += scalar_diff * scalar_diff;
        simd_denom += simd_diff * simd_diff;
    }

    let correlation = numerator / (scalar_denom.sqrt() * simd_denom.sqrt());
    println!(
        "Correlation between scalar and SIMD results: {}",
        correlation
    );

    // Expect strong correlation above 0.99
    assert!(
        correlation > 0.99,
        "Correlation between scalar and SIMD is too low: {}",
        correlation
    );

    // Also verify the SIMD implementation generally follows the same pattern
    // by checking that the start and end values are in the correct order
    assert!(
        simd_result[0] < simd_result[n_cols - 1],
        "SIMD implementation doesn't follow the general increasing pattern"
    );

    // And check that the average relative difference is acceptable
    let mut total_rel_diff = 0.0;
    for j in 0..n_cols {
        total_rel_diff += (scalar_result[j] - simd_result[j]).abs() / scalar_result[j].abs();
    }
    let avg_rel_diff = total_rel_diff / n_cols as f32;
    println!("Average relative difference: {:.2}%", avg_rel_diff * 100.0);

    // Allow up to 1% average relative difference
    assert!(
        avg_rel_diff < 0.01,
        "Average relative difference too high: {:.2}%",
        avg_rel_diff * 100.0
    );
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_sqrt_unaligned_data_handling() {
    use numpy::ndarray::arr2;

    // Test with array sizes that aren't multiples of SIMD width
    let array = arr2(&[
        [0.0, 1.0, 4.0, 9.0, 16.0],
        [25.0, 36.0, 49.0, 64.0, 81.0],
        [100.0, 121.0, 144.0, 169.0, 196.0],
    ]);
    let weights = vec![0.5, 1.0, 1.5];

    // Run both implementations
    let scalar_result = axis_sqrt_dot_product_scalar(&array, &weights);
    let simd_result = axis_sqrt_dot_product_neon(&array, &weights);

    // Verify reasonable similarity
    for (j, (s, v)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        println!(
            "Value {}: scalar={}, simd={}, diff={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs(),
            (s - v).abs() / s.abs() * 100.0
        );

        // Allow 1% relative error due to SIMD sqrt approximation
        // SIMD uses fast reciprocal square root approximation which introduces small errors
        assert!(
            (s - v).abs() / s.abs() < 0.01,
            "Value {} differs too much: scalar={}, simd={}, rel_diff={:.2}%",
            j,
            s,
            v,
            (s - v).abs() / s.abs() * 100.0
        );
    }
}
