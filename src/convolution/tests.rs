use super::*;
use crate::kernel::GaussianKernel;
use approx::assert_relative_eq;
use numpy::ndarray::Array2;

#[test]
fn test_basic_convolution() {
    // Create a simple Gaussian kernel
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 5, 1.0);

    // Create a simple test XIC with a single row
    let xic_data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let xic = Array2::from_shape_vec((1, 5), xic_data).unwrap();

    // Apply convolution
    let result = convolution(&kernel, &xic);

    // Check dimensions match
    assert_eq!(result.dim(), xic.dim());
}

#[test]
fn test_edge_cases() {
    // Test case: kernel size equals the data size
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 5, 1.0);

    let xic_data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let xic = Array2::from_shape_vec((1, 5), xic_data).unwrap();

    let result = convolution(&kernel, &xic);
    assert_eq!(result.dim(), xic.dim());

    // Test case: kernel size larger than data size
    let kernel_large = GaussianKernel::new(sigma, 1.0, 7, 1.0);
    let xic_small = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.0]).unwrap();

    let result = convolution(&kernel_large, &xic_small);
    assert_eq!(result.dim(), xic_small.dim());
}

#[test]
fn test_empty_input() {
    // Test with empty array (0 rows)
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 5, 1.0);

    let xic_empty = Array2::<f32>::zeros((0, 10));
    let result = convolution(&kernel, &xic_empty);
    assert_eq!(result.dim(), xic_empty.dim());

    // Test with empty array (0 columns)
    let xic_empty = Array2::<f32>::zeros((5, 0));
    let result = convolution(&kernel, &xic_empty);
    assert_eq!(result.dim(), xic_empty.dim());
}

#[test]
fn test_multiple_fragments() {
    // Test with multiple fragments
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 5, 1.0);

    let mut xic = Array2::<f32>::zeros((3, 10));
    // Set some test values
    xic[[0, 5]] = 1.0;
    xic[[1, 3]] = 1.0;
    xic[[2, 7]] = 1.0;

    let result = convolution(&kernel, &xic);
    assert_eq!(result.dim(), xic.dim());
}

#[test]
fn test_against_reference_implementation() {
    // Test against our safe reference implementation
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 5, 1.0);

    let xic_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let xic = Array2::from_shape_vec((1, 10), xic_data).unwrap();

    let result = convolution(&kernel, &xic);
    let reference_result = scalar::convolution_scalar(&kernel, &xic);

    // Compare optimized implementation with reference implementation
    assert_eq!(result.dim(), reference_result.dim());

    // Compare values where both implementations should produce non-zero results
    let (_, n_points) = xic.dim();
    let half_kernel = kernel.kernel_array.len() / 2;

    if n_points > 2 * half_kernel {
        for i in half_kernel..(n_points - half_kernel) {
            assert_relative_eq!(result[[0, i]], reference_result[[0, i]], epsilon = 1e-5);
        }
    }
}

#[test]
fn test_very_small_input() {
    let sigma = 0.5;
    let kernel = GaussianKernel::new(sigma, 1.0, 3, 1.0);

    // Test with very small input (1x1)
    let xic_small = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let result = convolution(&kernel, &xic_small);
    assert_eq!(result.dim(), xic_small.dim());

    // Test with input smaller than half kernel
    let xic_small = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let result = convolution(&kernel, &xic_small);
    assert_eq!(result.dim(), xic_small.dim());
}

#[test]
fn test_specific_out_of_bounds_case() {
    // Create a test case similar to the one causing the error
    let sigma = 1.0;
    let kernel_sizes = vec![3, 5, 7, 9, 11]; // Various kernel sizes

    // Test with various input sizes
    let input_sizes = vec![1, 2, 3, 5, 10];

    for &kernel_size in &kernel_sizes {
        let kernel = GaussianKernel::new(sigma, 1.0, kernel_size, 1.0);

        for &input_size in &input_sizes {
            // Create test input
            let mut xic = Array2::<f32>::zeros((1, input_size));
            // Set a value in the middle (if possible)
            if input_size > 0 {
                xic[[0, input_size / 2]] = 1.0;
            }

            // This should not panic
            let result = convolution(&kernel, &xic);

            // Verify dimensions match
            assert_eq!(result.dim(), xic.dim());

            // Compare with reference implementation
            let reference = scalar::convolution_scalar(&kernel, &xic);
            assert_eq!(result.dim(), reference.dim());
        }
    }
}

// Add another test for the edge case likely triggering the bug
#[test]
fn test_edge_case_kernel_larger_than_input() {
    let sigma = 1.0;
    let kernel = GaussianKernel::new(sigma, 1.0, 11, 1.0); // Large kernel

    // Small input
    let xic = Array2::<f32>::ones((1, 5));

    // This should not panic
    let result = convolution(&kernel, &xic);

    // Verify
    assert_eq!(result.dim(), xic.dim());
}

#[cfg(target_arch = "aarch64")]
mod tests_neon_v2 {
    use crate::convolution::neon::convolution_neon_v2;
    use crate::convolution::scalar::convolution_scalar;
    use crate::kernel::GaussianKernel;
    use numpy::ndarray::Array2;

    #[test]
    fn test_conv_neon_v2_vs_scalar() {
        let kernel = GaussianKernel::new(2.0, 1.0, 20, 1.0);
        let n_fragments = 12;
        let n_cols = 100;

        let mut xic = Array2::<f32>::zeros((n_fragments, n_cols));
        for i in 0..n_fragments {
            for j in 0..n_cols {
                xic[[i, j]] = ((i * n_cols + j) as f32 * 0.1).sin().abs();
            }
        }

        let scalar_result = convolution_scalar(&kernel, &xic);
        let v2_result = convolution_neon_v2(&kernel, &xic);

        assert_eq!(scalar_result.dim(), v2_result.dim());

        let half_kernel = kernel.kernel_array.len() / 2;
        for i in 0..n_fragments {
            for j in half_kernel..(n_cols - half_kernel) {
                let s = scalar_result[[i, j]];
                let v = v2_result[[i, j]];
                let diff = (s - v).abs();
                assert!(
                    diff < 1e-5,
                    "Fragment {} col {}: scalar={}, v2={}, diff={}",
                    i,
                    j,
                    s,
                    v,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_conv_neon_v2_unaligned_cols() {
        let kernel = GaussianKernel::new(0.5, 1.0, 5, 1.0);

        let mut xic = Array2::<f32>::zeros((3, 23));
        for i in 0..3 {
            for j in 0..23 {
                xic[[i, j]] = (j as f32) * 0.1 + (i as f32);
            }
        }

        let scalar_result = convolution_scalar(&kernel, &xic);
        let v2_result = convolution_neon_v2(&kernel, &xic);

        assert_eq!(scalar_result.dim(), v2_result.dim());

        let half_kernel = kernel.kernel_array.len() / 2;
        for i in 0..3 {
            for j in half_kernel..(23 - half_kernel) {
                let diff = (scalar_result[[i, j]] - v2_result[[i, j]]).abs();
                assert!(diff < 1e-5, "Fragment {} col {}: diff={}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_conv_neon_v2_wide_matrix() {
        let kernel = GaussianKernel::new(2.0, 1.0, 20, 1.0);
        let n_fragments = 48;
        let n_cols = 10000;

        let mut xic = Array2::<f32>::zeros((n_fragments, n_cols));
        for i in 0..n_fragments {
            for j in 0..n_cols {
                xic[[i, j]] = ((i * n_cols + j) as f32 * 0.001).sin().abs();
            }
        }

        let scalar_result = convolution_scalar(&kernel, &xic);
        let v2_result = convolution_neon_v2(&kernel, &xic);

        assert_eq!(scalar_result.dim(), v2_result.dim());

        let half_kernel = kernel.kernel_array.len() / 2;
        let mut max_diff: f32 = 0.0;
        for i in 0..n_fragments {
            for j in half_kernel..(n_cols - half_kernel) {
                let diff = (scalar_result[[i, j]] - v2_result[[i, j]]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        assert!(
            max_diff < 1e-4,
            "Max absolute difference too high: {}",
            max_diff
        );
    }
}
