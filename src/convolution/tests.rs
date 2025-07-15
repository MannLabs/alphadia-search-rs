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
