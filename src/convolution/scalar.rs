use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;

/// Scalar convolution implementation without SIMD optimizations
pub fn convolution_scalar(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions, initialized to zeros
    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    // Early return for empty inputs
    if n_fragments == 0 || n_points == 0 || n_points < kernel_size {
        return convolved;
    }

    // Pre-compute valid region boundaries
    let start_idx = half_kernel;
    let end_idx = n_points.saturating_sub(half_kernel);

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Only process if there's a valid region
        if start_idx < end_idx {
            // Scalar loop with no bounds checking needed in valid region
            for i in start_idx..end_idx {
                let mut sum = 0.0;

                // Center element
                sum += xic_row[i] * kernel.kernel_array[half_kernel];

                // Process symmetric pairs
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];

                    // Get pair of symmetric inputs - no bounds check needed in valid region
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];

                    // Add symmetric inputs, then multiply by kernel value once
                    sum += (left_val + right_val) * kernel_val;
                }

                conv_row[i] = sum;
            }
        }

        // Zeros remain at the edges (no padding)
    }

    convolved
}
