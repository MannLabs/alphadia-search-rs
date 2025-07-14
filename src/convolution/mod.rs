use numpy::ndarray::Array2;
use crate::kernel::GaussianKernel;

// Module declarations
pub mod scalar;
pub mod neon;

/// Main convolution function that uses the SIMD backend for optimization
/// Returns a convolved array with same dimensions as input, with zeros at edges (no padding)
pub fn convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    crate::simd::get_backend().convolution(kernel, xic)
}

// Safe reference implementation for testing
#[allow(dead_code)]
pub fn safe_reference_convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;
    
    // Create output array with same dimensions, initialized to zeros
    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));
    
    // Early return for empty inputs
    if n_fragments == 0 || n_points == 0 {
        return convolved;
    }
    
    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);
        
        // Only compute convolution for valid points (where kernel fits completely)
        let start_idx = half_kernel;
        let end_idx = n_points.saturating_sub(half_kernel);
        
        for i in start_idx..end_idx {
            let mut sum = 0.0;
            
            for k in 0..kernel_size {
                let data_idx = i + k - half_kernel;
                if data_idx < n_points {
                    sum += xic_row[data_idx] * kernel.kernel_array[k];
                }
            }
            
            conv_row[i] = sum;
        }
    }
    
    convolved
}



#[cfg(test)]
mod tests;
