use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;

// Module declarations
pub mod neon;
pub mod scalar;

/// Main convolution function that uses the SIMD backend for optimization
/// Returns a convolved array with same dimensions as input, with zeros at edges (no padding)
pub fn convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    crate::simd::get_backend().convolution(kernel, xic)
}

#[cfg(test)]
mod tests;
