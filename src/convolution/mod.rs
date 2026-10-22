use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;

// Module declarations
pub mod neon;
pub mod scalar;

/// Main convolution function that uses the SIMD backend for optimization
/// Returns a convolved array with same dimensions as input. Within half a kernel of either end the kernel is
/// truncated to the points that exist and rescaled to the full kernel's weight.
pub fn convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    crate::simd::get_backend().convolution(kernel, xic)
}

#[cfg(test)]
mod tests;
