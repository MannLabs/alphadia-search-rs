use numpy::ndarray::{Array2, Array1};

// Module declarations
pub mod scalar;
pub mod neon;

// Re-export common scalar functions for convenience
pub use scalar::{axis_dot_product, axis_log_sum};

/// First applies square root to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_sqrt_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    crate::simd::get_backend().axis_sqrt_dot_product(array, weights)
}

/// First applies logarithm to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    crate::simd::get_backend().axis_log_dot_product(array, weights)
}

#[cfg(test)]
mod tests;
 