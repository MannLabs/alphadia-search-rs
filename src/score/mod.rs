use numpy::ndarray::{Array1, Array2};

// Module declarations
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod neon;
pub mod scalar;

/// First applies square root to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_sqrt_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    crate::simd::get_backend().axis_sqrt_dot_product(array, weights)
}

/// First applies logarithm to each element, then performs a weighted dot product along the first axis.
/// Returns a 1D array with the same length as the second dimension.
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    crate::simd::get_backend().axis_log_dot_product(array, weights)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_neon;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests_avx2;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests_avx512;
