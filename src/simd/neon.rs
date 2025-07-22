#[cfg(target_arch = "aarch64")]
use super::{Rank, SimdBackend};
#[cfg(target_arch = "aarch64")]
use numpy::ndarray::{Array1, Array2};

#[cfg(target_arch = "aarch64")]
cpufeatures::new!(neon_check, "neon");

#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn test_backend(&self) -> String {
        // Dummy function to track that neon backend was called
        "neon".to_string()
    }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::neon::axis_log_dot_product_neon(array, weights)
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::neon::axis_sqrt_dot_product_neon(array, weights)
    }

    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &Array2<f32>,
    ) -> Array2<f32> {
        crate::convolution::neon::convolution_neon(kernel, xic)
    }

    fn name(&self) -> &'static str {
        "neon"
    }

    fn is_available(&self) -> bool {
        neon_check::get()
    }

    fn priority(&self) -> Rank {
        Rank::Neon
    }
}
