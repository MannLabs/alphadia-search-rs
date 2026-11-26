//! AVX2 + FMA backend with the x86 ports of the NEON score and convolution kernels.

use super::{Rank, SimdBackend};
use numpy::ndarray::{Array1, Array2};

cpufeatures::new!(avx2_check, "avx2", "fma");

pub struct Avx2Backend;

// SAFETY (every `unsafe` block below): `get_backend` returns this backend only when
// `is_available` is true. Then the CPU supports AVX2 + FMA.
impl SimdBackend for Avx2Backend {
    fn test_backend(&self) -> String {
        "avx2".to_string()
    }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        unsafe { crate::score::avx2::axis_log_dot_product_avx2(array, weights) }
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        unsafe { crate::score::avx2::axis_sqrt_dot_product_avx2(array, weights) }
    }

    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &Array2<f32>,
    ) -> Array2<f32> {
        unsafe { crate::convolution::avx2::convolution_avx2(kernel, xic) }
    }

    fn name(&self) -> &'static str {
        "avx2"
    }

    fn is_available(&self) -> bool {
        avx2_check::get()
    }

    fn priority(&self) -> Rank {
        Rank::Avx2
    }
}
