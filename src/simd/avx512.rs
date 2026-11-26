//! AVX-512F backend with the x86 ports of the NEON score and convolution kernels.

use super::{Rank, SimdBackend};
use numpy::ndarray::{Array1, Array2};

cpufeatures::new!(avx512_check, "avx512f");

pub struct Avx512Backend;

// SAFETY (every `unsafe` block below): `get_backend` returns this backend only when
// `is_available` is true. Then the CPU supports AVX-512F.
impl SimdBackend for Avx512Backend {
    fn test_backend(&self) -> String {
        "avx512".to_string()
    }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        unsafe { crate::score::avx512::axis_log_dot_product_avx512(array, weights) }
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        unsafe { crate::score::avx512::axis_sqrt_dot_product_avx512(array, weights) }
    }

    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &Array2<f32>,
    ) -> Array2<f32> {
        unsafe { crate::convolution::avx512::convolution_avx512(kernel, xic) }
    }

    fn name(&self) -> &'static str {
        "avx512"
    }

    fn is_available(&self) -> bool {
        avx512_check::get()
    }

    fn priority(&self) -> Rank {
        Rank::Avx512
    }
}
