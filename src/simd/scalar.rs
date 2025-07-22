use super::{Rank, SimdBackend};
use numpy::ndarray::{Array1, Array2};

pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn test_backend(&self) -> String {
        // Dummy function to track that scalar backend was called
        "scalar".to_string()
    }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::scalar::axis_log_dot_product_scalar(array, weights)
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::scalar::axis_sqrt_dot_product_scalar(array, weights)
    }

    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &Array2<f32>,
    ) -> Array2<f32> {
        crate::convolution::scalar::convolution_scalar(kernel, xic)
    }

    fn name(&self) -> &'static str {
        "scalar"
    }

    fn is_available(&self) -> bool {
        true // Scalar backend is always available
    }

    fn priority(&self) -> Rank {
        Rank::Scalar
    }
}
