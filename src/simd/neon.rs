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

    fn mlp_dense_forward(
        &self,
        x: &[f32],
        rows: usize,
        k: usize,
        wt: &[f32],
        n: usize,
        bias: &[f32],
        activation: crate::mlp::Activation,
        out: &mut [f32],
    ) {
        crate::mlp::neon::dense_forward_neon(x, rows, k, wt, n, bias, activation, out)
    }

    fn mlp_dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &mut [f32],
    ) {
        crate::mlp::neon::dense_backward_input_neon(dz, rows, n, w, k, a_prev, d_prev)
    }

    fn mlp_dense_weight_grad(
        &self,
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &mut [f32],
        grad_b: &mut [f32],
    ) {
        crate::mlp::neon::dense_weight_grad_neon(a_prev, rows, k, dz, n, grad_wt, grad_b)
    }

    fn mlp_output_layer_train(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        y: &[f32],
        loss_scale: f32,
        dz_prev: &mut [f32],
        grad_w: &mut [f32],
        grad_b: &mut f32,
    ) -> f32 {
        crate::mlp::neon::output_layer_train_neon(
            a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b,
        )
    }

    fn mlp_output_layer_logits(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        logits: &mut [f32],
    ) {
        crate::mlp::neon::output_layer_logits_neon(a, rows, k, w, b, logits)
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
