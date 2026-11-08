//! AVX2 + FMA backend: the dense-layer kernels of [`crate::mlp::avx2`]. Scoring and convolution
//! have no AVX2 kernels yet and use the scalar code.

use super::{Rank, SimdBackend};
use numpy::ndarray::{Array1, Array2};

cpufeatures::new!(avx2_check, "avx2", "fma");

pub struct Avx2Backend;

impl SimdBackend for Avx2Backend {
    fn test_backend(&self) -> String {
        // Dummy function to track which backend was called
        "avx2".to_string()
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
        crate::mlp::avx2::dense_forward_avx2(x, rows, k, wt, n, bias, activation, out)
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
        crate::mlp::avx2::dense_backward_input_avx2(dz, rows, n, w, k, a_prev, d_prev)
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
        crate::mlp::avx2::dense_weight_grad_avx2(a_prev, rows, k, dz, n, grad_wt, grad_b)
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
    ) {
        crate::mlp::avx2::output_layer_train_avx2(
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
        crate::mlp::avx2::output_layer_logits_avx2(a, rows, k, w, b, logits)
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
