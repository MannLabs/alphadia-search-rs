//! AVX-512F backend with the layer kernels of [`crate::mlp::avx512`] and the AVX-512 quantile
//! transform.
//! Every other operation uses the scalar default of [`SimdBackend`].

use super::{Rank, SimdBackend};

cpufeatures::new!(avx512_check, "avx512f");

pub struct Avx512Backend;

// SAFETY (every `unsafe` block below): `get_backend` returns this backend only when
// `is_available` is true. Then the CPU supports the instruction set of the kernels.
impl SimdBackend for Avx512Backend {
    fn test_backend(&self) -> String {
        // Dummy function to track which backend was called
        "avx512".to_string()
    }

    fn quantile_transform_column(
        &self,
        map: &crate::fdr_classifier::quantile::FeatureMap,
        x: &[f32],
        out: &mut [f32],
    ) {
        unsafe { crate::fdr_classifier::quantile::avx512::apply_column(map, x, out) }
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
        unsafe {
            crate::mlp::avx512::dense_forward_avx512(x, rows, k, wt, n, bias, activation, out)
        }
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
        unsafe {
            crate::mlp::avx512::dense_backward_input_avx512(dz, rows, n, w, k, a_prev, d_prev)
        }
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
        unsafe {
            crate::mlp::avx512::dense_weight_grad_avx512(a_prev, rows, k, dz, n, grad_wt, grad_b)
        }
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
        unsafe {
            crate::mlp::avx512::output_layer_train_avx512(
                a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b,
            )
        }
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
        unsafe { crate::mlp::avx512::output_layer_logits_avx512(a, rows, k, w, b, logits) }
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
