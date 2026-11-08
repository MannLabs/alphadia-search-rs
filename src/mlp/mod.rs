//! A small feed-forward ReLU network with one output logit (the FDR classifier): [`Network`]
//! trains on row chunks and predicts, composed of fused dense-layer kernels.
//!
//! All matrices are row-major `f32`. A dense layer with `k` inputs and `n` outputs keeps its
//! weights transposed, `wt[k x n]`, so the forward pass streams contiguous weight rows.
//! The kernels dispatch through the SIMD backend (portable, NEON, AVX2 or AVX-512); every SIMD
//! backend is a [`LayerKernels`] implementation. The tests compare each against the naive loops
//! in `reference`.

mod aligned;
pub mod neon;
mod network;
#[cfg(test)]
mod reference;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod x86;

pub use aligned::AlignedBuf;
pub use network::{LayerKernels, Network, TrainWorkspace};

/// What a forward kernel applies to `x @ wt` before storing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `bias + x @ wt`
    Linear,
    /// `max(bias + x @ wt, 0)`
    Relu,
}

/// `out[rows x n] = activation(bias + x[rows x k] @ wt[k x n])`.
#[allow(clippy::too_many_arguments)]
pub fn dense_forward(
    x: &[f32],
    rows: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: &[f32],
    activation: Activation,
    out: &mut [f32],
) {
    check_len(x, rows * k, "x");
    check_len(wt, k * n, "wt");
    check_len(bias, n, "bias");
    check_len(out, rows * n, "out");
    crate::simd::get_backend().mlp_dense_forward(x, rows, k, wt, n, bias, activation, out)
}

/// Gradient w.r.t. the input of a dense layer that follows a ReLU:
/// `d_prev[rows x k] = (dz[rows x n] @ w[n x k]) * (a_prev > 0)`, with `w` the
/// non-transposed weights.
#[allow(clippy::too_many_arguments)]
pub fn dense_backward_input(
    dz: &[f32],
    rows: usize,
    n: usize,
    w: &[f32],
    k: usize,
    a_prev: &[f32],
    d_prev: &mut [f32],
) {
    check_len(dz, rows * n, "dz");
    check_len(w, n * k, "w");
    check_len(a_prev, rows * k, "a_prev");
    check_len(d_prev, rows * k, "d_prev");
    crate::simd::get_backend().mlp_dense_backward_input(dz, rows, n, w, k, a_prev, d_prev)
}

/// Accumulate the weight and bias gradients of a dense layer:
/// `grad_wt[k x n] += a_prev[rows x k]^T @ dz[rows x n]`, `grad_b[n] += colsum(dz)`.
#[allow(clippy::too_many_arguments)]
pub fn dense_weight_grad(
    a_prev: &[f32],
    rows: usize,
    k: usize,
    dz: &[f32],
    n: usize,
    grad_wt: &mut [f32],
    grad_b: &mut [f32],
) {
    check_len(a_prev, rows * k, "a_prev");
    check_len(dz, rows * n, "dz");
    check_len(grad_wt, k * n, "grad_wt");
    check_len(grad_b, n, "grad_b");
    crate::simd::get_backend().mlp_dense_weight_grad(a_prev, rows, k, dz, n, grad_wt, grad_b)
}

/// Single-logit output layer for training, fused with BCE-with-logits:
/// `logit = a[r] . w + b`, `d = (sigmoid(logit) - y[r]) * loss_scale`,
/// `dz_prev[r] = d * w * (a[r] > 0)`, `grad_w += d * a[r]`, `grad_b += d`.
#[allow(clippy::too_many_arguments)]
pub fn output_layer_train(
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
    check_len(a, rows * k, "a");
    check_len(w, k, "w");
    check_len(y, rows, "y");
    check_len(dz_prev, rows * k, "dz_prev");
    check_len(grad_w, k, "grad_w");
    crate::simd::get_backend()
        .mlp_output_layer_train(a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b)
}

/// `logits[r] = a[r] . w + b`.
pub fn output_layer_logits(
    a: &[f32],
    rows: usize,
    k: usize,
    w: &[f32],
    b: f32,
    logits: &mut [f32],
) {
    check_len(a, rows * k, "a");
    check_len(w, k, "w");
    check_len(logits, rows, "logits");
    crate::simd::get_backend().mlp_output_layer_logits(a, rows, k, w, b, logits)
}

fn check_len(values: &[f32], expected: usize, name: &str) {
    assert!(
        values.len() >= expected,
        "{name} has {} elements, expected at least {expected}",
        values.len()
    );
}

#[cfg(test)]
mod tests;
