//! Portable dense-layer kernels: plain safe Rust that LLVM autovectorises for whatever the target
//! offers (SSE2, NEON, wasm-simd128, ...). The SIMD backends also call them for shapes outside
//! their fast paths.
//!
//! The scheme of the hand-written SIMD kernels, in fixed-size arrays: a register-blocked tile of
//! `M` rows x `N` columns of `C = A @ B` stays in registers for the whole reduction, LLVM
//! vectorises it along the columns, and the epilogue (bias + ReLU, bias, the ReLU mask of the
//! previous layer, or accumulation into `C`) is fused into the store. A reduction step reads `M`
//! elements of A (broadcast) and one `N`-wide row of B, so A is never packed and may be read
//! transposed: the weight gradient runs the same tile on the transposed activations. Row and
//! column tails use smaller tiles, so any shape works.

use super::Activation;
use std::array;

/// Rows of a full tile.
const MR: usize = 4;
/// Columns of a full tile: two vectors of 4 lanes on 16-register targets, four on the
/// 32-register aarch64 (as the NEON kernel).
const NR: usize = if cfg!(target_arch = "aarch64") { 16 } else { 8 };
/// Rows the weight-gradient kernel keeps in cache while it sweeps all tiles.
const GRAD_ROW_BLOCK: usize = 64;

/// `a * b + c`, fused where the target has a fused multiply-add instruction (elsewhere
/// `mul_add` would be a libm call).
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `acc[i][j] += a[i] * b[j]` for each reduction step `(a, b)`.
#[inline(always)]
fn accumulate<const M: usize, const N: usize>(
    acc: &mut [[f32; N]; M],
    steps: impl Iterator<Item = ([f32; M], [f32; N])>,
) {
    for (a, b) in steps {
        for (acc_row, &a_i) in acc.iter_mut().zip(&a) {
            for (c, &b_j) in acc_row.iter_mut().zip(&b) {
                *c = madd(a_i, b_j, *c);
            }
        }
    }
}

/// The `N` elements of `row` from `j0`.
#[inline(always)]
fn part<const N: usize>(row: &[f32], j0: usize) -> [f32; N] {
    row[j0..j0 + N].try_into().unwrap()
}

/// A kernel that computes one `M x N` tile of its output at a time.
trait Tiled {
    fn tile<const M: usize, const N: usize>(&mut self, i0: usize, j0: usize);
}

/// Covers an `m x n` output with tiles: row tiles outer (heights `MR`, 2, 1), column tiles inner
/// (widths `NR`, 8, 4, 1).
fn for_each_tile(m: usize, n: usize, kernel: &mut impl Tiled) {
    fn columns<const M: usize>(i0: usize, n: usize, kernel: &mut impl Tiled) {
        let mut j0 = 0;
        while j0 + NR <= n {
            kernel.tile::<M, NR>(i0, j0);
            j0 += NR;
        }
        if NR > 8 && j0 + 8 <= n {
            kernel.tile::<M, 8>(i0, j0);
            j0 += 8;
        }
        while j0 + 4 <= n {
            kernel.tile::<M, 4>(i0, j0);
            j0 += 4;
        }
        for j in j0..n {
            kernel.tile::<M, 1>(i0, j);
        }
    }
    let mut i0 = 0;
    while i0 + MR <= m {
        columns::<MR>(i0, n, kernel);
        i0 += MR;
    }
    if i0 + 2 <= m {
        columns::<2>(i0, n, kernel);
        i0 += 2;
    }
    if i0 < m {
        columns::<1>(i0, n, kernel);
    }
}

/// `out = bias + x @ wt`, then ReLU if `RELU` (a runtime flag would make LLVM unswitch the
/// tile loops and give up vectorising them).
struct Forward<'a, const RELU: bool> {
    x: &'a [f32],
    k: usize,
    wt: &'a [f32],
    n: usize,
    bias: &'a [f32],
    out: &'a mut [f32],
}

impl<const RELU: bool> Tiled for Forward<'_, RELU> {
    #[inline(always)]
    fn tile<const M: usize, const N: usize>(&mut self, i0: usize, j0: usize) {
        let k = self.k;
        let x: [&[f32]; M] = array::from_fn(|i| &self.x[(i0 + i) * k..][..k]);
        let mut acc = [[0.0; N]; M];
        let steps = (0..k).zip(self.wt.chunks_exact(self.n));
        accumulate(
            &mut acc,
            steps.map(|(p, wt_row)| (array::from_fn(|i| x[i][p]), part(wt_row, j0))),
        );
        let bias = &self.bias[j0..][..N];
        for (i, acc_row) in acc.iter().enumerate() {
            let out = &mut self.out[(i0 + i) * self.n + j0..][..N];
            for ((o, &v), &b) in out.iter_mut().zip(acc_row).zip(bias) {
                *o = if RELU { (v + b).max(0.0) } else { v + b };
            }
        }
    }
}

/// `d_prev = (dz @ w) * (a_prev > 0)`.
struct BackwardInput<'a> {
    dz: &'a [f32],
    n: usize,
    w: &'a [f32],
    k: usize,
    a_prev: &'a [f32],
    d_prev: &'a mut [f32],
}

impl Tiled for BackwardInput<'_> {
    #[inline(always)]
    fn tile<const M: usize, const N: usize>(&mut self, i0: usize, j0: usize) {
        let (n, k) = (self.n, self.k);
        let dz: [&[f32]; M] = array::from_fn(|i| &self.dz[(i0 + i) * n..][..n]);
        let mut acc = [[0.0; N]; M];
        let steps = (0..n).zip(self.w.chunks_exact(k));
        accumulate(
            &mut acc,
            steps.map(|(p, w_row)| (array::from_fn(|i| dz[i][p]), part(w_row, j0))),
        );
        for (i, acc_row) in acc.iter().enumerate() {
            let at = (i0 + i) * k + j0;
            let a_prev = &self.a_prev[at..][..N];
            let d_prev = &mut self.d_prev[at..][..N];
            for ((d, &v), &a) in d_prev.iter_mut().zip(acc_row).zip(a_prev) {
                *d = if a > 0.0 { v } else { 0.0 };
            }
        }
    }
}

/// `grad_wt += a_prev^T @ dz` over one block of rows: the tile rows are the `k` inputs.
struct WeightGrad<'a> {
    a_prev: &'a [f32],
    k: usize,
    dz: &'a [f32],
    n: usize,
    grad_wt: &'a mut [f32],
}

impl Tiled for WeightGrad<'_> {
    #[inline(always)]
    fn tile<const M: usize, const N: usize>(&mut self, i0: usize, j0: usize) {
        let n = self.n;
        let mut acc: [[f32; N]; M] =
            array::from_fn(|i| part(&self.grad_wt[(i0 + i) * n..], j0));
        let steps = self.a_prev.chunks_exact(self.k).zip(self.dz.chunks_exact(n));
        accumulate(
            &mut acc,
            steps.map(|(a_row, dz_row)| (part(a_row, i0), part(dz_row, j0))),
        );
        for (i, acc_row) in acc.iter().enumerate() {
            self.grad_wt[(i0 + i) * n + j0..][..N].copy_from_slice(acc_row);
        }
    }
}

/// `out[rows x n] = activation(bias + x[rows x k] @ wt[k x n])`.
#[allow(clippy::too_many_arguments)]
pub fn dense_forward_scalar(
    x: &[f32],
    rows: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: &[f32],
    activation: Activation,
    out: &mut [f32],
) {
    let (x, wt, bias, out) = (&x[..rows * k], &wt[..k * n], &bias[..n], &mut out[..rows * n]);
    match activation {
        Activation::Relu => {
            for_each_tile(rows, n, &mut Forward::<true> { x, k, wt, n, bias, out })
        }
        Activation::Linear => {
            for_each_tile(rows, n, &mut Forward::<false> { x, k, wt, n, bias, out })
        }
    }
}

/// `d_prev[rows x k] = (dz[rows x n] @ w[n x k]) * (a_prev > 0)`.
#[allow(clippy::too_many_arguments)]
pub fn dense_backward_input_scalar(
    dz: &[f32],
    rows: usize,
    n: usize,
    w: &[f32],
    k: usize,
    a_prev: &[f32],
    d_prev: &mut [f32],
) {
    let mut kernel = BackwardInput {
        dz: &dz[..rows * n],
        n,
        w: &w[..n * k],
        k,
        a_prev: &a_prev[..rows * k],
        d_prev: &mut d_prev[..rows * k],
    };
    for_each_tile(rows, k, &mut kernel);
}

/// `grad_wt[k x n] += a_prev[rows x k]^T @ dz[rows x n]`, `grad_b[n] += colsum(dz)`.
#[allow(clippy::too_many_arguments)]
pub fn dense_weight_grad_scalar(
    a_prev: &[f32],
    rows: usize,
    k: usize,
    dz: &[f32],
    n: usize,
    grad_wt: &mut [f32],
    grad_b: &mut [f32],
) {
    if k == 0 || n == 0 {
        return;
    }
    let blocks = a_prev[..rows * k]
        .chunks(GRAD_ROW_BLOCK * k)
        .zip(dz[..rows * n].chunks(GRAD_ROW_BLOCK * n));
    for (a_prev, dz) in blocks {
        let mut kernel = WeightGrad {
            a_prev,
            k,
            dz,
            n,
            grad_wt: &mut grad_wt[..k * n],
        };
        for_each_tile(k, n, &mut kernel);
    }
    let grad_b = &mut grad_b[..n];
    for dz_row in dz[..rows * n].chunks_exact(n) {
        for (g, &d) in grad_b.iter_mut().zip(dz_row) {
            *g += d;
        }
    }
}

/// `a . w`, with independent accumulators so that the sum vectorises.
#[inline(always)]
fn dot(a: &[f32], w: &[f32]) -> f32 {
    let w = &w[..a.len()];
    let (a_chunks, a_tail) = a.as_chunks::<NR>();
    let (w_chunks, w_tail) = w.as_chunks::<NR>();
    let mut acc = [0.0f32; NR];
    for (a, w) in a_chunks.iter().zip(w_chunks) {
        for ((c, &x), &y) in acc.iter_mut().zip(a).zip(w) {
            *c = madd(x, y, *c);
        }
    }
    let tail: f32 = a_tail.iter().zip(w_tail).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// Numerically stable `log(1 + exp(-|z|)) + max(z, 0) - y z`.
pub(crate) fn bce_with_logits(logit: f32, y: f32) -> f32 {
    logit.max(0.0) - logit * y + (-logit.abs()).exp().ln_1p()
}

/// Single-logit output layer fused with BCE-with-logits; returns the summed loss.
#[allow(clippy::too_many_arguments)]
pub fn output_layer_train_scalar(
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
    if k == 0 {
        return output_layer_train_scalar_no_inputs(rows, b, y, loss_scale, grad_b);
    }
    let (w, grad_w) = (&w[..k], &mut grad_w[..k]);
    let mut loss = 0.0f32;
    let rows_iter = a[..rows * k]
        .chunks_exact(k)
        .zip(dz_prev[..rows * k].chunks_exact_mut(k))
        .zip(&y[..rows]);
    for ((a_row, dz_row), &label) in rows_iter {
        let logit = b + dot(a_row, w);
        // one exp for both the stable loss and the sigmoid
        let e = (-logit.abs()).exp();
        loss += logit.max(0.0) - logit * label + e.ln_1p();
        let sigmoid = if logit >= 0.0 { 1.0 } else { e } / (1.0 + e);
        let d = (sigmoid - label) * loss_scale;
        *grad_b += d;
        for (((g, dz), &x), &wv) in grad_w.iter_mut().zip(dz_row).zip(a_row).zip(w) {
            *g = madd(d, x, *g);
            *dz = if x > 0.0 { d * wv } else { 0.0 };
        }
    }
    loss
}

/// The output layer of a network without hidden units: the logit is the bias.
fn output_layer_train_scalar_no_inputs(
    rows: usize,
    b: f32,
    y: &[f32],
    loss_scale: f32,
    grad_b: &mut f32,
) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-b).exp());
    y[..rows]
        .iter()
        .map(|&label| {
            *grad_b += (sigmoid - label) * loss_scale;
            bce_with_logits(b, label)
        })
        .sum()
}

/// `logits[r] = a[r] . w + b`.
pub fn output_layer_logits_scalar(
    a: &[f32],
    rows: usize,
    k: usize,
    w: &[f32],
    b: f32,
    logits: &mut [f32],
) {
    if k == 0 {
        logits[..rows].fill(b);
        return;
    }
    for (logit, a_row) in logits[..rows].iter_mut().zip(a.chunks_exact(k)) {
        *logit = b + dot(a_row, &w[..k]);
    }
}
