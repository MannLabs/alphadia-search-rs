//! Portable dense-layer kernels: plain safe Rust that LLVM autovectorises for whatever the target
//! offers (SSE2, NEON, wasm-simd128, AVX with `target-cpu=native`, ...). The SIMD backends also
//! call them for shapes outside their fast paths.
//!
//! The three dense kernels are one GEMM, `C = epilogue(A @ B)`, in the scheme of the hand-written
//! SIMD kernels: a register-blocked tile of `M` rows x `N` columns of `C` (fixed-size arrays)
//! stays in registers for a whole reduction block, LLVM vectorises it along the columns, and the
//! epilogue (bias + ReLU, the ReLU mask of the previous layer, or accumulation into `C`) is fused
//! into the store. Row tiles are outer, column tiles inner; row and column tails use smaller
//! tiles, so any shape works.
//!
//! Unlike the SIMD kernels, A is packed: for each row tile and reduction block its `M` values per
//! reduction step are copied into one contiguous `[f32; M]`. Reading them straight from `M`
//! strided rows makes LLVM's SLP vectoriser give up on the tile in some inlining contexts and
//! targets; from a contiguous panel it vectorises reliably. Packing also makes the transposed
//! A of the weight gradient (`a_prev^T`) just another way to fill the panel.

use super::Activation;

/// Rows of a full tile.
const MR: usize = 4;
/// Columns of a full tile: 8 lanes (two 128-bit vectors) where 16 vector registers hold the
/// tile; 16 lanes on the 32-register aarch64 (as the NEON kernel) and where AVX doubles the
/// vector width.
const NR: usize = if cfg!(any(target_arch = "aarch64", target_feature = "avx")) {
    16
} else {
    8
};
/// Reduction steps per block: the packed A panel (`MR x KC`) lives on the stack, and the weight
/// gradient keeps a block of rows in cache while it sweeps all tiles.
const KC: usize = 128;

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

/// The `N` elements of `row` from `j0`.
#[inline(always)]
fn part<const N: usize>(row: &[f32], j0: usize) -> [f32; N] {
    row[j0..j0 + N].try_into().unwrap()
}

/// The left operand of the GEMM.
trait Lhs {
    /// `panel[p] = [A[i0 + i][k0 + p] for i in 0..M]`.
    fn pack<const M: usize>(&self, i0: usize, k0: usize, panel: &mut [[f32; M]]);
}

/// A row-major `[m x kk]`.
struct Rows<'a> {
    a: &'a [f32],
    kk: usize,
}

impl Lhs for Rows<'_> {
    #[inline(always)]
    fn pack<const M: usize>(&self, i0: usize, k0: usize, panel: &mut [[f32; M]]) {
        for i in 0..M {
            let row = &self.a[(i0 + i) * self.kk + k0..][..panel.len()];
            for (column, &v) in panel.iter_mut().zip(row) {
                column[i] = v;
            }
        }
    }
}

/// The transpose of a row-major `[kk x m]`.
struct Transposed<'a> {
    a: &'a [f32],
    m: usize,
}

impl Lhs for Transposed<'_> {
    #[inline(always)]
    fn pack<const M: usize>(&self, i0: usize, k0: usize, panel: &mut [[f32; M]]) {
        let rows = self.a[k0 * self.m..].chunks_exact(self.m);
        for (column, row) in panel.iter_mut().zip(rows) {
            *column = part(row, i0);
        }
    }
}

/// What the GEMM does with a tile of `C` before and after the reduction.
trait Epilogue {
    /// The tile row `i` before the first reduction step; `c` is its current value.
    fn init<const N: usize>(&self, i: usize, j0: usize, c: &[f32; N]) -> [f32; N];
    /// The tile row `i` to store after the last reduction step.
    fn finish<const N: usize>(&self, i: usize, j0: usize, acc: [f32; N]) -> [f32; N];
}

/// `C = bias + A @ B`, then ReLU if `RELU` (a runtime flag would make LLVM unswitch the tile
/// loops and give up vectorising them).
struct Bias<'a, const RELU: bool>(&'a [f32]);

impl<const RELU: bool> Epilogue for Bias<'_, RELU> {
    #[inline(always)]
    fn init<const N: usize>(&self, _: usize, _: usize, _: &[f32; N]) -> [f32; N] {
        [0.0; N]
    }
    #[inline(always)]
    fn finish<const N: usize>(&self, _: usize, j0: usize, acc: [f32; N]) -> [f32; N] {
        let bias: [f32; N] = part(self.0, j0);
        // adding the bias here rather than starting from it keeps the SLP vectoriser on track
        std::array::from_fn(|j| {
            let v = acc[j] + bias[j];
            if RELU {
                v.max(0.0)
            } else {
                v
            }
        })
    }
}

/// `C = (A @ B) * (source > 0)`, with `source` shaped like `C`.
struct ReluMask<'a> {
    source: &'a [f32],
    n: usize,
}

impl Epilogue for ReluMask<'_> {
    #[inline(always)]
    fn init<const N: usize>(&self, _: usize, _: usize, _: &[f32; N]) -> [f32; N] {
        [0.0; N]
    }
    #[inline(always)]
    fn finish<const N: usize>(&self, i: usize, j0: usize, acc: [f32; N]) -> [f32; N] {
        let source: [f32; N] = part(&self.source[i * self.n..], j0);
        std::array::from_fn(|j| if source[j] > 0.0 { acc[j] } else { 0.0 })
    }
}

/// `C += A @ B`.
struct Accumulate;

impl Epilogue for Accumulate {
    #[inline(always)]
    fn init<const N: usize>(&self, _: usize, _: usize, c: &[f32; N]) -> [f32; N] {
        *c
    }
    #[inline(always)]
    fn finish<const N: usize>(&self, _: usize, _: usize, acc: [f32; N]) -> [f32; N] {
        acc
    }
}

/// `C[m x n] = epilogue(A[m x kk] @ B[kk x n])`, all row-major.
struct Gemm<'a, L> {
    lhs: L,
    m: usize,
    kk: usize,
    b: &'a [f32],
    n: usize,
    c: &'a mut [f32],
}

impl<L: Lhs> Gemm<'_, L> {
    fn run(mut self, epilogue: impl Epilogue) {
        let epilogue = &epilogue;
        let mut panel = [0.0f32; MR * KC];
        // one (empty) block if kk == 0, so that the epilogue still runs
        let mut k0 = 0;
        loop {
            let kc = KC.min(self.kk - k0);
            let mut i0 = 0;
            while i0 + MR <= self.m {
                self.rows::<MR>(i0, k0, kc, &mut panel, epilogue);
                i0 += MR;
            }
            if i0 + 2 <= self.m {
                self.rows::<2>(i0, k0, kc, &mut panel, epilogue);
                i0 += 2;
            }
            if i0 < self.m {
                self.rows::<1>(i0, k0, kc, &mut panel, epilogue);
            }
            k0 += kc;
            if k0 >= self.kk {
                break;
            }
        }
    }

    /// Reduction block `k0..k0 + kc` of rows `i0..i0 + M`.
    #[inline(always)]
    fn rows<const M: usize>(
        &mut self,
        i0: usize,
        k0: usize,
        kc: usize,
        panel: &mut [f32],
        epilogue: &impl Epilogue,
    ) {
        let panel = &mut panel.as_chunks_mut::<M>().0[..kc];
        self.lhs.pack(i0, k0, panel);
        let panel = &*panel;
        let mut j0 = 0;
        while j0 + NR <= self.n {
            self.tile::<M, NR>(i0, j0, k0, panel, epilogue);
            j0 += NR;
        }
        if NR > 8 && j0 + 8 <= self.n {
            self.tile::<M, 8>(i0, j0, k0, panel, epilogue);
            j0 += 8;
        }
        while j0 + 4 <= self.n {
            self.tile::<M, 4>(i0, j0, k0, panel, epilogue);
            j0 += 4;
        }
        for j in j0..self.n {
            self.tile::<M, 1>(i0, j, k0, panel, epilogue);
        }
    }

    /// `C[i0.., j0..]` (`M x N`) over the reduction block `k0..k0 + panel.len()`.
    #[inline(always)]
    fn tile<const M: usize, const N: usize>(
        &mut self,
        i0: usize,
        j0: usize,
        k0: usize,
        panel: &[[f32; M]],
        epilogue: &impl Epilogue,
    ) {
        let n = self.n;
        let (first, last) = (k0 == 0, k0 + panel.len() == self.kk);
        let mut acc: [[f32; N]; M] = std::array::from_fn(|i| {
            let c = part(&self.c[(i0 + i) * n..], j0);
            if first {
                epilogue.init(i0 + i, j0, &c)
            } else {
                c
            }
        });
        let b_rows = self.b[k0 * n..].chunks_exact(n);
        for (a, b_row) in panel.iter().zip(b_rows) {
            let b: [f32; N] = part(b_row, j0);
            for (acc_row, &a_i) in acc.iter_mut().zip(a) {
                for (c, &b_j) in acc_row.iter_mut().zip(&b) {
                    *c = madd(a_i, b_j, *c);
                }
            }
        }
        for (i, acc_row) in acc.into_iter().enumerate() {
            let value = if last {
                epilogue.finish(i0 + i, j0, acc_row)
            } else {
                acc_row
            };
            self.c[(i0 + i) * n + j0..][..N].copy_from_slice(&value);
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
    let gemm = Gemm {
        lhs: Rows { a: x, kk: k },
        m: rows,
        kk: k,
        b: &wt[..k * n],
        n,
        c: &mut out[..rows * n],
    };
    let bias = &bias[..n];
    match activation {
        Activation::Relu => gemm.run(Bias::<true>(bias)),
        Activation::Linear => gemm.run(Bias::<false>(bias)),
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
    Gemm {
        lhs: Rows { a: dz, kk: n },
        m: rows,
        kk: n,
        b: &w[..n * k],
        n: k,
        c: &mut d_prev[..rows * k],
    }
    .run(ReluMask {
        source: &a_prev[..rows * k],
        n: k,
    })
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
    if n == 0 {
        return;
    }
    Gemm {
        lhs: Transposed {
            a: &a_prev[..rows * k],
            m: k,
        },
        m: k,
        kk: rows,
        b: &dz[..rows * n],
        n,
        c: &mut grad_wt[..k * n],
    }
    .run(Accumulate);
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
    let (w, grad_w) = (&w[..k], &mut grad_w[..k]);
    let mut loss = 0.0f32;
    for (r, &label) in y[..rows].iter().enumerate() {
        let a_row = &a[r * k..][..k];
        let logit = b + dot(a_row, w);
        // one exp for both the stable loss and the sigmoid
        let e = (-logit.abs()).exp();
        loss += logit.max(0.0) - logit * label + e.ln_1p();
        let sigmoid = if logit >= 0.0 { 1.0 } else { e } / (1.0 + e);
        let d = (sigmoid - label) * loss_scale;
        *grad_b += d;
        let dz_row = &mut dz_prev[r * k..][..k];
        for (((g, dz), &x), &wv) in grad_w.iter_mut().zip(dz_row).zip(a_row).zip(w) {
            *g = madd(d, x, *g);
            *dz = if x > 0.0 { d * wv } else { 0.0 };
        }
    }
    loss
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
    for (r, logit) in logits[..rows].iter_mut().enumerate() {
        *logit = b + dot(&a[r * k..][..k], &w[..k]);
    }
}
