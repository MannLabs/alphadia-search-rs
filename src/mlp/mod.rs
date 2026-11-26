//! A small feed-forward ReLU network with one output logit, the network of the FDR classifier.
//! [`Network`] trains on row chunks and predicts with fused layer kernels.
//!
//! All matrices are row-major `f32`. A dense layer with `k` inputs and `n` outputs keeps its
//! weights transposed, `wt[k x n]`, so the forward pass streams contiguous weight rows.
//! The network gets its kernels from the SIMD backend registry ([`crate::simd`]). The default
//! kernels are the portable kernels of [`scalar`]. The NEON, AVX2 and AVX-512 backends override
//! them. The tests compare every backend with the naive loops in `reference`.

mod aligned;
pub(crate) mod neon;
mod network;
#[cfg(test)]
mod reference;
pub(crate) mod scalar;
// The kernel template comes before its instantiations, because its macros must be in scope there.
#[cfg(target_arch = "x86_64")]
#[macro_use]
mod x86;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

pub(crate) use aligned::AlignedBuf;
pub(crate) use network::{Network, TrainWorkspace};

/// What a forward kernel applies to `x @ wt` before storing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `bias + x @ wt`
    Linear,
    /// `max(bias + x @ wt, 0)`
    Relu,
}

/// The minimum slice lengths of the layer kernels. The SIMD kernels index through raw pointers,
/// so each public SIMD kernel checks its arguments with these functions first.
pub(crate) mod shapes {
    fn check(values: usize, expected: usize, name: &str) {
        assert!(
            values >= expected,
            "{name} has {values} elements, expected at least {expected}"
        );
    }

    pub fn dense_forward(
        x: &[f32],
        rows: usize,
        k: usize,
        wt: &[f32],
        n: usize,
        bias: &[f32],
        out: &[f32],
    ) {
        check(x.len(), rows * k, "x");
        check(wt.len(), k * n, "wt");
        check(bias.len(), n, "bias");
        check(out.len(), rows * n, "out");
    }

    pub fn dense_backward_input(
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &[f32],
    ) {
        check(dz.len(), rows * n, "dz");
        check(w.len(), n * k, "w");
        check(a_prev.len(), rows * k, "a_prev");
        check(d_prev.len(), rows * k, "d_prev");
    }

    pub fn dense_weight_grad(
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &[f32],
        grad_b: &[f32],
    ) {
        check(a_prev.len(), rows * k, "a_prev");
        check(dz.len(), rows * n, "dz");
        check(grad_wt.len(), k * n, "grad_wt");
        check(grad_b.len(), n, "grad_b");
    }

    pub fn output_layer(
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        per_row: &[f32],
        dz_prev: Option<&[f32]>,
        grad_w: Option<&[f32]>,
    ) {
        check(a.len(), rows * k, "a");
        check(w.len(), k, "w");
        check(per_row.len(), rows, "y / logits");
        if let Some(dz_prev) = dz_prev {
            check(dz_prev.len(), rows * k, "dz_prev");
        }
        if let Some(grad_w) = grad_w {
            check(grad_w.len(), k, "grad_w");
        }
    }
}

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;
