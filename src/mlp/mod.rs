//! A small feed-forward ReLU network with one output logit, the network of the FDR classifier.
//! [`Network`] trains on row chunks and predicts with fused layer kernels.
//!
//! All matrices are row-major `f32`. A dense layer with `k` inputs and `n` outputs keeps its
//! weights transposed, `wt[k x n]`, so the forward pass streams contiguous weight rows.
//! The network gets its kernels from the SIMD backend registry ([`crate::simd`]). The default
//! kernels are the portable kernels of [`scalar`]. A backend can override them with its own
//! kernels. The tests compare every backend with the naive loops in `reference`.

mod aligned;
mod network;
#[cfg(test)]
mod reference;
pub(crate) mod scalar;

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

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;
