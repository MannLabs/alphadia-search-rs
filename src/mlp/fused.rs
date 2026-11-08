//! Fused prediction: rows through every layer of a ReLU network to its single output logit.
//!
//! [`Network`] packs the layers once, padding every hidden width to a multiple of [`PAD`] with
//! zero weights and biases. The padded units stay at `relu(0) = 0` and carry zero weights into the
//! next layer, so any widths run on the SIMD tiles of every backend, without masks or a scalar
//! fallback. [`Network::logits`] then splits the rows over the rayon threads and carries each
//! block of [`BLOCK_ROWS`] rows through all layers in two buffers on the stack: the activations
//! never leave the L1 cache, and nothing is allocated after packing.
//!
//! Weights and activations start on cache lines and padded rows keep them there, so no full-width
//! vector load or store splits a cache line (up to 20 % of the AVX-512 kernel time).

use super::Activation;
use rayon::prelude::*;

/// One dense ReLU layer: `max(bias + x @ wt, 0)` with `wt [inputs x n]`.
#[derive(Clone, Copy, Debug)]
pub struct Dense<'a> {
    pub wt: &'a [f32],
    pub bias: &'a [f32],
    pub n: usize,
}

/// Hidden widths are padded to a multiple of this, which every column tile divides
/// (AVX-512: 32, AVX2 and NEON: 16).
const PAD: usize = 32;
/// Rows a block carries through all layers, a multiple of the row tiles (AVX-512: 8, NEON: 4).
const BLOCK_ROWS: usize = 64;
/// Floats per activation buffer of a block (32 KB): two of them stay in L1 for widths up to 128.
const BLOCK_FLOATS: usize = 8192;
/// Rows a rayon task predicts; large enough that a task's stack buffers are noise.
const CHUNK_ROWS: usize = 8192;

/// Floats per cache line.
const LINE: usize = 16;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct CacheLine([f32; LINE]);

/// Zero-initialised floats that start on a cache line.
struct Aligned {
    lines: Vec<CacheLine>,
    len: usize,
}

impl Aligned {
    fn zeros(len: usize) -> Self {
        Aligned {
            lines: vec![CacheLine([0.0; LINE]); len.div_ceil(LINE)],
            len,
        }
    }

    fn as_slice(&self) -> &[f32] {
        // SAFETY: `CacheLine` is `repr(C)` around `[f32; LINE]`, so the lines are contiguous
        // floats, and they hold at least `len` of them.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as in `as_slice`, borrowed mutably
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast(), self.len) }
    }
}

/// The activations of one block, on the stack.
#[repr(C, align(64))]
struct BlockBuffer([f32; BLOCK_FLOATS]);

struct Layer {
    /// `[inputs x n]`, inputs padded as the previous layer's `n` (except the first layer)
    wt: Aligned,
    bias: Aligned,
    n: usize,
}

/// A ReLU network with one output logit, packed for fused prediction.
pub struct Network {
    n_inputs: usize,
    layers: Vec<Layer>,
    /// output weights, padded like the last hidden layer
    w_out: Vec<f32>,
    b_out: f32,
    /// rows per block: as many as fit the widest layer into one buffer, at most [`BLOCK_ROWS`]
    block: usize,
}

impl Network {
    /// Packs `hidden` (in order, the first reading `n_inputs` features) and the output layer
    /// `logit = b_out + w_out . h`.
    pub fn new(n_inputs: usize, hidden: &[Dense], w_out: &[f32], b_out: f32) -> Self {
        let mut layers = Vec::with_capacity(hidden.len());
        let (mut k, mut k_pad) = (n_inputs, n_inputs);
        for layer in hidden {
            let n = layer.n;
            let n_pad = n.next_multiple_of(PAD);
            assert!(
                layer.wt.len() >= k * n && layer.bias.len() >= n,
                "layer too small"
            );
            let mut wt = Aligned::zeros(k_pad * n_pad);
            for (row, packed) in layer.wt[..k * n]
                .chunks_exact(n)
                .zip(wt.as_mut_slice().chunks_exact_mut(n_pad))
            {
                packed[..n].copy_from_slice(row);
            }
            let mut bias = Aligned::zeros(n_pad);
            bias.as_mut_slice()[..n].copy_from_slice(&layer.bias[..n]);
            layers.push(Layer { wt, bias, n: n_pad });
            (k, k_pad) = (n, n_pad);
        }
        assert!(w_out.len() >= k, "output weights too small");
        let mut w_out_pad = vec![0.0; k_pad.max(1)];
        w_out_pad[..k].copy_from_slice(&w_out[..k]);
        let width = layers.iter().map(|l| l.n).max().unwrap_or(1);
        Network {
            n_inputs,
            layers,
            w_out: w_out_pad,
            b_out,
            block: (BLOCK_FLOATS / width).clamp(1, BLOCK_ROWS),
        }
    }

    /// `logits[r]` of every row of `x [logits.len() x n_inputs]`, in parallel over rayon.
    pub fn logits(&self, x: &[f32], logits: &mut [f32]) {
        let k = self.n_inputs;
        assert_eq!(x.len(), logits.len() * k, "x does not match the logits");
        logits
            .par_chunks_mut(CHUNK_ROWS)
            .zip(x.par_chunks(CHUNK_ROWS * k))
            .for_each(|(out, x)| self.logits_serial(x, out));
    }

    /// [`Network::logits`] on the calling thread.
    pub fn logits_serial(&self, x: &[f32], logits: &mut [f32]) {
        let k = self.n_inputs;
        let backend = crate::simd::get_backend();
        let floats = self.block * self.layers.iter().map(|l| l.n).max().unwrap_or(1);
        let mut stack = [
            BlockBuffer([0.0; BLOCK_FLOATS]),
            BlockBuffer([0.0; BLOCK_FLOATS]),
        ];
        let (mut heap_a, mut heap_b);
        let (mut src, mut dst): (&mut [f32], &mut [f32]) = if floats <= BLOCK_FLOATS {
            let [a, b] = &mut stack;
            (&mut a.0, &mut b.0)
        } else {
            // a single row is wider than a stack buffer
            (heap_a, heap_b) = (Aligned::zeros(floats), Aligned::zeros(floats));
            (heap_a.as_mut_slice(), heap_b.as_mut_slice())
        };
        for (x, out) in x.chunks(self.block * k).zip(logits.chunks_mut(self.block)) {
            let m = out.len();
            let mut inputs = k;
            for (l, layer) in self.layers.iter().enumerate() {
                let input = if l == 0 { x } else { &src[..m * inputs] };
                let n = layer.n;
                backend.mlp_dense_forward(
                    input,
                    m,
                    inputs,
                    layer.wt.as_slice(),
                    n,
                    layer.bias.as_slice(),
                    Activation::Relu,
                    &mut dst[..m * n],
                );
                std::mem::swap(&mut src, &mut dst);
                inputs = n;
            }
            let last = if self.layers.is_empty() {
                x
            } else {
                &src[..m * inputs]
            };
            backend.mlp_output_layer_logits(last, m, inputs, &self.w_out, self.b_out, out);
        }
    }
}
