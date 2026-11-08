//! A whole ReLU network with one output logit: one training step over a chunk of rows (forward,
//! BCE-with-logits loss, backward and gradient accumulation) and fused prediction.
//!
//! Both run block by block, so activations and deltas stay in L1/L2 between the layers and no
//! buffer grows with the rows: training carries [`TRAIN_ROW_BLOCK`] rows through every layer
//! forward and back before the next block starts; prediction carries blocks of up to 64 rows
//! through every layer in two cache-line aligned stack buffers.
//!
//! Every pass is composed of the per-layer kernels of a [`LayerKernels`] implementation: any
//! SIMD backend of the registry, or a BLAS.

use super::{Activation, AlignedBuf};
use rayon::prelude::*;

/// Rows of a training chunk that pass through the whole network together.
pub(crate) const TRAIN_ROW_BLOCK: usize = 128;

/// Prediction pads hidden widths to a multiple of this, which every column tile divides
/// (AVX-512: 32, AVX2 and NEON: 16), so that every activation row starts on a cache line.
const PREDICT_PAD: usize = 32;
/// Most rows a prediction block carries through all layers (a multiple of the row tiles).
const PREDICT_BLOCK_ROWS: usize = 64;
/// Floats per prediction activation buffer (32 KB): two stay in L1 for widths up to 128.
const PREDICT_BLOCK_FLOATS: usize = 8192;
/// Rows a rayon task predicts; large enough that a task's stack buffers are noise.
const PREDICT_CHUNK_ROWS: usize = 8192;

/// The per-layer kernels the network passes are composed of; see the free functions of
/// [`crate::mlp`] for their contracts.
pub trait LayerKernels {
    #[allow(clippy::too_many_arguments)]
    fn dense_forward(
        &self,
        x: &[f32],
        rows: usize,
        k: usize,
        wt: &[f32],
        n: usize,
        bias: &[f32],
        activation: Activation,
        out: &mut [f32],
    );
    #[allow(clippy::too_many_arguments)]
    fn dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &mut [f32],
    );
    #[allow(clippy::too_many_arguments)]
    fn dense_weight_grad(
        &self,
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &mut [f32],
        grad_b: &mut [f32],
    );
    #[allow(clippy::too_many_arguments)]
    fn output_layer_train(
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
    );
    fn output_layer_logits(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        logits: &mut [f32],
    );
}

/// A ReLU network with one output logit and its parameters on cache lines.
///
/// `dims` are the layer widths `[inputs, hidden..., 1]`. The parameters are flat, layer by layer:
/// the weights of each hidden layer transposed (`wt [k x n]`) followed by its bias, then the
/// output weights `[last hidden]` and the output bias. The network also keeps every hidden weight
/// block untransposed (`w [n x k]`) for the backward pass, which streams its rows.
#[derive(Clone)]
pub struct Network {
    dims: Vec<usize>,
    params: AlignedBuf,
    /// the layout of `params` with the hidden weights untransposed
    back: AlignedBuf,
}

impl Network {
    /// A network with `dims` and the first [`Network::param_count`] values of `params`.
    pub fn new(dims: &[usize], params: &[f32]) -> Self {
        assert!(
            dims.len() >= 3 && dims[dims.len() - 1] == 1 && !dims.contains(&0),
            "a network needs inputs, hidden layers and one output, got {dims:?}"
        );
        let size = Self::param_count(dims);
        assert!(
            params.len() >= size,
            "{} parameters, the network needs {size}",
            params.len()
        );
        let mut net = Network {
            dims: dims.to_vec(),
            params: AlignedBuf::from_slice(&params[..size]),
            back: AlignedBuf::zeros(size),
        };
        net.untranspose();
        net
    }

    /// Length of the flat parameter vector of a network with `dims`.
    pub fn param_count(dims: &[usize]) -> usize {
        dims.windows(2).map(|d| d[0] * d[1] + d[1]).sum()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Changes the parameters in place, e.g. by an optimizer step.
    pub fn update(&mut self, change: impl FnOnce(&mut [f32])) {
        change(&mut self.params);
        self.untranspose();
    }

    /// One training step over the rows of `x [rows x inputs]` with binary labels `y [rows]`:
    /// forward, BCE-with-logits loss scaled by `loss_scale`, backward. Adds the parameter
    /// gradients to `grad` (laid out as the parameters) and, if given, writes the gradient
    /// w.r.t. the input to `input_grad [rows x inputs]`.
    #[allow(clippy::too_many_arguments)]
    pub fn train_chunk<K: LayerKernels + ?Sized>(
        &self,
        kernels: &K,
        x: &[f32],
        y: &[f32],
        loss_scale: f32,
        ws: &mut TrainWorkspace,
        grad: &mut [f32],
        mut input_grad: Option<&mut [f32]>,
    ) {
        let dims = &self.dims[..];
        let (rows, inputs, depth) = (y.len(), dims[0], self.depth());
        assert_eq!(x.len(), rows * inputs, "x does not match the labels");
        assert_eq!(
            grad.len(),
            self.params.len(),
            "gradient of the wrong length"
        );
        if let Some(g) = input_grad.as_deref() {
            assert_eq!(g.len(), x.len(), "input gradient of the wrong length");
        }
        assert_eq!(ws.dims, dims, "workspace of another network");
        for r0 in (0..rows).step_by(TRAIN_ROW_BLOCK) {
            let m = TRAIN_ROW_BLOCK.min(rows - r0);
            let x = &x[r0 * inputs..(r0 + m) * inputs];
            for l in 0..depth {
                let (prev, out) = forward_pair(&mut ws.activations, dims, l, m);
                let input = if l == 0 { x } else { prev };
                let (k, n) = (dims[l], dims[l + 1]);
                kernels.dense_forward(
                    input,
                    m,
                    k,
                    self.wt(l),
                    n,
                    self.bias(l),
                    Activation::Relu,
                    out,
                );
            }
            {
                let (head, tail) = grad.split_at_mut(self.bias_offset(depth));
                let (_, dz) = forward_pair(&mut ws.deltas, dims, depth - 1, m);
                kernels.output_layer_train(
                    layer(&ws.activations, dims, depth - 1, m),
                    m,
                    dims[depth],
                    self.wt(depth),
                    self.bias(depth)[0],
                    &y[r0..r0 + m],
                    loss_scale,
                    dz,
                    &mut head[self.weight_offset(depth)..],
                    &mut tail[0],
                );
            }
            for l in (0..depth).rev() {
                let (k, n) = (dims[l], dims[l + 1]);
                let input = if l == 0 {
                    x
                } else {
                    layer(&ws.activations, dims, l - 1, m)
                };
                let (head, tail) = grad.split_at_mut(self.bias_offset(l));
                kernels.dense_weight_grad(
                    input,
                    m,
                    k,
                    layer(&ws.deltas, dims, l, m),
                    n,
                    &mut head[self.weight_offset(l)..],
                    &mut tail[..n],
                );
                if l > 0 {
                    let (d_prev, dz) = backward_pair(&mut ws.deltas, dims, l, m);
                    let a_prev = layer(&ws.activations, dims, l - 1, m);
                    kernels.dense_backward_input(dz, m, n, self.w(l), k, a_prev, d_prev);
                } else if let Some(input_grad) = input_grad.as_deref_mut() {
                    // no activation before the input: d_x = dz @ W, a linear layer with weights w
                    let out = &mut input_grad[r0 * k..(r0 + m) * k];
                    let dz = layer(&ws.deltas, dims, 0, m);
                    kernels.dense_forward(
                        dz,
                        m,
                        n,
                        self.w(0),
                        k,
                        &ws.zeros,
                        Activation::Linear,
                        out,
                    );
                }
            }
        }
    }

    /// The output logit of every row of `x [logits.len() x inputs]`, in parallel over rayon.
    pub fn logits<K: LayerKernels + Sync + ?Sized>(
        &self,
        kernels: &K,
        x: &[f32],
        logits: &mut [f32],
    ) {
        let inputs = self.dims[0];
        assert_eq!(
            x.len(),
            logits.len() * inputs,
            "x does not match the logits"
        );
        let packed = Packed::new(self);
        logits
            .par_chunks_mut(PREDICT_CHUNK_ROWS)
            .zip(x.par_chunks(PREDICT_CHUNK_ROWS * inputs))
            .for_each(|(out, x)| packed.logits(kernels, x, out));
    }

    /// Number of hidden layers.
    fn depth(&self) -> usize {
        self.dims.len() - 2
    }

    /// Offset of the weights of layer `l` (the output layer is `depth()`).
    fn weight_offset(&self, l: usize) -> usize {
        Self::param_count(&self.dims[..=l])
    }

    /// Offset of the bias of layer `l`.
    fn bias_offset(&self, l: usize) -> usize {
        self.weight_offset(l) + self.dims[l] * self.dims[l + 1]
    }

    /// Transposed weights `[k x n]` of layer `l` (the output layer's are `[k]`).
    fn wt(&self, l: usize) -> &[f32] {
        &self.params[self.weight_offset(l)..self.bias_offset(l)]
    }

    /// Untransposed weights `[n x k]` of hidden layer `l`.
    fn w(&self, l: usize) -> &[f32] {
        &self.back[self.weight_offset(l)..self.bias_offset(l)]
    }

    fn bias(&self, l: usize) -> &[f32] {
        let at = self.bias_offset(l);
        &self.params[at..at + self.dims[l + 1]]
    }

    /// Writes the backward copy of the parameters.
    fn untranspose(&mut self) {
        for l in 0..self.depth() {
            let (k, n, at) = (self.dims[l], self.dims[l + 1], self.weight_offset(l));
            let (wt, w) = (&self.params[at..at + k * n], &mut self.back[at..at + k * n]);
            for (p, wt_row) in wt.chunks_exact(n).enumerate() {
                for (j, &v) in wt_row.iter().enumerate() {
                    w[j * k + p] = v;
                }
            }
        }
    }
}

/// Per-thread buffers of [`Network::train_chunk`], sized once for a network shape.
pub struct TrainWorkspace {
    dims: Vec<usize>,
    /// hidden activations of one row block, layer after layer (`TRAIN_ROW_BLOCK` rows each)
    activations: AlignedBuf,
    /// gradients at the hidden pre-activations of one row block, laid out like `activations`
    deltas: AlignedBuf,
    /// zero bias of the input-gradient product
    zeros: Vec<f32>,
}

impl TrainWorkspace {
    pub fn new(net: &Network) -> Self {
        let dims = net.dims();
        let hidden: usize = dims[1..dims.len() - 1].iter().sum();
        TrainWorkspace {
            dims: dims.to_vec(),
            activations: AlignedBuf::zeros(TRAIN_ROW_BLOCK * hidden),
            deltas: AlignedBuf::zeros(TRAIN_ROW_BLOCK * hidden),
            zeros: vec![0.0; dims[0]],
        }
    }
}

/// Start of hidden layer `l` in the workspace buffers.
fn block_offset(dims: &[usize], l: usize) -> usize {
    TRAIN_ROW_BLOCK * dims[1..=l].iter().sum::<usize>()
}

/// The blocks of hidden layer `l - 1` (read; empty for `l == 0`) and of layer `l` (written).
fn forward_pair<'b>(
    buf: &'b mut [f32],
    dims: &[usize],
    l: usize,
    rows: usize,
) -> (&'b [f32], &'b mut [f32]) {
    let (lower, upper) = buf.split_at_mut(block_offset(dims, l));
    let prev = if l == 0 {
        &[][..]
    } else {
        &lower[block_offset(dims, l - 1)..][..rows * dims[l]]
    };
    (prev, &mut upper[..rows * dims[l + 1]])
}

/// The blocks of hidden layer `l - 1` (written) and of layer `l` (read), `l > 0`.
fn backward_pair<'b>(
    buf: &'b mut [f32],
    dims: &[usize],
    l: usize,
    rows: usize,
) -> (&'b mut [f32], &'b [f32]) {
    let (lower, upper) = buf.split_at_mut(block_offset(dims, l));
    (
        &mut lower[block_offset(dims, l - 1)..][..rows * dims[l]],
        &upper[..rows * dims[l + 1]],
    )
}

/// The `rows x dims[l + 1]` block of hidden layer `l`.
fn layer<'b>(buf: &'b [f32], dims: &[usize], l: usize, rows: usize) -> &'b [f32] {
    &buf[block_offset(dims, l)..][..rows * dims[l + 1]]
}

/// The layers of a network packed for prediction: hidden widths padded to a multiple of
/// [`PREDICT_PAD`] with zero weights and biases. The padded units stay at `relu(0) = 0` and carry
/// zero weights into the next layer, so any widths run on the full SIMD tiles, and every weight
/// and activation row starts on a cache line.
struct Packed {
    inputs: usize,
    /// `(wt [inputs x n], bias [n], n)`, inputs padded as the previous layer's `n` (except the first)
    hidden: Vec<(AlignedBuf, AlignedBuf, usize)>,
    /// output weights, padded like the last hidden layer
    w_out: AlignedBuf,
    b_out: f32,
    /// rows per block: as many as fit the widest layer into one buffer, at most
    /// [`PREDICT_BLOCK_ROWS`]
    block: usize,
}

/// The activations of one prediction block, on the stack.
#[repr(C, align(64))]
struct BlockBuffer([f32; PREDICT_BLOCK_FLOATS]);

impl Packed {
    fn new(net: &Network) -> Self {
        let dims = net.dims();
        let mut hidden = Vec::with_capacity(net.depth());
        let (mut k, mut k_pad) = (dims[0], dims[0]);
        for l in 0..net.depth() {
            let n = dims[l + 1];
            let n_pad = n.next_multiple_of(PREDICT_PAD);
            let mut wt = AlignedBuf::zeros(k_pad * n_pad);
            for (row, packed) in net.wt(l).chunks_exact(n).zip(wt.chunks_exact_mut(n_pad)) {
                packed[..n].copy_from_slice(row);
            }
            let mut bias = AlignedBuf::zeros(n_pad);
            bias[..n].copy_from_slice(net.bias(l));
            hidden.push((wt, bias, n_pad));
            (k, k_pad) = (n, n_pad);
        }
        let mut w_out = AlignedBuf::zeros(k_pad);
        w_out[..k].copy_from_slice(net.wt(net.depth()));
        let width = k_pad.max(hidden.iter().map(|h| h.2).max().unwrap_or(1));
        Packed {
            inputs: dims[0],
            hidden,
            w_out,
            b_out: net.bias(net.depth())[0],
            block: (PREDICT_BLOCK_FLOATS / width).clamp(1, PREDICT_BLOCK_ROWS),
        }
    }

    /// The logits of `x [logits.len() x inputs]` on the calling thread.
    fn logits<K: LayerKernels + ?Sized>(&self, kernels: &K, x: &[f32], logits: &mut [f32]) {
        let inputs = self.inputs;
        let mut stack = [
            BlockBuffer([0.0; PREDICT_BLOCK_FLOATS]),
            BlockBuffer([0.0; PREDICT_BLOCK_FLOATS]),
        ];
        let width = self.hidden.iter().map(|h| h.2).max().unwrap_or(1);
        let (mut heap_a, mut heap_b);
        let (mut src, mut dst): (&mut [f32], &mut [f32]) =
            if self.block * width <= PREDICT_BLOCK_FLOATS {
                let [a, b] = &mut stack;
                (&mut a.0, &mut b.0)
            } else {
                // a single row is wider than a stack buffer
                (heap_a, heap_b) = (AlignedBuf::zeros(width), AlignedBuf::zeros(width));
                (&mut heap_a[..], &mut heap_b[..])
            };
        for (x, out) in x
            .chunks(self.block * inputs)
            .zip(logits.chunks_mut(self.block))
        {
            let m = out.len();
            let mut k = inputs;
            for (l, (wt, bias, n)) in self.hidden.iter().enumerate() {
                let input = if l == 0 { x } else { &src[..m * k] };
                kernels.dense_forward(
                    input,
                    m,
                    k,
                    wt,
                    *n,
                    bias,
                    Activation::Relu,
                    &mut dst[..m * n],
                );
                std::mem::swap(&mut src, &mut dst);
                k = *n;
            }
            kernels.output_layer_logits(&src[..m * k], m, k, &self.w_out, self.b_out, out);
        }
    }
}
