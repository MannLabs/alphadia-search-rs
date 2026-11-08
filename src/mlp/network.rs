//! A whole ReLU network with one output logit: its parameter layout and one training step over a
//! chunk of rows (forward, BCE-with-logits loss, backward and gradient accumulation).
//!
//! Training runs block by block: [`TRAIN_ROW_BLOCK`] rows go through every layer forward and
//! back before the next block starts, so their activations and deltas stay in L1/L2 between the
//! layers and the workspace does not grow with the chunk.

use super::Activation;

/// Rows of a chunk that pass through the whole network together.
pub const TRAIN_ROW_BLOCK: usize = 128;

/// Parameters of a ReLU network with one output logit.
///
/// `dims` are the layer widths `[inputs, hidden..., 1]`. `params` is flat, layer by layer: the
/// weights of each hidden layer transposed (`wt [k x n]`) followed by its bias, then the output
/// weights `[last hidden]` and the output bias. `back` has the same layout with every hidden
/// weight block untransposed (`w [n x k]`, see [`Network::backward_weights`]); the backward pass
/// streams its rows.
#[derive(Clone, Copy)]
pub struct Network<'a> {
    dims: &'a [usize],
    params: &'a [f32],
    back: &'a [f32],
}

impl<'a> Network<'a> {
    pub fn new(dims: &'a [usize], params: &'a [f32], back: &'a [f32]) -> Self {
        assert!(
            dims.len() >= 3 && dims[dims.len() - 1] == 1,
            "a network needs inputs, a hidden layer and one output, got {dims:?}"
        );
        let size = Self::param_count(dims);
        assert!(
            params.len() >= size && back.len() >= size,
            "parameters too short"
        );
        Network { dims, params, back }
    }

    /// Length of the flat parameter vector.
    pub fn param_count(dims: &[usize]) -> usize {
        dims.windows(2).map(|d| d[0] * d[1] + d[1]).sum()
    }

    /// Writes the backward copy of `params` into `back`: every hidden weight block untransposed.
    pub fn backward_weights(dims: &[usize], params: &[f32], back: &mut [f32]) {
        let mut offset = 0;
        for d in dims[..dims.len() - 1].windows(2) {
            let (k, n) = (d[0], d[1]);
            let (wt, w) = (
                &params[offset..offset + k * n],
                &mut back[offset..offset + k * n],
            );
            for (p, wt_row) in wt.chunks_exact(n).enumerate() {
                for (j, &v) in wt_row.iter().enumerate() {
                    w[j * k + p] = v;
                }
            }
            offset += k * n + n;
        }
    }

    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    /// Number of hidden layers.
    pub fn depth(&self) -> usize {
        self.dims.len() - 2
    }

    /// Offset of the weights of layer `l` (the output layer is `depth()`).
    pub fn weight_offset(&self, l: usize) -> usize {
        self.dims[..=l].windows(2).map(|d| d[0] * d[1] + d[1]).sum()
    }

    /// Offset of the bias of layer `l`.
    pub fn bias_offset(&self, l: usize) -> usize {
        self.weight_offset(l) + self.dims[l] * self.dims[l + 1]
    }

    /// Transposed weights `[k x n]` of layer `l` (the output layer's are `[k]`).
    pub fn wt(&self, l: usize) -> &'a [f32] {
        &self.params[self.weight_offset(l)..self.bias_offset(l)]
    }

    /// Untransposed weights `[n x k]` of hidden layer `l`.
    pub fn w(&self, l: usize) -> &'a [f32] {
        &self.back[self.weight_offset(l)..self.bias_offset(l)]
    }

    pub fn bias(&self, l: usize) -> &'a [f32] {
        let at = self.bias_offset(l);
        &self.params[at..at + self.dims[l + 1]]
    }
}

/// Per-thread buffers of [`train_chunk`](super::train_chunk), sized once for a network shape.
pub struct TrainWorkspace {
    /// hidden activations of one row block, layer after layer (`TRAIN_ROW_BLOCK` rows each)
    pub(crate) activations: Vec<f32>,
    /// gradients at the hidden pre-activations of one row block, laid out like `activations`
    pub(crate) deltas: Vec<f32>,
    /// zero bias of the input-gradient product
    pub(crate) zeros: Vec<f32>,
}

impl TrainWorkspace {
    pub fn new(dims: &[usize]) -> Self {
        let hidden: usize = dims[1..dims.len() - 1].iter().sum();
        TrainWorkspace {
            activations: vec![0.0; TRAIN_ROW_BLOCK * hidden],
            deltas: vec![0.0; TRAIN_ROW_BLOCK * hidden],
            zeros: vec![0.0; dims[0]],
        }
    }
}

/// Start of hidden layer `l` in the workspace buffers.
pub(crate) fn block_offset(dims: &[usize], l: usize) -> usize {
    TRAIN_ROW_BLOCK * dims[1..=l].iter().sum::<usize>()
}

/// The `rows x dims[l + 1]` block of hidden layer `l - 1` (read) and of layer `l` (written).
pub(crate) fn layer_pair<'b>(
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

/// The `rows x dims[l + 1]` block of hidden layer `l`.
pub(crate) fn layer<'b>(buf: &'b [f32], dims: &[usize], l: usize, rows: usize) -> &'b [f32] {
    &buf[block_offset(dims, l)..][..rows * dims[l + 1]]
}

/// Per-layer kernels a training step is composed of; see the free functions of [`crate::mlp`]
/// for their contracts.
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
    /// `w` and `wt` are the same weights as `[n x k]` and `[k x n]`.
    #[allow(clippy::too_many_arguments)]
    fn dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        wt: &[f32],
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
}

/// [`train_chunk`](super::train_chunk) composed of per-layer kernels.
#[allow(clippy::too_many_arguments)]
pub fn train_chunk_layered<K: LayerKernels + ?Sized>(
    kernels: &K,
    net: &Network,
    x: &[f32],
    y: &[f32],
    rows: usize,
    loss_scale: f32,
    ws: &mut TrainWorkspace,
    grad: &mut [f32],
    mut input_grad: Option<&mut [f32]>,
) {
    let dims = net.dims();
    let depth = net.depth();
    for r0 in (0..rows).step_by(TRAIN_ROW_BLOCK) {
        let m = TRAIN_ROW_BLOCK.min(rows - r0);
        let x = &x[r0 * dims[0]..(r0 + m) * dims[0]];
        for l in 0..depth {
            let (prev, out) = layer_pair(&mut ws.activations, dims, l, m);
            let input = if l == 0 { x } else { prev };
            let (k, n) = (dims[l], dims[l + 1]);
            kernels.dense_forward(
                input,
                m,
                k,
                net.wt(l),
                n,
                net.bias(l),
                Activation::Relu,
                out,
            );
        }
        {
            let (head, tail) = grad.split_at_mut(net.bias_offset(depth));
            let (_, dz) = layer_pair(&mut ws.deltas, dims, depth - 1, m);
            kernels.output_layer_train(
                layer(&ws.activations, dims, depth - 1, m),
                m,
                dims[depth],
                net.wt(depth),
                net.bias(depth)[0],
                &y[r0..r0 + m],
                loss_scale,
                dz,
                &mut head[net.weight_offset(depth)..],
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
            let (head, tail) = grad.split_at_mut(net.bias_offset(l));
            kernels.dense_weight_grad(
                input,
                m,
                k,
                layer(&ws.deltas, dims, l, m),
                n,
                &mut head[net.weight_offset(l)..],
                &mut tail[..n],
            );
            if l > 0 {
                let (d_prev, dz) = upper_pair(&mut ws.deltas, dims, l, m);
                let a_prev = layer(&ws.activations, dims, l - 1, m);
                kernels.dense_backward_input(dz, m, n, net.w(l), net.wt(l), k, a_prev, d_prev);
            } else if let Some(input_grad) = input_grad.as_deref_mut() {
                // no activation before the input: d_x = dz @ W, a linear layer with weights w
                let out = &mut input_grad[r0 * k..(r0 + m) * k];
                let dz = layer(&ws.deltas, dims, 0, m);
                kernels.dense_forward(dz, m, n, net.w(0), k, &ws.zeros, Activation::Linear, out);
            }
        }
    }
}

/// The deltas of hidden layer `l - 1` (written) and of layer `l` (read), `l > 0`.
pub(crate) fn upper_pair<'b>(
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
