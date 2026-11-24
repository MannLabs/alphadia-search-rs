//! The FDR classifier network: fully connected ReLU layers and one output logit, trained with
//! BCE-with-logits and Adam (torch semantics: L2 weight decay added to the gradient).
//!
//! Parameters live in one flat vector, layer by layer: hidden weights transposed (`[in x out]`)
//! followed by their bias, then the output weights `[last hidden]` and the output bias.
//! Each batch is split into one row chunk per rayon thread; chunks run forward and backward
//! independently and their gradients are summed.
//!
//! With `input_batch_norm`, the input passes a torch `BatchNorm1d` first (momentum 0.1, eps 1e-5):
//! training normalizes each minibatch with its own mean and biased variance and learns a scale and
//! shift per feature; prediction uses the running mean and unbiased variance, folded into the first
//! layer. Its parameters follow the output bias: scale, shift, running mean, running variance
//! (`n_features` each).

use super::ops::Backend;
use crate::mlp::{self, Network, TrainWorkspace};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const ADAM_BETA_1: f32 = 0.9;
const ADAM_BETA_2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
/// Row chunks are multiples of the 4-row SIMD kernel tile.
const ROW_ALIGN: usize = 4;
/// Rows per task of the parallel row shuffle.
const GATHER_ROWS: usize = 8192;
const BN_MOMENTUM: f32 = 0.1;
const BN_EPS: f32 = 1e-5;

pub struct TrainConfig {
    pub hidden_layers: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub backend: Backend,
    /// Decay the learning rate along a half cosine from `learning_rate` to zero over all minibatches of the fit
    /// (as torch `CosineAnnealingLR(T_max = steps, eta_min = 0)`, stepped once per minibatch).
    pub cosine_schedule: bool,
    /// Return the mean of the parameters over the minibatches of the last epoch instead of the last parameters.
    pub average_last_epoch: bool,
    /// Normalize the input with a batch norm layer (see the module docs).
    pub input_batch_norm: bool,
}

/// The flat parameter vector: the network's parameters ([`Network`]), then those of the input
/// batch norm, if any: scale, shift, running mean, running variance (`n_features` each).
pub struct Layout {
    dims: Vec<usize>, // input, hidden..., 1
    /// length of the network's parameters, where the batch norm parameters start
    n_net: usize,
    batch_norm: bool,
    pub size: usize,
}

impl Layout {
    pub fn new(
        n_features: usize,
        hidden_layers: &[usize],
        input_batch_norm: bool,
    ) -> Result<Self, String> {
        if n_features == 0 {
            return Err("the network needs at least one input feature".to_string());
        }
        if hidden_layers.is_empty() || hidden_layers.contains(&0) {
            return Err(format!(
                "hidden layers must be non-empty and positive, got {hidden_layers:?}"
            ));
        }
        let mut dims = vec![n_features];
        dims.extend_from_slice(hidden_layers);
        dims.push(1);
        let n_net = Network::param_count(&dims);
        Ok(Layout {
            dims,
            n_net,
            batch_norm: input_batch_norm,
            size: n_net + if input_batch_norm { 4 * n_features } else { 0 },
        })
    }

    /// Block `i` of the batch norm parameters: 0 scale, 1 shift, 2 running mean, 3 running
    /// variance.
    fn bn<'a>(&self, params: &'a [f32], i: usize) -> &'a [f32] {
        let k = self.dims[0];
        &params[self.n_net + i * k..self.n_net + (i + 1) * k]
    }

    /// torch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases.
    fn init_params(&self, rng: &mut StdRng) -> Vec<f32> {
        let mut params = vec![0.0f32; self.size];
        let mut offset = 0;
        for d in self.dims.windows(2) {
            let bound = 1.0 / (d[0] as f32).sqrt();
            let len = d[0] * d[1] + d[1];
            for p in &mut params[offset..offset + len] {
                *p = rng.random_range(-bound..bound);
            }
            offset += len;
        }
        if self.batch_norm {
            let k = self.dims[0];
            params[self.n_net..self.n_net + k].fill(1.0); // scale
            params[self.n_net + 3 * k..].fill(1.0); // running variance
        }
        params
    }
}

/// Buffers of one training thread, allocated once per fit.
struct ChunkState {
    ws: TrainWorkspace,
    /// gradients of the network parameters, then of the batch norm scale and shift
    grad: mlp::AlignedBuf,
    /// gradient at the batch norm output (only with the input batch norm)
    input_grad: Vec<f32>,
}

/// Gradients of one row chunk into `state.grad`; `x_hat` is the standardized input of the
/// batch norm (if any), `bn` its parameters.
fn train_chunk(
    backend: Backend,
    net: &Network,
    state: &mut ChunkState,
    x: &[f32],
    y: &[f32],
    loss_scale: f32,
    x_hat: Option<&[f32]>,
) {
    let ChunkState {
        ws,
        grad,
        input_grad,
    } = state;
    grad.fill(0.0);
    let (net_grad, bn_grad) = grad.split_at_mut(Network::param_count(net.dims()));
    let input_grad = &mut input_grad[..x_hat.map_or(0, |_| x.len())];
    let wanted = x_hat.is_some().then_some(&mut *input_grad);
    match backend {
        Backend::Simd => net.train_chunk(
            crate::simd::get_backend(),
            x,
            y,
            loss_scale,
            ws,
            net_grad,
            wanted,
        ),
        #[cfg(target_os = "macos")]
        Backend::Accelerate => net.train_chunk(
            &super::ops::Accelerate,
            x,
            y,
            loss_scale,
            ws,
            net_grad,
            wanted,
        ),
    }
    if let Some(x_hat) = x_hat {
        // the input has no parameters, so scale and shift get exact gradients from the
        // gradient at the batch norm output
        let k = net.dims()[0];
        let (scale, shift) = bn_grad.split_at_mut(k);
        for (d_row, h_row) in input_grad.chunks_exact(k).zip(x_hat.chunks_exact(k)) {
            for (p, (&d, &h)) in d_row.iter().zip(h_row).enumerate() {
                scale[p] += d * h;
                shift[p] += d;
            }
        }
    }
}

/// Batch norm of one training batch (torch `BatchNorm1d` in training mode): standardize every
/// feature of `x [rows x k]` with the batch mean and biased variance into `x_hat`, apply scale and
/// shift into `x_norm`, and move the running mean and (unbiased) variance towards the batch
/// statistics. `bn` holds scale, shift, running mean and running variance.
fn normalize_batch(bn: &mut [f32], x: &[f32], x_norm: &mut [f32], x_hat: &mut [f32]) {
    let k = bn.len() / 4;
    let rows = x.len() / k;
    let mut mean = vec![0.0f64; k];
    let mut sq = vec![0.0f64; k];
    for row in x.chunks_exact(k) {
        for p in 0..k {
            let v = f64::from(row[p]);
            mean[p] += v;
            sq[p] += v * v;
        }
    }
    let n = rows as f64;
    let (scale_shift, running) = bn.split_at_mut(2 * k);
    let (scale, shift) = scale_shift.split_at(k);
    let (running_mean, running_var) = running.split_at_mut(k);
    for p in 0..k {
        mean[p] /= n;
        let var = (sq[p] / n - mean[p] * mean[p]).max(0.0);
        let unbiased = if rows > 1 { var * n / (n - 1.0) } else { var };
        running_mean[p] = (1.0 - BN_MOMENTUM) * running_mean[p] + BN_MOMENTUM * mean[p] as f32;
        running_var[p] = (1.0 - BN_MOMENTUM) * running_var[p] + BN_MOMENTUM * unbiased as f32;
        sq[p] = 1.0 / (var + f64::from(BN_EPS)).sqrt(); // reuse as inverse standard deviation
    }
    for r in 0..rows {
        for p in 0..k {
            let h = ((f64::from(x[r * k + p]) - mean[p]) * sq[p]) as f32;
            x_hat[r * k + p] = h;
            x_norm[r * k + p] = scale[p] * h + shift[p];
        }
    }
}

/// The network's parameters with the prediction-time batch norm folded into the first layer.
///
/// The batch norm in torch evaluation mode (running mean and variance, scale and shift) is an
/// affine map `a * x + c` per feature: `wt'[p] = a[p] * wt[p]`, `bias' = bias + sum_p c[p] * wt[p]`.
fn fold_input_batch_norm(layout: &Layout, params: &[f32]) -> Vec<f32> {
    let (k, n) = (layout.dims[0], layout.dims[1]);
    let (scale, shift, mean, var) = (
        layout.bn(params, 0),
        layout.bn(params, 1),
        layout.bn(params, 2),
        layout.bn(params, 3),
    );
    let mut folded = params[..layout.n_net].to_vec();
    // layer 0: wt [k x n], then its bias
    let (wt, rest) = folded.split_at_mut(k * n);
    let mut bias: Vec<f64> = rest[..n].iter().map(|&b| f64::from(b)).collect();
    for p in 0..k {
        let a = scale[p] / (var[p] + BN_EPS).sqrt();
        let c = f64::from(shift[p]) - f64::from(a) * f64::from(mean[p]);
        for (w, b) in wt[p * n..(p + 1) * n].iter_mut().zip(&mut bias) {
            *b += c * f64::from(*w);
            *w *= a;
        }
    }
    for (folded, b) in rest[..n].iter_mut().zip(bias) {
        *folded = b as f32;
    }
    folded
}

/// One Adam step (torch semantics: L2 weight decay added to the gradient) on `params`.
fn adam_step(
    params: &mut [f32],
    grad: &[f32],
    (m, v): (&mut [f32], &mut [f32]),
    step_size: f32,
    inv_bias_c2_sqrt: f32,
    weight_decay: f32,
) {
    for (((p, m), v), &g) in params.iter_mut().zip(m).zip(v).zip(grad) {
        let g = g + weight_decay * *p;
        *m = ADAM_BETA_1 * *m + (1.0 - ADAM_BETA_1) * g;
        *v = ADAM_BETA_2 * *v + (1.0 - ADAM_BETA_2) * g * g;
        *p -= step_size * *m / (v.sqrt() * inv_bias_c2_sqrt + ADAM_EPS);
    }
}

/// Train on `x [n_rows x n_features]` with binary labels `y`, starting from `init` if given
/// (warm start) and from torch-style random weights otherwise. Returns the parameters.
pub fn train(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    y: &[f32],
    config: &TrainConfig,
    init: Option<Vec<f32>>,
) -> Result<Vec<f32>, String> {
    let layout = Layout::new(n_features, &config.hidden_layers, config.input_batch_norm)?;
    if x.len() != n_rows * n_features || y.len() != n_rows {
        return Err(format!(
            "x has {} values and y {} for {n_rows} rows x {n_features} features",
            x.len(),
            y.len()
        ));
    }
    if n_rows == 0 || config.batch_size == 0 || config.epochs == 0 {
        return Err(
            "training needs rows, a positive batch size and at least one epoch".to_string(),
        );
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let params = match init {
        Some(params) if params.len() == layout.size => params,
        Some(params) => {
            return Err(format!(
                "initial parameters have {} values, the network needs {}",
                params.len(),
                layout.size
            ))
        }
        None => layout.init_params(&mut rng),
    };
    let n_net = layout.n_net;
    let mut net = Network::new(&layout.dims, &params);
    // scale, shift, running mean, running variance; empty without the batch norm
    let mut bn = params[n_net..].to_vec();

    // rows in random order once, then contiguous batches in a new random order every epoch
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.shuffle(&mut rng);
    let mut x_shuffled = vec![0.0f32; x.len()];
    let mut y_shuffled = vec![0.0f32; n_rows];
    x_shuffled
        .par_chunks_mut(GATHER_ROWS * n_features)
        .zip(y_shuffled.par_chunks_mut(GATHER_ROWS))
        .zip(order.par_chunks(GATHER_ROWS))
        .for_each(|((x_rows, y_rows), rows)| {
            for ((x_row, y_row), &r) in x_rows.chunks_exact_mut(n_features).zip(y_rows).zip(rows) {
                x_row.copy_from_slice(&x[r * n_features..(r + 1) * n_features]);
                *y_row = y[r];
            }
        });

    let batch_size = config.batch_size.min(n_rows);
    let n_batches = (n_rows / batch_size).max(1);
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_rows = batch_size.div_ceil(n_chunks).div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let n_chunks = batch_size.div_ceil(chunk_rows);
    let input_grad_len = if layout.batch_norm {
        chunk_rows * n_features
    } else {
        0
    };
    let mut states: Vec<ChunkState> = (0..n_chunks)
        .map(|_| ChunkState {
            ws: TrainWorkspace::new(&net),
            grad: mlp::AlignedBuf::zeros(layout.size),
            input_grad: vec![0.0; input_grad_len],
        })
        .collect();
    let mut adam_m = vec![0.0f32; layout.size];
    let mut adam_v = vec![0.0f32; layout.size];
    let mut grad = vec![0.0f32; layout.size];
    let loss_scale = 1.0 / batch_size as f32;
    let mut batch_starts: Vec<usize> = (0..n_batches).map(|b| b * batch_size).collect();
    let mut step = 0i32;
    let total_steps = (n_batches * config.epochs) as f64;
    let mut averaged: Option<Vec<f32>> = None;
    let mut n_averaged = 0usize;
    // batch-normalized input of the current batch, and its standardized values before scale and shift
    let (mut x_norm, mut x_hat) = if layout.batch_norm {
        (
            vec![0.0f32; batch_size * n_features],
            vec![0.0f32; batch_size * n_features],
        )
    } else {
        (Vec::new(), Vec::new())
    };

    for epoch in 0..config.epochs {
        batch_starts.shuffle(&mut rng);
        for &batch_start in &batch_starts {
            let batch = batch_start * n_features..(batch_start + batch_size) * n_features;
            if layout.batch_norm {
                normalize_batch(&mut bn, &x_shuffled[batch.clone()], &mut x_norm, &mut x_hat);
            }
            let (xs, ys) = (&x_shuffled, &y_shuffled);
            let (x_norm_ref, x_hat_ref) = (&x_norm, &x_hat);
            states.par_iter_mut().enumerate().for_each(|(c, state)| {
                let r0 = batch_start + c * chunk_rows;
                let rows = chunk_rows.min(batch_start + batch_size - r0);
                // rows of this chunk within the batch
                let (b0, b1) = (
                    (r0 - batch_start) * n_features,
                    (r0 - batch_start + rows) * n_features,
                );
                let (input, x_hat_chunk) = if layout.batch_norm {
                    (&x_norm_ref[b0..b1], Some(&x_hat_ref[b0..b1]))
                } else {
                    (&xs[r0 * n_features..(r0 + rows) * n_features], None)
                };
                let y = &ys[r0..r0 + rows];
                train_chunk(
                    config.backend,
                    &net,
                    state,
                    input,
                    y,
                    loss_scale,
                    x_hat_chunk,
                );
            });

            grad.copy_from_slice(&states[0].grad);
            for state in &states[1..] {
                for (g, w) in grad.iter_mut().zip(state.grad.iter()) {
                    *g += w;
                }
            }

            let learning_rate = if config.cosine_schedule {
                let progress = f64::from(step) / total_steps;
                (f64::from(config.learning_rate)
                    * 0.5
                    * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
            } else {
                config.learning_rate
            };
            step += 1;
            let step_size = learning_rate / (1.0 - ADAM_BETA_1.powi(step));
            let inv_bias_c2_sqrt = 1.0 / (1.0 - ADAM_BETA_2.powi(step)).sqrt();
            let (m_net, m_bn) = adam_m.split_at_mut(n_net);
            let (v_net, v_bn) = adam_v.split_at_mut(n_net);
            let (grad_net, grad_bn) = grad.split_at(n_net);
            let decay = config.weight_decay;
            net.update(|params| {
                adam_step(
                    params,
                    grad_net,
                    (m_net, v_net),
                    step_size,
                    inv_bias_c2_sqrt,
                    decay,
                )
            });
            // the running statistics of the batch norm are buffers, not trained parameters
            let trained = bn.len() / 2;
            adam_step(
                &mut bn[..trained],
                grad_bn,
                (m_bn, v_bn),
                step_size,
                inv_bias_c2_sqrt,
                decay,
            );

            if config.average_last_epoch && epoch + 1 == config.epochs {
                n_averaged += 1;
                let current = net.params().iter().chain(&bn);
                match averaged.as_mut() {
                    None => averaged = Some(current.copied().collect()),
                    Some(mean) => {
                        let weight = 1.0 / n_averaged as f32;
                        for (a, &p) in mean.iter_mut().zip(current) {
                            *a += (p - *a) * weight;
                        }
                    }
                }
            }
        }
    }
    Ok(averaged.unwrap_or_else(|| net.params().iter().chain(&bn).copied().collect()))
}

/// Output logit of every row of `x`.
pub fn predict(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    hidden_layers: &[usize],
    params: &[f32],
    backend: Backend,
    input_batch_norm: bool,
) -> Result<Vec<f32>, String> {
    let layout = Layout::new(n_features, hidden_layers, input_batch_norm)?;
    if params.len() != layout.size {
        return Err(format!(
            "parameters have {} values, a network with {n_features} inputs and layers {hidden_layers:?} needs {}",
            params.len(),
            layout.size
        ));
    }
    if x.len() != n_rows * n_features {
        return Err(format!(
            "x has {} values, expected {n_rows} x {n_features}",
            x.len()
        ));
    }
    let net = if layout.batch_norm {
        Network::new(&layout.dims, &fold_input_batch_norm(&layout, params))
    } else {
        Network::new(&layout.dims, params)
    };
    let mut logits = vec![0.0f32; n_rows];
    match backend {
        Backend::Simd => net.logits(crate::simd::get_backend(), x, &mut logits),
        #[cfg(target_os = "macos")]
        Backend::Accelerate => net.logits(&super::ops::Accelerate, x, &mut logits),
    }
    Ok(logits)
}
