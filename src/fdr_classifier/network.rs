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
//! shift per feature; prediction uses the running mean and unbiased variance. Its parameters follow
//! the output bias: scale, shift, running mean, running variance (`n_features` each).

use super::ops::{self, Backend};
use crate::mlp;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const ADAM_BETA_1: f32 = 0.9;
const ADAM_BETA_2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
/// Row chunks are multiples of the 4-row SIMD kernel tile.
const ROW_ALIGN: usize = 4;
const PREDICT_CHUNK_ROWS: usize = 4096;
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

pub struct Layout {
    dims: Vec<usize>, // input, hidden..., 1
    weights: Vec<usize>,
    biases: Vec<usize>,
    /// offset of the input batch norm parameters (scale, shift, running mean, running variance)
    input_bn: Option<usize>,
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
        let (mut weights, mut biases, mut offset) = (Vec::new(), Vec::new(), 0);
        for l in 0..dims.len() - 1 {
            weights.push(offset);
            offset += dims[l] * dims[l + 1];
            biases.push(offset);
            offset += dims[l + 1];
        }
        let input_bn = input_batch_norm.then_some(offset);
        if input_batch_norm {
            offset += 4 * n_features;
        }
        Ok(Layout {
            dims,
            weights,
            biases,
            input_bn,
            size: offset,
        })
    }

    fn n_hidden(&self) -> usize {
        self.dims.len() - 2
    }

    fn wt<'a>(&self, params: &'a [f32], l: usize) -> &'a [f32] {
        &params[self.weights[l]..self.weights[l] + self.dims[l] * self.dims[l + 1]]
    }

    fn bias<'a>(&self, params: &'a [f32], l: usize) -> &'a [f32] {
        &params[self.biases[l]..self.biases[l] + self.dims[l + 1]]
    }

    /// Offset of input batch norm block `i`: 0 scale, 1 shift, 2 running mean, 3 running variance.
    fn bn(&self, i: usize) -> Option<usize> {
        self.input_bn.map(|o| o + i * self.dims[0])
    }

    /// torch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases.
    fn init_params(&self, rng: &mut StdRng) -> Vec<f32> {
        let mut params = vec![0.0f32; self.size];
        for l in 0..self.dims.len() - 1 {
            let bound = 1.0 / (self.dims[l] as f32).sqrt();
            for p in &mut params[self.weights[l]..self.biases[l] + self.dims[l + 1]] {
                *p = rng.random_range(-bound..bound);
            }
        }
        if let (Some(scale), Some(var)) = (self.bn(0), self.bn(3)) {
            let k = self.dims[0];
            params[scale..scale + k].fill(1.0);
            params[var..var + k].fill(1.0);
        }
        params
    }
}

struct Workspace {
    activations: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
    grad: Vec<f32>,
}

impl Workspace {
    fn new(layout: &Layout, rows: usize) -> Self {
        let hidden = &layout.dims[1..layout.dims.len() - 1];
        Workspace {
            activations: hidden.iter().map(|&d| vec![0.0; rows * d]).collect(),
            deltas: hidden.iter().map(|&d| vec![0.0; rows * d]).collect(),
            grad: vec![0.0; layout.size],
        }
    }
}

fn forward_hidden(
    backend: Backend,
    params: &[f32],
    layout: &Layout,
    ws: &mut Workspace,
    x: &[f32],
    rows: usize,
) {
    for l in 0..layout.n_hidden() {
        let (k, n) = (layout.dims[l], layout.dims[l + 1]);
        let (done, rest) = ws.activations.split_at_mut(l);
        let input = if l == 0 { x } else { &done[l - 1][..rows * k] };
        ops::dense_forward_relu(
            backend,
            input,
            rows,
            k,
            layout.wt(params, l),
            n,
            layout.bias(params, l),
            &mut rest[0][..rows * n],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn train_chunk(
    backend: Backend,
    params: &[f32],
    weights_untransposed: &[Vec<f32>],
    layout: &Layout,
    ws: &mut Workspace,
    x: &[f32],
    y: &[f32],
    rows: usize,
    loss_scale: f32,
    x_hat: Option<&[f32]>,
) {
    let n_hidden = layout.n_hidden();
    forward_hidden(backend, params, layout, ws, x, rows);
    ws.grad.iter_mut().for_each(|g| *g = 0.0);

    let k = layout.dims[n_hidden];
    {
        let (grad_w, grad_rest) = ws.grad.split_at_mut(layout.biases[n_hidden]);
        mlp::output_layer_train(
            &ws.activations[n_hidden - 1][..rows * k],
            rows,
            k,
            layout.wt(params, n_hidden),
            layout.bias(params, n_hidden)[0],
            y,
            loss_scale,
            &mut ws.deltas[n_hidden - 1][..rows * k],
            &mut grad_w[layout.weights[n_hidden]..],
            &mut grad_rest[0],
        );
    }

    for l in (0..n_hidden).rev() {
        let (k, n) = (layout.dims[l], layout.dims[l + 1]);
        let input = if l == 0 {
            x
        } else {
            &ws.activations[l - 1][..rows * k]
        };
        let (grad_w, grad_rest) = ws.grad.split_at_mut(layout.biases[l]);
        ops::dense_weight_grad(
            backend,
            input,
            rows,
            k,
            &ws.deltas[l][..rows * n],
            n,
            &mut grad_w[layout.weights[l]..],
            &mut grad_rest[..n],
        );
        if l == 0 {
            if let (Some(x_hat), Some(scale), Some(shift)) = (x_hat, layout.bn(0), layout.bn(1)) {
                // the input has no parameters, so scale and shift get exact gradients from the
                // gradient at the batch norm output: d_in = delta_0 . W_0^T
                let wt0 = layout.wt(params, 0);
                let deltas = &ws.deltas[0][..rows * n];
                for p in 0..k {
                    let w_row = &wt0[p * n..(p + 1) * n];
                    let (mut g_scale, mut g_shift) = (0.0f32, 0.0f32);
                    for r in 0..rows {
                        let d_in: f32 = deltas[r * n..(r + 1) * n]
                            .iter()
                            .zip(w_row)
                            .map(|(d, w)| d * w)
                            .sum();
                        g_scale += d_in * x_hat[r * k + p];
                        g_shift += d_in;
                    }
                    ws.grad[scale + p] += g_scale;
                    ws.grad[shift + p] += g_shift;
                }
            }
        } else {
            let (lower, upper) = ws.deltas.split_at_mut(l);
            ops::dense_backward_input(
                backend,
                &upper[0][..rows * n],
                rows,
                n,
                &weights_untransposed[l],
                layout.wt(params, l),
                k,
                &ws.activations[l - 1][..rows * k],
                &mut lower[l - 1][..rows * k],
            );
        }
    }
}

/// Batch norm of one training batch (torch `BatchNorm1d` in training mode): standardize every
/// feature with the batch mean and biased variance into `x_hat`, apply scale and shift into
/// `x_norm`, and move the running mean and (unbiased) variance towards the batch statistics.
fn normalize_batch(
    layout: &Layout,
    params: &mut [f32],
    x: &[f32],
    rows: usize,
    x_norm: &mut [f32],
    x_hat: &mut [f32],
) {
    let k = layout.dims[0];
    let (scale, shift, mean_at, var_at) = (
        layout.bn(0).unwrap(),
        layout.bn(1).unwrap(),
        layout.bn(2).unwrap(),
        layout.bn(3).unwrap(),
    );
    let mut mean = vec![0.0f64; k];
    let mut sq = vec![0.0f64; k];
    for row in x.chunks_exact(k).take(rows) {
        for p in 0..k {
            let v = f64::from(row[p]);
            mean[p] += v;
            sq[p] += v * v;
        }
    }
    let n = rows as f64;
    for p in 0..k {
        mean[p] /= n;
        let var = (sq[p] / n - mean[p] * mean[p]).max(0.0);
        let unbiased = if rows > 1 { var * n / (n - 1.0) } else { var };
        params[mean_at + p] =
            (1.0 - BN_MOMENTUM) * params[mean_at + p] + BN_MOMENTUM * mean[p] as f32;
        params[var_at + p] =
            (1.0 - BN_MOMENTUM) * params[var_at + p] + BN_MOMENTUM * unbiased as f32;
        sq[p] = 1.0 / (var + f64::from(BN_EPS)).sqrt(); // reuse as inverse standard deviation
    }
    for r in 0..rows {
        for p in 0..k {
            let h = ((f64::from(x[r * k + p]) - mean[p]) * sq[p]) as f32;
            x_hat[r * k + p] = h;
            x_norm[r * k + p] = params[scale + p] * h + params[shift + p];
        }
    }
}

/// Batch norm at prediction (torch evaluation mode): running mean and variance, scale and shift.
fn normalize_rows(layout: &Layout, params: &[f32], x: &[f32], out: &mut [f32]) {
    let k = layout.dims[0];
    let (scale, shift, mean_at, var_at) = (
        layout.bn(0).unwrap(),
        layout.bn(1).unwrap(),
        layout.bn(2).unwrap(),
        layout.bn(3).unwrap(),
    );
    for (row, out_row) in x.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
        for p in 0..k {
            let inv_std = 1.0 / (params[var_at + p] + BN_EPS).sqrt();
            out_row[p] =
                params[scale + p] * (row[p] - params[mean_at + p]) * inv_std + params[shift + p];
        }
    }
}

fn transpose(wt: &[f32], k: usize, n: usize, w: &mut [f32]) {
    for p in 0..k {
        for j in 0..n {
            w[j * k + p] = wt[p * n + j];
        }
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
    let mut params = match init {
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

    // rows in random order once, then contiguous batches in a new random order every epoch
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.shuffle(&mut rng);
    let mut x_shuffled = vec![0.0f32; x.len()];
    let mut y_shuffled = vec![0.0f32; n_rows];
    for (i, &r) in order.iter().enumerate() {
        x_shuffled[i * n_features..(i + 1) * n_features]
            .copy_from_slice(&x[r * n_features..(r + 1) * n_features]);
        y_shuffled[i] = y[r];
    }

    let batch_size = config.batch_size.min(n_rows);
    let n_batches = (n_rows / batch_size).max(1);
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_rows = batch_size.div_ceil(n_chunks).div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let n_chunks = batch_size.div_ceil(chunk_rows);
    let mut workspaces: Vec<Workspace> = (0..n_chunks)
        .map(|_| Workspace::new(&layout, chunk_rows))
        .collect();
    let n_hidden = layout.n_hidden();
    // the input-gradient kernel reads the weights as [out x in]
    let mut weights_untransposed: Vec<Vec<f32>> = (0..n_hidden)
        .map(|l| vec![0.0; layout.dims[l] * layout.dims[l + 1]])
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
    let (mut x_norm, mut x_hat) = if layout.input_bn.is_some() {
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
            if layout.input_bn.is_some() {
                normalize_batch(
                    &layout,
                    &mut params,
                    &x_shuffled[batch_start * n_features..(batch_start + batch_size) * n_features],
                    batch_size,
                    &mut x_norm,
                    &mut x_hat,
                );
            }
            for l in 1..n_hidden {
                transpose(
                    layout.wt(&params, l),
                    layout.dims[l],
                    layout.dims[l + 1],
                    &mut weights_untransposed[l],
                );
            }
            let (params_ref, w_ref, layout_ref) = (&params, &weights_untransposed, &layout);
            let (xs, ys) = (&x_shuffled, &y_shuffled);
            let (x_norm_ref, x_hat_ref) = (&x_norm, &x_hat);
            let batch_norm = layout.input_bn.is_some();
            workspaces.par_iter_mut().enumerate().for_each(|(c, ws)| {
                let r0 = batch_start + c * chunk_rows;
                let rows = chunk_rows.min(batch_start + batch_size - r0);
                // rows of this chunk within the batch
                let (b0, b1) = (
                    (r0 - batch_start) * n_features,
                    (r0 - batch_start + rows) * n_features,
                );
                let (input, x_hat_chunk) = if batch_norm {
                    (&x_norm_ref[b0..b1], Some(&x_hat_ref[b0..b1]))
                } else {
                    (&xs[r0 * n_features..(r0 + rows) * n_features], None)
                };
                train_chunk(
                    config.backend,
                    params_ref,
                    w_ref,
                    layout_ref,
                    ws,
                    input,
                    &ys[r0..r0 + rows],
                    rows,
                    loss_scale,
                    x_hat_chunk,
                );
            });

            grad.copy_from_slice(&workspaces[0].grad);
            for ws in &workspaces[1..] {
                for (g, w) in grad.iter_mut().zip(&ws.grad) {
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
            // the running statistics of the batch norm are buffers, not trained parameters
            let trained = layout.bn(2).unwrap_or(layout.size);
            for (((p, m), v), &g) in params[..trained]
                .iter_mut()
                .zip(&mut adam_m)
                .zip(&mut adam_v)
                .zip(&grad)
            {
                let g = g + config.weight_decay * *p;
                *m = ADAM_BETA_1 * *m + (1.0 - ADAM_BETA_1) * g;
                *v = ADAM_BETA_2 * *v + (1.0 - ADAM_BETA_2) * g * g;
                *p -= step_size * *m / (v.sqrt() * inv_bias_c2_sqrt + ADAM_EPS);
            }

            if config.average_last_epoch && epoch + 1 == config.epochs {
                n_averaged += 1;
                match averaged.as_mut() {
                    None => averaged = Some(params.clone()),
                    Some(mean) => {
                        let weight = 1.0 / n_averaged as f32;
                        for (a, &p) in mean.iter_mut().zip(&params) {
                            *a += (p - *a) * weight;
                        }
                    }
                }
            }
        }
    }
    Ok(averaged.unwrap_or(params))
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
    let n_hidden = layout.n_hidden();
    let mut logits = vec![0.0f32; n_rows];
    logits
        .par_chunks_mut(PREDICT_CHUNK_ROWS)
        .enumerate()
        .for_each_init(
            || {
                let normalized = vec![
                    0.0f32;
                    layout
                        .input_bn
                        .map_or(0, |_| PREDICT_CHUNK_ROWS * n_features)
                ];
                (Workspace::new(&layout, PREDICT_CHUNK_ROWS), normalized)
            },
            |(ws, normalized), (c, out)| {
                let rows = out.len();
                let r0 = c * PREDICT_CHUNK_ROWS;
                let mut input = &x[r0 * n_features..(r0 + rows) * n_features];
                if layout.input_bn.is_some() {
                    normalize_rows(&layout, params, input, &mut normalized[..rows * n_features]);
                    input = &normalized[..rows * n_features];
                }
                forward_hidden(backend, params, &layout, ws, input, rows);
                let k = layout.dims[n_hidden];
                mlp::output_layer_logits(
                    &ws.activations[n_hidden - 1][..rows * k],
                    rows,
                    k,
                    layout.wt(params, n_hidden),
                    layout.bias(params, n_hidden)[0],
                    out,
                );
            },
        );
    Ok(logits)
}
