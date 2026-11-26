//! The FDR classifier network: fully connected ReLU layers and one output logit, trained with
//! BCE-with-logits and Adam (torch semantics: L2 weight decay added to the gradient).
//!
//! Parameters live in one flat vector, layer by layer: hidden weights transposed (`[in x out]`)
//! followed by their bias, then the output weights `[last hidden]` and the output bias.
//! Each batch is split into units of rows. The units run forward and backward in parallel, and
//! their gradients are summed in a fixed order.
//!
//! With `input_batch_norm`, the input goes through a torch `BatchNorm1d` first (`BN_MOMENTUM`,
//! `BN_EPS`). Training normalizes each minibatch with its own mean and biased variance, and learns
//! a scale and shift per feature. Prediction uses the running mean and unbiased variance, folded
//! into the first layer. The batch norm parameters follow the output bias: scale, shift, running
//! mean, running variance (`n_features` each).

use crate::mlp::{AlignedBuf, Network, TrainWorkspace, TRAIN_ROW_BLOCK};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Mutex;

const ADAM_BETA_1: f32 = 0.9;
const ADAM_BETA_2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
/// Rows per task of the parallel row shuffle.
const GATHER_ROWS: usize = 8192;
const BN_MOMENTUM: f32 = 0.1;
const BN_EPS: f32 = 1e-5;
/// Rows of a unit of parallel gradient work. Small units balance the load on cores of different
/// speed. One unit is one row block of the network kernels.
const UNIT_ROWS: usize = TRAIN_ROW_BLOCK;
/// Parameters per task of the parallel gradient sum.
const REDUCE_PARAMS: usize = 1024;

pub struct TrainConfig {
    pub hidden_layers: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    /// Decay the learning rate along a half cosine from `learning_rate` to zero over all
    /// minibatches of the fit (as torch `CosineAnnealingLR(T_max = steps, eta_min = 0)`, stepped
    /// once per minibatch).
    pub cosine_schedule: bool,
    /// Return the mean of the parameters over the minibatches of the last epoch, not the last
    /// parameters.
    pub average_last_epoch: bool,
    /// Normalize the input with a batch norm layer (see the module docs).
    pub input_batch_norm: bool,
}

/// The flat parameter vector: the network's parameters ([`Network`]), then those of the input
/// batch norm, if any: scale, shift, running mean, running variance (`n_features` each).
pub struct Layout {
    dims: Vec<usize>, // input, hidden..., 1
    /// The length of the network parameters, which is also the start of the batch norm
    /// parameters.
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

/// Adam with torch semantics: L2 weight decay added to the gradient.
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    steps: i32,
    weight_decay: f32,
}

impl Adam {
    fn new(n_params: usize, weight_decay: f32) -> Self {
        Adam {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            steps: 0,
            weight_decay,
        }
    }

    /// One step with `learning_rate` on parameter groups that lie one after the other in the
    /// moment buffers, each with its gradient.
    fn step(&mut self, learning_rate: f32, groups: &mut [(&mut [f32], &[f32])]) {
        self.steps += 1;
        let step_size = learning_rate / (1.0 - ADAM_BETA_1.powi(self.steps));
        let inv_bias_c2_sqrt = 1.0 / (1.0 - ADAM_BETA_2.powi(self.steps)).sqrt();
        let mut offset = 0;
        for (params, grad) in groups.iter_mut() {
            let moments = offset..offset + params.len();
            let (m, v) = (&mut self.m[moments.clone()], &mut self.v[moments]);
            for (((p, m), v), &g) in params.iter_mut().zip(m).zip(v).zip(*grad) {
                let g = g + self.weight_decay * *p;
                *m = ADAM_BETA_1 * *m + (1.0 - ADAM_BETA_1) * g;
                *v = ADAM_BETA_2 * *v + (1.0 - ADAM_BETA_2) * g * g;
                *p -= step_size * *m / (v.sqrt() * inv_bias_c2_sqrt + ADAM_EPS);
            }
            offset += params.len();
        }
    }
}

/// The rows of `x [n_rows x n_features]` and `y` in a random order.
fn shuffle_rows(x: &[f32], y: &[f32], n_features: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.shuffle(rng);
    let mut x_shuffled = vec![0.0f32; x.len()];
    let mut y_shuffled = vec![0.0f32; y.len()];
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
    (x_shuffled, y_shuffled)
}

/// The gradient buffers of one row unit, allocated once per fit.
struct UnitGrad {
    /// The gradients of the network parameters, then of the batch norm scale and shift.
    grad: AlignedBuf,
    /// The gradient at the batch norm output (only with the input batch norm).
    input_grad: Vec<f32>,
}

/// The gradients of one row unit into `unit.grad`. `x_hat` is the standardized input of the
/// batch norm, if any.
fn unit_gradient(
    net: &Network,
    ws: &mut TrainWorkspace,
    unit: &mut UnitGrad,
    x: &[f32],
    x_hat: Option<&[f32]>,
    y: &[f32],
    loss_scale: f32,
) {
    let UnitGrad { grad, input_grad } = unit;
    grad.fill(0.0);
    let (net_grad, bn_grad) = grad.split_at_mut(Network::param_count(net.dims()));
    let input_grad = &mut input_grad[..x_hat.map_or(0, |_| x.len())];
    let wanted = x_hat.is_some().then_some(&mut *input_grad);
    net.train_chunk(
        crate::simd::get_backend(),
        x,
        y,
        loss_scale,
        ws,
        net_grad,
        wanted,
    );
    if let Some(x_hat) = x_hat {
        // The input has no parameters. Thus the gradient at the batch norm output gives the exact
        // gradients of scale and shift.
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

/// A batch split into units of [`UNIT_ROWS`] rows. Rayon hands the units to the threads as they
/// become free, so faster cores take more units (e.g. performance and efficiency cores). Every
/// unit has its own gradient buffer, and the buffers are summed in a fixed order. Thus the
/// gradient does not depend on the thread count or the scheduling.
struct Chunks {
    units: Vec<UnitGrad>,
    /// The scratch buffers of each rayon thread, by thread index. A thread only locks its own,
    /// so the locks never wait. `Network::train_chunk` must not use rayon: a thread that steals a
    /// unit while it holds its lock would wait on itself.
    workspaces: Vec<Mutex<TrainWorkspace>>,
}

impl Chunks {
    fn new(net: &Network, n_params: usize, batch_size: usize, input_grad: bool) -> Self {
        let input_grad_len = if input_grad {
            UNIT_ROWS * net.dims()[0]
        } else {
            0
        };
        let units = (0..batch_size.div_ceil(UNIT_ROWS))
            .map(|_| UnitGrad {
                grad: AlignedBuf::zeros(n_params),
                input_grad: vec![0.0; input_grad_len],
            })
            .collect();
        let workspaces = (0..rayon::current_num_threads().max(1))
            .map(|_| Mutex::new(TrainWorkspace::new(net)))
            .collect();
        Chunks { units, workspaces }
    }

    /// The gradient of the mean loss over the batch `x`, `y` into `grad`. With the input batch
    /// norm, `x` is its output and `x_hat` its standardized input, and `grad` also gets the
    /// gradients of its scale and shift.
    fn gradient(
        &mut self,
        net: &Network,
        x: &[f32],
        x_hat: Option<&[f32]>,
        y: &[f32],
        grad: &mut [f32],
    ) {
        let (inputs, loss_scale) = (net.dims()[0], 1.0 / y.len() as f32);
        let unit_len = UNIT_ROWS * inputs;
        let workspaces = &self.workspaces;
        let units = &mut self.units[..y.len().div_ceil(UNIT_ROWS)];
        units
            .par_iter_mut()
            .with_max_len(1)
            .zip(x.par_chunks(unit_len))
            .zip(y.par_chunks(UNIT_ROWS))
            .enumerate()
            .for_each(|(u, ((unit, x), y))| {
                let x_hat = x_hat.map(|x_hat| &x_hat[u * unit_len..u * unit_len + x.len()]);
                let thread = rayon::current_thread_index().unwrap_or(0) % workspaces.len();
                let mut ws = workspaces[thread].lock().unwrap();
                unit_gradient(net, &mut ws, unit, x, x_hat, y, loss_scale);
            });
        let units = &*units;
        grad.par_chunks_mut(REDUCE_PARAMS)
            .enumerate()
            .for_each(|(c, grad)| {
                let range = c * REDUCE_PARAMS..c * REDUCE_PARAMS + grad.len();
                grad.copy_from_slice(&units[0].grad[range.clone()]);
                for unit in &units[1..] {
                    for (g, u) in grad.iter_mut().zip(&unit.grad[range.clone()]) {
                        *g += u;
                    }
                }
            });
    }
}

/// Batch norm of one training batch (torch `BatchNorm1d` in training mode). It standardizes every
/// feature of `x [rows x k]` with the batch mean and biased variance into `x_hat`. It applies
/// scale and shift into `x_norm`. It moves the running mean and (unbiased) variance towards the
/// batch statistics. `bn` holds scale, shift, running mean and running variance.
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
        sq[p] = 1.0 / (var + f64::from(BN_EPS)).sqrt(); // now the inverse standard deviation
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

/// The input batch norm during training: its parameters (scale, shift, running mean, running
/// variance) and the normalized input of the current batch.
struct InputBatchNorm {
    params: Vec<f32>,
    /// The output of the current batch, after scale and shift.
    x_norm: Vec<f32>,
    /// The current batch, standardized with its own statistics.
    x_hat: Vec<f32>,
}

impl InputBatchNorm {
    fn new(params: Vec<f32>, batch_len: usize) -> Self {
        InputBatchNorm {
            params,
            x_norm: vec![0.0; batch_len],
            x_hat: vec![0.0; batch_len],
        }
    }

    /// Normalizes `x`, moves the running statistics, and returns the output and the
    /// standardized input.
    fn normalize(&mut self, x: &[f32]) -> (&[f32], &[f32]) {
        normalize_batch(&mut self.params, x, &mut self.x_norm, &mut self.x_hat);
        (&self.x_norm, &self.x_hat)
    }

    /// Scale and shift, the trained parameters (the running statistics are buffers).
    fn trained(&mut self) -> &mut [f32] {
        let k = self.params.len() / 4;
        &mut self.params[..2 * k]
    }
}

/// The learning rate of minibatch `step` (counting from 0) of `steps`. It is constant, or with
/// `cosine_schedule` it follows a half cosine from `learning_rate` to zero (torch
/// `CosineAnnealingLR(T_max = steps, eta_min = 0)`, stepped once per minibatch).
fn learning_rate(config: &TrainConfig, step: usize, steps: usize) -> f32 {
    if !config.cosine_schedule {
        return config.learning_rate;
    }
    let progress = step as f64 / steps as f64;
    (f64::from(config.learning_rate) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
}

/// The running mean of parameter vectors.
#[derive(Default)]
struct ParameterAverage {
    mean: Option<Vec<f32>>,
    count: usize,
}

impl ParameterAverage {
    fn add<'a>(&mut self, params: impl Iterator<Item = &'a f32>) {
        self.count += 1;
        match self.mean.as_mut() {
            None => self.mean = Some(params.copied().collect()),
            Some(mean) => {
                let weight = 1.0 / self.count as f32;
                for (a, &p) in mean.iter_mut().zip(params) {
                    *a += (p - *a) * weight;
                }
            }
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
    let mut net = Network::new(&layout.dims, &params);

    // The rows are shuffled once. Every epoch then takes the contiguous batches in a new random
    // order.
    let (x, y) = shuffle_rows(x, y, n_features, &mut rng);
    let batch_size = config.batch_size.min(n_rows);
    let mut batch_starts: Vec<usize> = (0..(n_rows / batch_size).max(1))
        .map(|b| b * batch_size)
        .collect();
    let mut batch_norm = layout
        .batch_norm
        .then(|| InputBatchNorm::new(params[layout.n_net..].to_vec(), batch_size * n_features));
    let mut chunks = Chunks::new(&net, layout.size, batch_size, layout.batch_norm);
    let mut adam = Adam::new(layout.size, config.weight_decay);
    let mut grad = vec![0.0f32; layout.size];
    let mut average = ParameterAverage::default();
    let steps = batch_starts.len() * config.epochs;
    let mut step = 0;
    for epoch in 0..config.epochs {
        batch_starts.shuffle(&mut rng);
        for &start in &batch_starts {
            let rows = start..start + batch_size;
            let x_batch = &x[rows.start * n_features..rows.end * n_features];
            match batch_norm.as_mut() {
                Some(bn) => {
                    let (x_norm, x_hat) = bn.normalize(x_batch);
                    chunks.gradient(&net, x_norm, Some(x_hat), &y[rows], &mut grad);
                }
                None => chunks.gradient(&net, x_batch, None, &y[rows], &mut grad),
            }
            let learning_rate = learning_rate(config, step, steps);
            step += 1;
            let (grad_net, grad_bn) = grad.split_at(layout.n_net);
            net.update(|params| match batch_norm.as_mut() {
                Some(bn) => adam.step(
                    learning_rate,
                    &mut [(params, grad_net), (bn.trained(), grad_bn)],
                ),
                None => adam.step(learning_rate, &mut [(params, grad_net)]),
            });
            if config.average_last_epoch && epoch + 1 == config.epochs {
                average.add(net.params().iter().chain(bn_params(&batch_norm)));
            }
        }
    }
    let last = net.params().iter().chain(bn_params(&batch_norm)).copied();
    Ok(average.mean.unwrap_or_else(|| last.collect()))
}

/// The parameters of the input batch norm, if any.
fn bn_params(batch_norm: &Option<InputBatchNorm>) -> &[f32] {
    batch_norm.as_ref().map_or(&[], |bn| &bn.params)
}

/// The output logit of every row of `x`.
pub fn predict(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    hidden_layers: &[usize],
    params: &[f32],
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
    net.logits(crate::simd::get_backend(), x, &mut logits);
    Ok(logits)
}
