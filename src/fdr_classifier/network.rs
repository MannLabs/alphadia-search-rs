//! The FDR classifier network: fully connected ReLU layers and one output logit, trained with
//! BCE-with-logits and Adam (torch semantics: L2 weight decay added to the gradient).
//!
//! Parameters live in one flat vector, layer by layer: hidden weights transposed (`[in x out]`)
//! followed by their bias, then the output weights `[last hidden]` and the output bias.
//! Each batch has one row chunk per rayon thread. The chunks run forward and backward
//! independently, and their gradients are summed.

use crate::mlp::{AlignedBuf, Network, TrainWorkspace};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const ADAM_BETA_1: f32 = 0.9;
const ADAM_BETA_2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
/// Rows per task of the parallel row shuffle.
const GATHER_ROWS: usize = 8192;

pub struct TrainConfig {
    pub hidden_layers: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

/// The flat parameter vector of the network ([`Network`]).
pub struct Layout {
    dims: Vec<usize>, // input, hidden..., 1
    pub size: usize,
}

impl Layout {
    pub fn new(n_features: usize, hidden_layers: &[usize]) -> Result<Self, String> {
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
        let size = Network::param_count(&dims);
        Ok(Layout { dims, size })
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

/// Buffers of one training thread, allocated once per fit.
struct ChunkState {
    ws: TrainWorkspace,
    /// The gradients of the network parameters.
    grad: AlignedBuf,
}

/// A batch split into one row chunk per rayon thread. The chunks run forward and backward in
/// parallel.
struct Chunks {
    rows: usize,
    states: Vec<ChunkState>,
}

impl Chunks {
    fn new(net: &Network, n_params: usize, batch_size: usize) -> Self {
        let rows = batch_size.div_ceil(rayon::current_num_threads().max(1));
        let states = (0..batch_size.div_ceil(rows))
            .map(|_| ChunkState {
                ws: TrainWorkspace::new(net),
                grad: AlignedBuf::zeros(n_params),
            })
            .collect();
        Chunks { rows, states }
    }

    /// The gradient of the mean loss over the batch `x`, `y` into `grad`.
    fn gradient(&mut self, net: &Network, x: &[f32], y: &[f32], grad: &mut [f32]) {
        let (inputs, loss_scale) = (net.dims()[0], 1.0 / y.len() as f32);
        self.states
            .par_iter_mut()
            .zip(x.par_chunks(self.rows * inputs))
            .zip(y.par_chunks(self.rows))
            .for_each(|((state, x), y)| {
                let ChunkState { ws, grad } = state;
                grad.fill(0.0);
                net.train_chunk(crate::simd::get_backend(), x, y, loss_scale, ws, grad, None);
            });
        grad.copy_from_slice(&self.states[0].grad);
        for state in &self.states[1..] {
            for (g, c) in grad.iter_mut().zip(state.grad.iter()) {
                *g += c;
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
    let layout = Layout::new(n_features, &config.hidden_layers)?;
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
    let mut chunks = Chunks::new(&net, layout.size, batch_size);
    let mut adam = Adam::new(layout.size, config.weight_decay);
    let mut grad = vec![0.0f32; layout.size];
    for _ in 0..config.epochs {
        batch_starts.shuffle(&mut rng);
        for &start in &batch_starts {
            let rows = start..start + batch_size;
            let x_batch = &x[rows.start * n_features..rows.end * n_features];
            chunks.gradient(&net, x_batch, &y[rows], &mut grad);
            net.update(|params| adam.step(config.learning_rate, &mut [(params, &grad)]));
        }
    }
    Ok(net.params().to_vec())
}

/// The output logit of every row of `x`.
pub fn predict(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    hidden_layers: &[usize],
    params: &[f32],
) -> Result<Vec<f32>, String> {
    let layout = Layout::new(n_features, hidden_layers)?;
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
    let net = Network::new(&layout.dims, params);
    let mut logits = vec![0.0f32; n_rows];
    net.logits(crate::simd::get_backend(), x, &mut logits);
    Ok(logits)
}
