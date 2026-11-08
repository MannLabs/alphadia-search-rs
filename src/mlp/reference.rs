//! Naive reference implementations of the dense-layer kernels: the oracle that the tests compare
//! every backend against, including the portable kernels in `scalar`. Keep these obviously correct
//! rather than fast.

use super::scalar::bce_with_logits;
use super::Activation;

#[allow(clippy::too_many_arguments)]
pub fn dense_forward(
    x: &[f32],
    rows: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: &[f32],
    activation: Activation,
    out: &mut [f32],
) {
    for r in 0..rows {
        let out_row = &mut out[r * n..(r + 1) * n];
        out_row.copy_from_slice(&bias[..n]);
        for p in 0..k {
            let a = x[r * k + p];
            let w_row = &wt[p * n..(p + 1) * n];
            for (o, &w) in out_row.iter_mut().zip(w_row) {
                *o += a * w;
            }
        }
        if activation == Activation::Relu {
            out_row.iter_mut().for_each(|o| *o = o.max(0.0));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dense_backward_input(
    dz: &[f32],
    rows: usize,
    n: usize,
    w: &[f32],
    k: usize,
    a_prev: &[f32],
    d_prev: &mut [f32],
) {
    for r in 0..rows {
        let d_row = &mut d_prev[r * k..(r + 1) * k];
        d_row.iter_mut().for_each(|d| *d = 0.0);
        for j in 0..n {
            let g = dz[r * n + j];
            let w_row = &w[j * k..(j + 1) * k];
            for (d, &wv) in d_row.iter_mut().zip(w_row) {
                *d += g * wv;
            }
        }
        for (d, &a) in d_row.iter_mut().zip(&a_prev[r * k..(r + 1) * k]) {
            if a <= 0.0 {
                *d = 0.0;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dense_weight_grad(
    a_prev: &[f32],
    rows: usize,
    k: usize,
    dz: &[f32],
    n: usize,
    grad_wt: &mut [f32],
    grad_b: &mut [f32],
) {
    for r in 0..rows {
        let dz_row = &dz[r * n..(r + 1) * n];
        for (gb, &d) in grad_b.iter_mut().zip(dz_row) {
            *gb += d;
        }
        for p in 0..k {
            let a = a_prev[r * k + p];
            for (g, &d) in grad_wt[p * n..(p + 1) * n].iter_mut().zip(dz_row) {
                *g += a * d;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn output_layer_train(
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
    let mut loss = 0.0;
    for r in 0..rows {
        let a_row = &a[r * k..(r + 1) * k];
        let logit = b + a_row.iter().zip(w).map(|(x, wv)| x * wv).sum::<f32>();
        loss += bce_with_logits(logit, y[r]);
        let d = (1.0 / (1.0 + (-logit).exp()) - y[r]) * loss_scale;
        *grad_b += d;
        for p in 0..k {
            grad_w[p] += d * a_row[p];
            dz_prev[r * k + p] = if a_row[p] > 0.0 { d * w[p] } else { 0.0 };
        }
    }
    loss
}

pub fn output_layer_logits(
    a: &[f32],
    rows: usize,
    k: usize,
    w: &[f32],
    b: f32,
    logits: &mut [f32],
) {
    for r in 0..rows {
        logits[r] = b + a[r * k..(r + 1) * k]
            .iter()
            .zip(w)
            .map(|(x, wv)| x * wv)
            .sum::<f32>();
    }
}
