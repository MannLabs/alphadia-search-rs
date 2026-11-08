//! Naive reference implementations of the dense-layer kernels: the oracle that the tests compare
//! every backend against, including the portable kernels in `scalar`. Keep these obviously correct
//! rather than fast.

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

/// Numerically stable `log(1 + exp(-|z|)) + max(z, 0) - y z`.
pub fn bce_with_logits(logit: f32, y: f32) -> f32 {
    logit.max(0.0) - logit * y + (-logit.abs()).exp().ln_1p()
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
) {
    for r in 0..rows {
        let a_row = &a[r * k..(r + 1) * k];
        let logit = b + a_row.iter().zip(w).map(|(x, wv)| x * wv).sum::<f32>();
        let d = (1.0 / (1.0 + (-logit).exp()) - y[r]) * loss_scale;
        *grad_b += d;
        for p in 0..k {
            grad_w[p] += d * a_row[p];
            dz_prev[r * k + p] = if a_row[p] > 0.0 { d * w[p] } else { 0.0 };
        }
    }
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

/// The kernels above as a backend of [`super::Network`].
pub struct Kernels;

impl super::LayerKernels for Kernels {
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
    ) {
        dense_forward(x, rows, k, wt, n, bias, activation, out)
    }

    fn dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &mut [f32],
    ) {
        dense_backward_input(dz, rows, n, w, k, a_prev, d_prev)
    }

    fn dense_weight_grad(
        &self,
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &mut [f32],
        grad_b: &mut [f32],
    ) {
        dense_weight_grad(a_prev, rows, k, dz, n, grad_wt, grad_b)
    }

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
    ) {
        output_layer_train(a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b)
    }

    fn output_layer_logits(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        logits: &mut [f32],
    ) {
        output_layer_logits(a, rows, k, w, b, logits)
    }
}

/// The logits of `net` for every row of `x`, one row and layer at a time from the flat parameters.
pub fn logits(net: &super::Network, x: &[f32], logits: &mut [f32]) {
    let (dims, params) = (net.dims(), net.params());
    for (row, logit) in x.chunks_exact(dims[0]).zip(logits.iter_mut()) {
        let (mut a, mut offset) = (row.to_vec(), 0);
        for (l, d) in dims.windows(2).enumerate() {
            let (k, n) = (d[0], d[1]);
            let mut z = vec![0.0; n];
            dense_forward(
                &a,
                1,
                k,
                &params[offset..],
                n,
                &params[offset + k * n..],
                Activation::Linear,
                &mut z,
            );
            let hidden = l + 2 < dims.len();
            a = if hidden {
                z.iter().map(|v| v.max(0.0)).collect()
            } else {
                z
            };
            offset += k * n + n;
        }
        *logit = a[0];
    }
}
