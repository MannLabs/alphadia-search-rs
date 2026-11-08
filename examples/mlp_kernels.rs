//! Single-threaded throughput of the dense-layer kernels on the FDR network shapes
//! (40-128-64-32-1), and of the whole network built from them: a training step per row chunk and
//! fused prediction. `cargo run --release --example mlp_kernels [avx512|avx2|scalar] [rows]`
//! prints GFLOP/s per kernel and shape and ns per row for the network (best of several repeats).
//! `MLP_MISALIGN=<floats>` offsets the kernel buffers from a cache line.

#[cfg(target_arch = "x86_64")]
use alphadia_search_rs::mlp::x86::{avx2, avx512};
use alphadia_search_rs::mlp::{scalar, Activation, LayerKernels, Network, TrainWorkspace};
use std::hint::black_box;
use std::time::Instant;

#[allow(clippy::type_complexity)]
struct Kernels {
    forward: fn(&[f32], usize, usize, &[f32], usize, &[f32], Activation, &mut [f32]),
    backward_input: fn(&[f32], usize, usize, &[f32], usize, &[f32], &mut [f32]),
    weight_grad: fn(&[f32], usize, usize, &[f32], usize, &mut [f32], &mut [f32]),
    output_train:
        fn(&[f32], usize, usize, &[f32], f32, &[f32], f32, &mut [f32], &mut [f32], &mut f32),
    output_logits: fn(&[f32], usize, usize, &[f32], f32, &mut [f32]),
}

impl LayerKernels for Kernels {
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
        (self.forward)(x, rows, k, wt, n, bias, activation, out)
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
        (self.backward_input)(dz, rows, n, w, k, a_prev, d_prev)
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
        (self.weight_grad)(a_prev, rows, k, dz, n, grad_wt, grad_b)
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
        (self.output_train)(a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b)
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
        (self.output_logits)(a, rows, k, w, b, logits)
    }
}

fn kernels(name: &str) -> Kernels {
    match name {
        #[cfg(target_arch = "x86_64")]
        "avx512" => Kernels {
            forward: avx512::dense_forward,
            backward_input: avx512::dense_backward_input,
            weight_grad: avx512::dense_weight_grad,
            output_train: avx512::output_layer_train,
            output_logits: avx512::output_layer_logits,
        },
        #[cfg(target_arch = "x86_64")]
        "avx2" => Kernels {
            forward: avx2::dense_forward,
            backward_input: avx2::dense_backward_input,
            weight_grad: avx2::dense_weight_grad,
            output_train: avx2::output_layer_train,
            output_logits: avx2::output_layer_logits,
        },
        _ => Kernels {
            forward: scalar::dense_forward_scalar,
            backward_input: scalar::dense_backward_input_scalar,
            weight_grad: scalar::dense_weight_grad_scalar,
            output_train: scalar::output_layer_train_scalar,
            output_logits: scalar::output_layer_logits_scalar,
        },
    }
}

/// A buffer whose first element is `MLP_MISALIGN` floats (default 0) past a 64-byte boundary,
/// so tile changes are not confused with the alignment the allocator happens to return.
struct Buf {
    v: Vec<f32>,
    off: usize,
    len: usize,
}

impl std::ops::Deref for Buf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.v[self.off..self.off + self.len]
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.v[self.off..self.off + self.len]
    }
}

fn misalign() -> usize {
    std::env::var("MLP_MISALIGN").map_or(0, |v| v.parse().unwrap())
}

fn zeros(len: usize) -> Buf {
    let v = vec![0.0f32; len + 32];
    let to_boundary = (64 - v.as_ptr() as usize % 64) % 64 / 4;
    Buf {
        v,
        off: to_boundary + misalign(),
        len,
    }
}

fn data(len: usize, seed: u32) -> Buf {
    let mut buf = zeros(len);
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *x = (s as f32 / u32::MAX as f32) - 0.5;
    }
    buf
}

/// Best-of-repeats seconds per call of `f`.
fn time(flops: f64, mut f: impl FnMut()) -> f64 {
    let calls = ((2e7 / flops) as usize).max(1);
    let mut best = f64::INFINITY;
    for _ in 0..50 {
        let t = Instant::now();
        for _ in 0..calls {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / calls as f64);
    }
    best
}

fn report(name: &str, flops: f64, secs: f64) {
    println!(
        "{name:<28} {:>8.1} GFLOP/s  {:>8.2} us",
        flops / secs * 1e-9,
        secs * 1e6
    );
}

/// Peak single-core FMA throughput: 16 independent accumulator chains, no memory traffic.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn peak_fma(iters: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = [_mm512_set1_ps(0.0); 16];
    let (a, b) = (_mm512_set1_ps(0.999), _mm512_set1_ps(1e-6));
    for _ in 0..iters {
        for v in acc.iter_mut() {
            *v = _mm512_fmadd_ps(*v, a, b);
        }
    }
    acc.iter().map(|&v| _mm512_reduce_add_ps(v)).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend = args.get(1).map_or("avx512", |s| s.as_str());
    let train_rows: usize = args.get(2).map_or(625, |s| s.parse().unwrap());
    let predict_rows = 4096;
    let kn = kernels(backend);
    // optional `k n`: time only that layer shape
    if let (Some(k), Some(n)) = (args.get(3), args.get(4)) {
        let (k, n): (usize, usize) = (k.parse().unwrap(), n.parse().unwrap());
        let rows = train_rows;
        let (x, wt, bias, dz) = (
            data(rows * k, 1),
            data(k * n, 2),
            data(n, 3),
            data(rows * n, 4),
        );
        let (mut out, mut d_prev) = (zeros(rows * n), zeros(rows * k));
        let (mut gw, mut gb) = (zeros(k * n), zeros(n));
        let flops = 2.0 * (rows * k * n) as f64;
        let s = time(flops, || {
            (kn.forward)(
                black_box(&x),
                rows,
                k,
                &wt,
                n,
                &bias,
                Activation::Relu,
                &mut out,
            )
        });
        report(&format!("forward {k}x{n}"), flops, s);
        let s = time(flops, || {
            (kn.backward_input)(black_box(&dz), rows, n, &wt, k, &x, &mut d_prev)
        });
        report(&format!("backward_in {n}->{k}"), flops, s);
        let s = time(flops, || {
            (kn.weight_grad)(black_box(&x), rows, k, &dz, n, &mut gw, &mut gb)
        });
        report(&format!("weight_grad {k}x{n}"), flops, s);
        return;
    }
    let layers = [(40usize, 128usize), (128, 64), (64, 32)];
    println!(
        "backend {backend}, train rows {train_rows}, predict rows {predict_rows}, buffers {} floats past 64-byte alignment",
        misalign()
    );
    #[cfg(target_arch = "x86_64")]
    if backend == "avx512" {
        let iters = 1_000_000;
        let flops = (iters * 16 * 16 * 2) as f64;
        // SAFETY: only run on AVX-512 machines when asked for that backend
        let s = time(flops, || {
            black_box(unsafe { peak_fma(black_box(iters)) });
        });
        report("peak fma (no memory)", flops, s);
    }
    let mut total_train = 0.0;
    for (rows, label) in [(train_rows, "train"), (predict_rows, "pred")] {
        for &(k, n) in &layers {
            let x = data(rows * k, 1);
            let wt = data(k * n, 2);
            let bias = data(n, 3);
            let mut out = zeros(rows * n);
            let flops = 2.0 * (rows * k * n) as f64;
            let s = time(flops, || {
                (kn.forward)(
                    black_box(&x),
                    rows,
                    k,
                    &wt,
                    n,
                    &bias,
                    Activation::Relu,
                    &mut out,
                )
            });
            report(&format!("{label} forward {k}x{n}"), flops, s);
            if label == "train" {
                total_train += s;
            }
        }
    }
    let rows = train_rows;
    for &(k, n) in &layers[1..] {
        let dz = data(rows * n, 4);
        let w = data(n * k, 5);
        let a_prev = data(rows * k, 6);
        let mut d_prev = zeros(rows * k);
        let flops = 2.0 * (rows * k * n) as f64;
        let s = time(flops, || {
            (kn.backward_input)(black_box(&dz), rows, n, &w, k, &a_prev, &mut d_prev)
        });
        report(&format!("train backward_in {n}->{k}"), flops, s);
        total_train += s;
    }
    for &(k, n) in &layers {
        let a_prev = data(rows * k, 7);
        let dz = data(rows * n, 8);
        let mut gw = zeros(k * n);
        let mut gb = zeros(n);
        let flops = 2.0 * (rows * k * n) as f64;
        let s = time(flops, || {
            (kn.weight_grad)(black_box(&a_prev), rows, k, &dz, n, &mut gw, &mut gb)
        });
        report(&format!("train weight_grad {k}x{n}"), flops, s);
        total_train += s;
    }
    {
        let k = 32;
        let a = data(rows * k, 9);
        let w = data(k, 10);
        let y: Vec<f32> = (0..rows).map(|r| (r % 2) as f32).collect();
        let mut dz = zeros(rows * k);
        let mut gw = zeros(k);
        let mut gb = 0.0;
        let flops = 4.0 * (rows * k) as f64;
        let s = time(flops, || {
            (kn.output_train)(
                black_box(&a),
                rows,
                k,
                &w,
                0.1,
                &y,
                1.0,
                &mut dz,
                &mut gw,
                &mut gb,
            );
        });
        report("train output_train 32", flops, s);
        total_train += s;
        let rows = predict_rows;
        let a = data(rows * k, 11);
        let mut logits = zeros(rows);
        let flops = 2.0 * (rows * k) as f64;
        let s = time(flops, || {
            (kn.output_logits)(black_box(&a), rows, k, &w, 0.1, &mut logits)
        });
        report("pred output_logits 32", flops, s);
    }
    println!("train chunk total {:.2} us", total_train * 1e6);

    // the whole network: one training step per chunk, and fused prediction on one thread
    let dims = [40usize, 128, 64, 32, 1];
    let params: Vec<f32> = data(Network::param_count(&dims), 12)
        .iter()
        .map(|v| v * 0.3)
        .collect();
    let net = Network::new(&dims, &params);
    let x = data(train_rows * dims[0], 13);
    let y: Vec<f32> = (0..train_rows).map(|r| (r % 3 == 0) as u8 as f32).collect();
    let mut ws = TrainWorkspace::new(&net);
    let mut grad = vec![0.0f32; params.len()];
    // forward and weight gradient of every layer, input gradient of all but the first
    let products: Vec<usize> = dims.windows(2).map(|d| d[0] * d[1]).collect();
    let per_row = 4 * products.iter().sum::<usize>() + 2 * products[1..].iter().sum::<usize>();
    let flops = (per_row * train_rows) as f64;
    let s = time(flops, || {
        net.train_chunk(&kn, black_box(&x), &y, 1e-3, &mut ws, &mut grad, None)
    });
    println!(
        "{:<28} {:>8.1} GFLOP/s  {:>8.1} ns/row",
        format!("network train_chunk {train_rows}"),
        flops / s * 1e-9,
        s * 1e9 / train_rows as f64
    );
    let rows = 65536;
    let x = data(rows * dims[0], 14);
    let mut logits = vec![0.0f32; rows];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let flops = (2 * products.iter().sum::<usize>() * rows) as f64;
    let s = pool.install(|| time(flops, || net.logits(&kn, black_box(&x), &mut logits)));
    println!(
        "{:<28} {:>8.1} GFLOP/s  {:>8.1} ns/row",
        format!("network logits {rows}"),
        flops / s * 1e-9,
        s * 1e9 / rows as f64
    );
}
