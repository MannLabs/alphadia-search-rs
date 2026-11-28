//! Single-threaded throughput of the layer kernels on the FDR network shapes and of the whole
//! network, for every SIMD backend that this machine supports. Run it with
//!
//! `cargo test --release --lib mlp::bench -- --ignored --nocapture`
//!
//! The output is GFLOP/s per kernel and shape, and ns per row for the network (best of several
//! repeats). `MLP_BENCH_ROWS` sets the rows of a training chunk (default 625).

use super::{Activation, Network, TrainWorkspace};
use crate::simd::{get_backend, SimdBackend};
use std::hint::black_box;
use std::time::Instant;

const LAYERS: [(usize, usize); 3] = [(40, 128), (128, 64), (64, 32)];
const PREDICT_ROWS: usize = 4096;

fn data(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
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

fn kernels(kn: &dyn SimdBackend, train_rows: usize) {
    for (rows, label) in [(train_rows, "train"), (PREDICT_ROWS, "pred")] {
        for &(k, n) in &LAYERS {
            let (x, wt, bias) = (data(rows * k, 1), data(k * n, 2), data(n, 3));
            let mut out = vec![0.0; rows * n];
            let flops = 2.0 * (rows * k * n) as f64;
            let s = time(flops, || {
                kn.mlp_dense_forward(
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
        }
    }
    let rows = train_rows;
    for &(k, n) in &LAYERS[1..] {
        let (dz, w, a_prev) = (data(rows * n, 4), data(n * k, 5), data(rows * k, 6));
        let mut d_prev = vec![0.0; rows * k];
        let flops = 2.0 * (rows * k * n) as f64;
        let s = time(flops, || {
            kn.mlp_dense_backward_input(black_box(&dz), rows, n, &w, k, &a_prev, &mut d_prev)
        });
        report(&format!("train backward_in {n}->{k}"), flops, s);
    }
    for &(k, n) in &LAYERS {
        let (a_prev, dz) = (data(rows * k, 7), data(rows * n, 8));
        let (mut gw, mut gb) = (vec![0.0; k * n], vec![0.0; n]);
        let flops = 2.0 * (rows * k * n) as f64;
        let s = time(flops, || {
            kn.mlp_dense_weight_grad(black_box(&a_prev), rows, k, &dz, n, &mut gw, &mut gb)
        });
        report(&format!("train weight_grad {k}x{n}"), flops, s);
    }
    let k = 32;
    let (a, w) = (data(rows * k, 9), data(k, 10));
    let y: Vec<f32> = (0..rows).map(|r| (r % 2) as f32).collect();
    let (mut dz, mut gw, mut gb) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
    let flops = 4.0 * (rows * k) as f64;
    let s = time(flops, || {
        kn.mlp_output_layer_train(
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
        )
    });
    report("train output_train 32", flops, s);
    let a = data(PREDICT_ROWS * k, 11);
    let mut logits = vec![0.0; PREDICT_ROWS];
    let flops = 2.0 * (PREDICT_ROWS * k) as f64;
    let s = time(flops, || {
        kn.mlp_output_layer_logits(black_box(&a), PREDICT_ROWS, k, &w, 0.1, &mut logits)
    });
    report("pred output_logits 32", flops, s);
}

fn network(kn: &dyn SimdBackend, train_rows: usize) {
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
    // The flops of the forward pass and the weight gradient of every layer, and of the input
    // gradient of every layer but the first.
    let products: Vec<usize> = dims.windows(2).map(|d| d[0] * d[1]).collect();
    let per_row = 4 * products.iter().sum::<usize>() + 2 * products[1..].iter().sum::<usize>();
    let flops = (per_row * train_rows) as f64;
    let s = time(flops, || {
        net.train_chunk(kn, black_box(&x), &y, 1e-3, &mut ws, &mut grad, None)
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
    let s = pool.install(|| time(flops, || net.logits(kn, black_box(&x), &mut logits)));
    println!(
        "{:<28} {:>8.1} GFLOP/s  {:>8.1} ns/row",
        format!("network logits {rows}"),
        flops / s * 1e-9,
        s * 1e9 / rows as f64
    );
}

#[test]
#[ignore = "benchmark"]
#[serial_test::serial]
fn bench_mlp_kernels() {
    let train_rows = std::env::var("MLP_BENCH_ROWS").map_or(625, |v| v.parse().unwrap());
    for name in ["scalar", "neon", "avx2", "avx512"] {
        if crate::simd::set_backend(name).is_err() {
            continue;
        }
        println!("=== backend {name}, train rows {train_rows}, predict rows {PREDICT_ROWS}");
        kernels(get_backend(), train_rows);
        network(get_backend(), train_rows);
    }
    crate::simd::clear_backend();
}
