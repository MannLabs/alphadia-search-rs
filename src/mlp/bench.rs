//! Single-thread throughput of the dense-layer kernels on the FDR network's shapes, for each
//! available backend and the naive reference:
//! `cargo test --release mlp::bench -- --ignored --nocapture`.

use super::reference;
use super::*;
use std::hint::black_box;
use std::time::Instant;

/// GFLOP/s of `run`, which performs `flops` per call; best of 30 repeats of ~10 ms each (the
/// best filters out other load on the core and its hyperthread sibling).
fn gflops(flops: usize, mut run: impl FnMut()) -> f64 {
    run();
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed().as_secs_f64() < 0.01 {
        run();
        calls += 1;
    }
    let calls = calls.max(1);
    (0..30)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..calls {
                run();
            }
            (flops * calls) as f64 / start.elapsed().as_secs_f64() / 1e9
        })
        .fold(0.0, f64::max)
}

#[allow(clippy::type_complexity)]
type Kernels = (
    fn(&[f32], usize, usize, &[f32], usize, &[f32], Activation, &mut [f32]),
    fn(&[f32], usize, usize, &[f32], usize, &[f32], &mut [f32]),
    fn(&[f32], usize, usize, &[f32], usize, &mut [f32], &mut [f32]),
    fn(&[f32], usize, usize, &[f32], f32, &[f32], f32, &mut [f32], &mut [f32], &mut f32),
);

#[test]
#[ignore]
#[serial_test::serial]
fn bench_kernels() {
    let rows = 625;
    let layers = [(40, 128), (128, 64), (64, 32), (127, 61)];
    let dispatched: Kernels = (
        dense_forward,
        dense_backward_input,
        dense_weight_grad,
        output_layer_train,
    );
    let naive: Kernels = (
        reference::dense_forward,
        reference::dense_backward_input,
        reference::dense_weight_grad,
        reference::output_layer_train,
    );
    let mut backends: Vec<(&str, Kernels)> = vec![("reference", naive)];
    for name in ["scalar", "neon", "avx2", "avx512"] {
        if crate::simd::set_backend(name).is_ok() {
            backends.push((name, dispatched));
        }
    }
    println!("GFLOP/s, {rows} rows, one thread");
    println!(
        "{:10} {:>10} {:>8} {:>8} {:>8}",
        "backend", "k x n", "fwd", "bwd_in", "w_grad"
    );
    for (name, (forward, backward, weight_grad, output)) in backends {
        if name != "reference" {
            crate::simd::set_backend(name).unwrap();
        }
        for (k, n) in layers {
            let x = vec![0.5f32; rows * k];
            let wt = vec![0.01f32; k * n];
            let bias = vec![0.1f32; n];
            let dz = vec![0.2f32; rows * n];
            let mut out = vec![0.0f32; rows * n];
            let mut d_prev = vec![0.0f32; rows * k];
            let (mut gw, mut gb) = (vec![0.0f32; k * n], vec![0.0f32; n]);
            let flops = 2 * rows * k * n;
            let f = gflops(flops, || {
                forward(
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
            let b = gflops(flops, || {
                backward(black_box(&dz), rows, n, &wt, k, &x, &mut d_prev)
            });
            let g = gflops(flops, || {
                weight_grad(black_box(&x), rows, k, &dz, n, &mut gw, &mut gb)
            });
            println!(
                "{name:10} {:>10} {f:8.2} {b:8.2} {g:8.2}",
                format!("{k}x{n}")
            );
        }
        let k = 32;
        let (a, w, y) = (vec![0.5f32; rows * k], vec![0.01f32; k], vec![1.0f32; rows]);
        let (mut dz, mut gw, mut gb) = (vec![0.0f32; rows * k], vec![0.0f32; k], 0.0f32);
        let o = gflops(4 * rows * k, || {
            black_box(output(
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
            ));
        });
        println!("{name:10} {:>10} output_train {o:.2}", "32x1");
    }
    crate::simd::clear_backend();
}
