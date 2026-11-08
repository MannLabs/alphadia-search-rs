//! Per-kernel throughput of the dense-layer kernels on the FDR network shapes (40-128-64-32-1),
//! single-threaded. `cargo run --release --example mlp_kernels [avx512|avx2|scalar] [rows]`
//! prints GFLOP/s per kernel and shape (best of several repeats).

#[cfg(target_arch = "x86_64")]
use alphadia_search_rs::mlp::x86::{avx2, avx512};
use alphadia_search_rs::mlp::{scalar, Activation};
use std::hint::black_box;
use std::time::Instant;

struct Kernels {
    forward: fn(&[f32], usize, usize, &[f32], usize, &[f32], Activation, &mut [f32]),
    backward_input: fn(&[f32], usize, usize, &[f32], usize, &[f32], &mut [f32]),
    weight_grad: fn(&[f32], usize, usize, &[f32], usize, &mut [f32], &mut [f32]),
    #[allow(clippy::type_complexity)]
    output_train:
        fn(&[f32], usize, usize, &[f32], f32, &[f32], f32, &mut [f32], &mut [f32], &mut f32) -> f32,
    output_logits: fn(&[f32], usize, usize, &[f32], f32, &mut [f32]),
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
    let layers = [(40usize, 128usize), (128, 64), (64, 32)];
    println!("backend {backend}, train rows {train_rows}, predict rows {predict_rows}");
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
            let mut out = vec![0.0; rows * n];
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
        let mut d_prev = vec![0.0; rows * k];
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
        let mut gw = vec![0.0; k * n];
        let mut gb = vec![0.0; n];
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
        let mut dz = vec![0.0; rows * k];
        let mut gw = vec![0.0; k];
        let mut gb = 0.0;
        let flops = 4.0 * (rows * k) as f64;
        let s = time(flops, || {
            black_box((kn.output_train)(
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
        report("train output_train 32", flops, s);
        total_train += s;
        let rows = predict_rows;
        let a = data(rows * k, 11);
        let mut logits = vec![0.0; rows];
        let flops = 2.0 * (rows * k) as f64;
        let s = time(flops, || {
            (kn.output_logits)(black_box(&a), rows, k, &w, 0.1, &mut logits)
        });
        report("pred output_logits 32", flops, s);
    }
    println!("train chunk total {:.2} us", total_train * 1e6);
}
