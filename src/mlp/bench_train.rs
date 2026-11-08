//! Single-thread timing of `train_chunk` on the FDR network shape (min of 7 runs):
//! `ROWS=512 BACKENDS=avx512,avx2 cargo test --release --lib bench_train -- --ignored --nocapture`
//! (pin it to one core with `taskset`).
use super::*;
use std::time::Instant;

fn rnd(n: usize, s: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

#[test]
#[ignore]
fn bench_train() {
    let rows: usize = std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let dims = [40usize, 128, 64, 32, 1];
    let size = Network::param_count(&dims);
    let mut s = 0x1234_5678_9abc_def1u64;
    // MISALIGN=1 shifts the parameter buffers by 16 bytes, as a Vec<f32> may be
    let shift = std::env::var("MISALIGN").map_or(0, |_| 4);
    let mut params = AlignedBuf::zeros(size + shift);
    params[shift..].copy_from_slice(
        &rnd(size, &mut s)
            .iter()
            .map(|v| v * 0.3)
            .collect::<Vec<_>>(),
    );
    let mut back = AlignedBuf::zeros(size + shift);
    Network::backward_weights(&dims, &params[shift..], &mut back[shift..]);
    let net = Network::new(&dims, &params[shift..], &back[shift..]);
    let x = rnd(rows * 40, &mut s);
    let y: Vec<f32> = rnd(rows, &mut s)
        .iter()
        .map(|v| (*v > 0.0) as u8 as f32)
        .collect();
    let mut ws = TrainWorkspace::new(&dims);
    let mut grad_buf = AlignedBuf::zeros(size + shift);
    let grad = &mut grad_buf[shift..];
    for name in std::env::var("BACKENDS")
        .unwrap_or("avx512".into())
        .split(',')
    {
        crate::simd::set_backend(name).unwrap();
        let reps = 1_000_000 / rows;
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..reps {
                train_chunk(&net, &x, &y, rows, 1e-3, &mut ws, grad, None);
            }
            best = best.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
        }
        println!("{name}: train_chunk {rows} rows: {best:.1} ns/row");
    }
    crate::simd::clear_backend();
}
