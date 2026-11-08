use super::reference;
use super::reference::bce_with_logits;
use super::fused::{self, Dense};
use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-4;

fn random(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOLERANCE * (1.0 + e.abs()),
            "index {i}: {a} != {e}"
        );
    }
}

/// Runs `check` once with every SIMD backend this machine supports selected.
fn for_each_backend(check: impl Fn()) {
    for name in ["scalar", "neon", "avx2", "avx512"] {
        if crate::simd::set_backend(name).is_ok() {
            check();
        }
    }
    crate::simd::clear_backend();
}

// (rows, k, n): shapes of the FDR network, row tails, column counts that are not a multiple of
// the vector width (masked tails) on either side of the tile width, odd shapes that exercise
// every row and column tail tile of the portable kernels, a reduction longer than the
// weight-gradient block, and an empty reduction
const SHAPES: [(usize, usize, usize); 15] = [
    (64, 48, 128),
    (8, 40, 32),
    (130, 128, 64),
    (7, 64, 32),
    (257, 32, 16),
    (9, 5, 7),
    (7, 3, 29),
    (2, 17, 1),
    (1, 1, 2),
    (3, 0, 5),
    (21, 33, 40),
    (13, 17, 100),
    (3, 1, 1),
    (300, 24, 48),
    (11, 70, 9),
];

#[test]
#[serial]
fn test_dense_forward_matches_reference() {
    for_each_backend(|| {
        // Given random layers of several shapes
        let mut rng = StdRng::seed_from_u64(1);
        for (rows, k, n) in SHAPES {
            for activation in [Activation::Linear, Activation::Relu] {
                let x = random(&mut rng, rows * k);
                let wt = random(&mut rng, k * n);
                let bias = random(&mut rng, n);

                // When computing the forward pass with the dispatched backend and the reference
                let mut out = vec![0.0; rows * n];
                dense_forward(&x, rows, k, &wt, n, &bias, activation, &mut out);
                let mut expected = vec![0.0; rows * n];
                reference::dense_forward(&x, rows, k, &wt, n, &bias, activation, &mut expected);

                // Then both agree
                assert_close(&out, &expected);
            }
        }
    });
}

#[test]
#[serial]
fn test_dense_backward_input_matches_reference() {
    for_each_backend(|| {
        // Given random gradients, weights and previous activations
        let mut rng = StdRng::seed_from_u64(2);
        for (rows, k, n) in SHAPES {
            let dz = random(&mut rng, rows * n);
            let w = random(&mut rng, n * k);
            let a_prev = random(&mut rng, rows * k);

            // When propagating to the input
            let mut d_prev = vec![0.0; rows * k];
            dense_backward_input(&dz, rows, n, &w, k, &a_prev, &mut d_prev);
            let mut expected = vec![0.0; rows * k];
            reference::dense_backward_input(&dz, rows, n, &w, k, &a_prev, &mut expected);

            // Then both agree and inactive units get no gradient
            assert_close(&d_prev, &expected);
            for (d, a) in d_prev.iter().zip(&a_prev) {
                if *a <= 0.0 {
                    assert_eq!(*d, 0.0);
                }
            }
        }
    });
}

#[test]
#[serial]
fn test_dense_weight_grad_accumulates_like_reference() {
    for_each_backend(|| {
        // Given non-zero gradient buffers to accumulate into
        let mut rng = StdRng::seed_from_u64(3);
        for (rows, k, n) in SHAPES {
            let a_prev = random(&mut rng, rows * k);
            let dz = random(&mut rng, rows * n);
            let grad_wt_start = random(&mut rng, k * n);
            let grad_b_start = random(&mut rng, n);

            // When accumulating the weight gradients
            let (mut grad_wt, mut grad_b) = (grad_wt_start.clone(), grad_b_start.clone());
            dense_weight_grad(&a_prev, rows, k, &dz, n, &mut grad_wt, &mut grad_b);
            let (mut expected_wt, mut expected_b) = (grad_wt_start, grad_b_start);
            reference::dense_weight_grad(
                &a_prev,
                rows,
                k,
                &dz,
                n,
                &mut expected_wt,
                &mut expected_b,
            );

            // Then both agree
            assert_close(&grad_wt, &expected_wt);
            assert_close(&grad_b, &expected_b);
        }
    });
}

#[test]
#[serial]
fn test_output_layer_matches_reference() {
    for_each_backend(|| {
        // Given a final hidden layer and binary labels
        let mut rng = StdRng::seed_from_u64(4);
        for (rows, k) in [(33, 32), (8, 6), (5, 37), (3, 1), (16, 16), (37, 40), (100, 23)] {
            let a = random(&mut rng, rows * k);
            let w = random(&mut rng, k);
            let y: Vec<f32> = (0..rows).map(|r| (r % 2) as f32).collect();

            // When running the fused output layer
            let (mut dz, mut grad_w, mut grad_b) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
            output_layer_train(
                &a,
                rows,
                k,
                &w,
                0.3,
                &y,
                0.5,
                &mut dz,
                &mut grad_w,
                &mut grad_b,
            );
            let (mut e_dz, mut e_grad_w, mut e_grad_b) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
            reference::output_layer_train(
                &a,
                rows,
                k,
                &w,
                0.3,
                &y,
                0.5,
                &mut e_dz,
                &mut e_grad_w,
                &mut e_grad_b,
            );
            let mut logits = vec![0.0; rows];
            output_layer_logits(&a, rows, k, &w, 0.3, &mut logits);
            let mut e_logits = vec![0.0; rows];
            reference::output_layer_logits(&a, rows, k, &w, 0.3, &mut e_logits);

            // Then gradients and logits agree
            assert_close(&dz, &e_dz);
            assert_close(&grad_w, &e_grad_w);
            assert_close(&[grad_b], &[e_grad_b]);
            assert_close(&logits, &e_logits);
        }
    });
}

#[test]
fn test_output_layer_gradient_matches_finite_difference() {
    // Given one row through the output layer
    let mut rng = StdRng::seed_from_u64(5);
    let k = 8;
    let a: Vec<f32> = random(&mut rng, k).iter().map(|v| v.abs()).collect();
    let mut w = random(&mut rng, k);
    let y = [1.0];

    // When comparing the analytic weight gradient with a central difference of the loss
    let (mut dz, mut grad_w, mut grad_b) = (vec![0.0; k], vec![0.0; k], 0.0);
    reference::output_layer_train(
        &a,
        1,
        k,
        &w,
        0.1,
        &y,
        1.0,
        &mut dz,
        &mut grad_w,
        &mut grad_b,
    );
    let eps = 1e-3;
    let loss_at = |w: &[f32]| {
        let logit = 0.1 + a.iter().zip(w).map(|(x, wv)| x * wv).sum::<f32>();
        bce_with_logits(logit, y[0])
    };
    w[2] += eps;
    let up = loss_at(&w);
    w[2] -= 2.0 * eps;
    let down = loss_at(&w);

    // Then they agree
    assert!((grad_w[2] - (up - down) / (2.0 * eps)).abs() < 1e-3);
}

#[test]
#[serial]
fn test_kernels_leave_memory_past_the_output_untouched() {
    for_each_backend(|| {
        // Given outputs followed by sentinels, for shapes with masked column and row tails
        let mut rng = StdRng::seed_from_u64(6);
        for (rows, k, n) in SHAPES {
            let x = random(&mut rng, rows * k);
            let wt = random(&mut rng, k * n);
            let bias = random(&mut rng, n);
            let dz = random(&mut rng, rows * n);
            let sentinel = 7.5;
            let mut out = vec![sentinel; rows * n + 40];
            let mut d_prev = vec![sentinel; rows * k + 40];
            let mut logits = vec![sentinel; rows + 40];
            let (mut grad_wt, mut grad_b) = (vec![0.0; k * n + 40], vec![0.0; n + 40]);
            grad_wt[k * n..].fill(sentinel);
            grad_b[n..].fill(sentinel);

            // When running the kernels on the leading part of the buffers
            dense_forward(&x, rows, k, &wt, n, &bias, Activation::Relu, &mut out);
            dense_backward_input(&dz, rows, n, &wt, k, &x, &mut d_prev);
            dense_weight_grad(&x, rows, k, &dz, n, &mut grad_wt, &mut grad_b);
            output_layer_logits(&x, rows, k, &wt[..k], 0.1, &mut logits);

            // Then nothing past the outputs changed
            assert!(out[rows * n..].iter().all(|&v| v == sentinel));
            assert!(d_prev[rows * k..].iter().all(|&v| v == sentinel));
            assert!(grad_wt[k * n..].iter().all(|&v| v == sentinel));
            assert!(grad_b[n..].iter().all(|&v| v == sentinel));
            assert!(logits[rows..].iter().all(|&v| v == sentinel));
        }
    });
}

#[test]
#[serial]
fn test_kernels_match_reference_at_any_buffer_offset() {
    for_each_backend(|| {
        // Given operands that start 0-3 floats past an allocation (cache-line splits, staging),
        // including a weight gradient longer than one staging block
        let mut rng = StdRng::seed_from_u64(7);
        for (rows, k, n) in [(37, 40, 128), (700, 24, 48), (10, 3, 20)] {
            for offset in 0..4 {
                let x = random(&mut rng, rows * k + offset);
                let wt = random(&mut rng, k * n + offset);
                let bias = random(&mut rng, n);
                let dz = random(&mut rng, rows * n + offset);
                let (x, wt, dz) = (&x[offset..], &wt[offset..], &dz[offset..]);

                // When running the kernels on the offset operands
                let mut out = vec![0.0; rows * n];
                dense_forward(x, rows, k, wt, n, &bias, Activation::Relu, &mut out);
                let mut d_prev = vec![0.0; rows * k];
                dense_backward_input(dz, rows, n, wt, k, x, &mut d_prev);
                let (mut grad_wt, mut grad_b) = (vec![0.5; k * n], vec![0.5; n]);
                dense_weight_grad(x, rows, k, dz, n, &mut grad_wt, &mut grad_b);

                // Then they agree with the reference
                let mut e_out = vec![0.0; rows * n];
                reference::dense_forward(x, rows, k, wt, n, &bias, Activation::Relu, &mut e_out);
                let mut e_d_prev = vec![0.0; rows * k];
                reference::dense_backward_input(dz, rows, n, wt, k, x, &mut e_d_prev);
                let (mut e_grad_wt, mut e_grad_b) = (vec![0.5; k * n], vec![0.5; n]);
                reference::dense_weight_grad(x, rows, k, dz, n, &mut e_grad_wt, &mut e_grad_b);
                assert_close(&out, &e_out);
                assert_close(&d_prev, &e_d_prev);
                assert_close(&grad_wt, &e_grad_wt);
                assert_close(&grad_b, &e_grad_b);
            }
        }
    });
}

#[test]
#[serial]
fn test_output_layer_matches_reference_for_saturated_logits() {
    for_each_backend(|| {
        // Given logits from about -200 to 200, where exp(-|logit|) underflows
        let mut rng = StdRng::seed_from_u64(8);
        let (rows, k) = (45, 16);
        let a: Vec<f32> = random(&mut rng, rows * k)
            .iter()
            .map(|v| v * 25.0)
            .collect();
        let w = random(&mut rng, k);
        let y: Vec<f32> = (0..rows).map(|r| (r % 3 == 0) as u8 as f32).collect();

        // When running the fused output layer with the dispatched backend and the reference
        let (mut dz, mut grad_w, mut grad_b) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
        output_layer_train(
            &a,
            rows,
            k,
            &w,
            -0.2,
            &y,
            1.0,
            &mut dz,
            &mut grad_w,
            &mut grad_b,
        );
        let (mut e_dz, mut e_grad_w, mut e_grad_b) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
        reference::output_layer_train(
            &a,
            rows,
            k,
            &w,
            -0.2,
            &y,
            1.0,
            &mut e_dz,
            &mut e_grad_w,
            &mut e_grad_b,
        );

        // Then the gradients agree
        assert_close(&dz, &e_dz);
        assert_close(&grad_w, &e_grad_w);
        assert_close(&[grad_b], &[e_grad_b]);
    });
}

/// A random network with `dims` and its backward weight copy.
fn random_network(rng: &mut StdRng, dims: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let size = Network::param_count(dims);
    let params: Vec<f32> = random(rng, size).iter().map(|v| v * 0.5).collect();
    let mut back = vec![0.0; size];
    Network::backward_weights(dims, &params, &mut back);
    (params, back)
}

/// Gradients of one `train_chunk` call with the selected backend: (params, input).
fn chunk_gradients(
    dims: &[usize],
    params: &[f32],
    back: &[f32],
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rows = y.len();
    let net = Network::new(dims, params, back);
    let mut ws = TrainWorkspace::new(dims);
    let mut grad = vec![0.25; params.len()];
    let mut input_grad = vec![0.0; rows * dims[0]];
    train_chunk(
        &net,
        x,
        y,
        rows,
        0.5,
        &mut ws,
        &mut grad,
        Some(&mut input_grad),
    );
    (grad, input_grad)
}

// network shapes: the FDR network, odd widths (scalar fallbacks), one hidden layer
const NETWORKS: [&[usize]; 3] = [&[40, 128, 64, 32, 1], &[7, 48, 17, 1], &[16, 32, 1]];

#[test]
#[serial]
fn test_train_chunk_matches_scalar_on_every_backend() {
    let mut rng = StdRng::seed_from_u64(6);
    for dims in NETWORKS {
        // Given a random network and more rows than one row block, with a ragged tail
        let rows = 2 * TRAIN_ROW_BLOCK + 13;
        let (params, back) = random_network(&mut rng, dims);
        let x = random(&mut rng, rows * dims[0]);
        let y: Vec<f32> = (0..rows).map(|r| (r % 3 == 0) as u8 as f32).collect();
        crate::simd::set_backend("scalar").unwrap();
        let (expected, expected_input) = chunk_gradients(dims, &params, &back, &x, &y);

        for_each_backend(|| {
            // When training one chunk with each backend
            let (grad, input_grad) = chunk_gradients(dims, &params, &back, &x, &y);

            // Then the gradients agree with the scalar reference
            assert_close(&grad, &expected);
            assert_close(&input_grad, &expected_input);
        });
    }
}

/// Layer-by-layer reference of [`fused::Network::logits`] with the scalar kernels.
fn network_reference(
    x: &[f32],
    rows: usize,
    k: usize,
    hidden: &[Dense],
    w_out: &[f32],
    b_out: f32,
) -> Vec<f32> {
    let (mut input, mut inputs) = (x.to_vec(), k);
    for layer in hidden {
        let mut out = vec![0.0; rows * layer.n];
        reference::dense_forward(
            &input,
            rows,
            inputs,
            layer.wt,
            layer.n,
            layer.bias,
            Activation::Relu,
            &mut out,
        );
        (input, inputs) = (out, layer.n);
    }
    let mut logits = vec![0.0; rows];
    reference::output_layer_logits(&input, rows, inputs, w_out, b_out, &mut logits);
    logits
}

#[test]
#[serial]
fn test_train_chunk_gradient_matches_finite_difference() {
    // Given a small network and a few rows
    let mut rng = StdRng::seed_from_u64(7);
    let dims = [3, 5, 4, 1];
    let (params, back) = random_network(&mut rng, &dims);
    let rows = 6;
    let x = random(&mut rng, rows * dims[0]);
    let y: Vec<f32> = (0..rows).map(|r| (r % 2) as f32).collect();
    crate::simd::set_backend("scalar").unwrap();
    let (grad, input_grad) = chunk_gradients(&dims, &params, &back, &x, &y);
    crate::simd::clear_backend();

    // When comparing with central differences of the summed loss (scaled by 0.5)
    let loss = |params: &[f32], x: &[f32]| -> f64 {
        let mut total = 0.0;
        for r in 0..rows {
            let mut a: Vec<f32> = x[r * dims[0]..(r + 1) * dims[0]].to_vec();
            let mut offset = 0;
            for l in 0..dims.len() - 1 {
                let (k, n) = (dims[l], dims[l + 1]);
                let mut z = params[offset + k * n..offset + k * n + n].to_vec();
                for p in 0..k {
                    for j in 0..n {
                        z[j] += a[p] * params[offset + p * n + j];
                    }
                }
                offset += k * n + n;
                a = if l + 2 < dims.len() {
                    z.iter().map(|v| v.max(0.0)).collect()
                } else {
                    z
                };
            }
            total += f64::from(bce_with_logits(a[0], y[r])) * 0.5;
        }
        total
    };
    let eps = 1e-2;
    let central = |f: &dyn Fn(f32) -> f64| ((f(eps) - f(-eps)) / (2.0 * f64::from(eps))) as f32;

    // Then the parameter gradients (on top of the initial 0.25) and the input gradient agree
    for i in 0..params.len() {
        let numeric = central(&|d| {
            let mut p = params.clone();
            p[i] += d;
            loss(&p, &x)
        });
        assert!(
            (grad[i] - 0.25 - numeric).abs() < 2e-3,
            "param {i}: {} vs {numeric}",
            grad[i] - 0.25
        );
    }
    for i in 0..x.len() {
        let numeric = central(&|d| {
            let mut xd = x.clone();
            xd[i] += d;
            loss(&params, &xd)
        });
        assert!(
            (input_grad[i] - numeric).abs() < 2e-3,
            "input {i}: {} vs {numeric}",
            input_grad[i]
        );
    }
}

#[test]
fn test_aligned_buffer_starts_on_a_cache_line() {
    for len in [0, 1, 17, 15617] {
        // Given a buffer copied from a slice
        let values: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let buf = AlignedBuf::from_slice(&values);

        // Then it has the values and starts on a 64-byte boundary
        assert_eq!(&buf[..], &values[..]);
        assert_eq!(buf.as_ptr() as usize % 64, 0);
    }
}

#[test]
#[serial]
fn test_network_logits_match_the_layer_by_layer_reference() {
    for_each_backend(|| {
        // Given networks of the FDR shape, odd feature counts and widths that need padding,
        // a single row, and row counts that leave partial blocks and chunks
        let mut rng = StdRng::seed_from_u64(6);
        let cases: [(usize, usize, &[usize]); 6] = [
            (1000, 40, &[128, 64, 32]),
            (1, 40, &[128, 64, 32]),
            (8195, 13, &[128, 64, 32]),
            (77, 5, &[33, 7]),
            (130, 1, &[300]),
            (9, 3, &[1, 1, 1]),
        ];
        for (rows, k, widths) in cases {
            let x = random(&mut rng, rows * k);
            let (mut wts, mut biases, mut inputs) = (Vec::new(), Vec::new(), k);
            for &n in widths {
                wts.push(random(&mut rng, inputs * n));
                biases.push(random(&mut rng, n));
                inputs = n;
            }
            let hidden: Vec<Dense> = widths
                .iter()
                .enumerate()
                .map(|(l, &n)| Dense {
                    wt: &wts[l],
                    bias: &biases[l],
                    n,
                })
                .collect();
            let w_out = random(&mut rng, inputs);

            // When predicting fused, in parallel and on one thread
            let network = fused::Network::new(k, &hidden, &w_out, 0.2);
            let mut logits = vec![0.0; rows];
            network.logits(&x, &mut logits);
            let mut serial = vec![0.0; rows];
            network.logits_serial(&x, &mut serial);

            // Then both match the layer-by-layer reference
            let expected = network_reference(&x, rows, k, &hidden, &w_out, 0.2);
            assert_close(&logits, &expected);
            assert_close(&serial, &expected);
        }
    });
}
