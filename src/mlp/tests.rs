use super::fused::{self, Dense};
use super::scalar::*;
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

// (rows, k, n): fast-path shapes of the FDR network, row tails, and a scalar fallback
const SHAPES: [(usize, usize, usize); 6] = [
    (64, 48, 128),
    (8, 40, 32),
    (130, 128, 64),
    (7, 64, 32),
    (257, 32, 16),
    (9, 5, 7),
];

#[test]
#[serial]
fn test_dense_forward_matches_scalar() {
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
                dense_forward_scalar(&x, rows, k, &wt, n, &bias, activation, &mut expected);

                // Then both agree
                assert_close(&out, &expected);
            }
        }
    });
}

#[test]
#[serial]
fn test_dense_backward_input_matches_scalar() {
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
            dense_backward_input_scalar(&dz, rows, n, &w, k, &a_prev, &mut expected);

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
fn test_dense_weight_grad_accumulates_like_scalar() {
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
            dense_weight_grad_scalar(&a_prev, rows, k, &dz, n, &mut expected_wt, &mut expected_b);

            // Then both agree
            assert_close(&grad_wt, &expected_wt);
            assert_close(&grad_b, &expected_b);
        }
    });
}

#[test]
#[serial]
fn test_output_layer_matches_scalar() {
    for_each_backend(|| {
        // Given a final hidden layer and binary labels
        let mut rng = StdRng::seed_from_u64(4);
        for (rows, k) in [(33, 32), (8, 6)] {
            let a = random(&mut rng, rows * k);
            let w = random(&mut rng, k);
            let y: Vec<f32> = (0..rows).map(|r| (r % 2) as f32).collect();

            // When running the fused output layer
            let (mut dz, mut grad_w, mut grad_b) = (vec![0.0; rows * k], vec![0.0; k], 0.0);
            let loss = output_layer_train(
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
            let e_loss = output_layer_train_scalar(
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
            output_layer_logits_scalar(&a, rows, k, &w, 0.3, &mut e_logits);

            // Then loss, gradients and logits agree
            assert_close(&[loss], &[e_loss]);
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
    output_layer_train_scalar(
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
        dense_forward_scalar(
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
    output_layer_logits_scalar(&input, rows, inputs, w_out, b_out, &mut logits);
    logits
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
