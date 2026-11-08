use super::reference;
use super::reference::bce_with_logits;
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
            let e_loss = reference::output_layer_train(
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
        let loss = output_layer_train(
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
        let e_loss = reference::output_layer_train(
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

        // Then loss and gradients agree
        assert_close(&[loss], &[e_loss]);
        assert_close(&dz, &e_dz);
        assert_close(&grad_w, &e_grad_w);
        assert_close(&[grad_b], &[e_grad_b]);
    });
}
