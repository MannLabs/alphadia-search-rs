//! NEON dense-layer kernels.
//!
//! The forward and input-gradient passes share one register-blocked GEMM tile: 4 rows x 16
//! columns of `C = A @ B` live in 16 vector registers for the whole reduction. A is read as
//! 4-wide vectors along the reduction axis and applied with lane-indexed FMAs, so neither
//! operand is packed. The epilogue (bias + ReLU, bias, or a ReLU mask from the previous
//! layer) is fused into the store. Shapes outside the fast path fall back to the scalar code.

#[cfg(target_arch = "aarch64")]
use super::scalar;
#[cfg(target_arch = "aarch64")]
use super::Activation;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Rows of a GEMM tile.
#[cfg(target_arch = "aarch64")]
const MR: usize = 4;
/// Columns of a GEMM tile (4 vectors).
#[cfg(target_arch = "aarch64")]
const NR: usize = 16;
/// Rows the weight-gradient kernel keeps in L1 while it sweeps all tiles.
#[cfg(target_arch = "aarch64")]
const GRAD_ROW_BLOCK: usize = 128;

#[cfg(target_arch = "aarch64")]
const EPILOGUE_LINEAR: u8 = 0;
#[cfg(target_arch = "aarch64")]
const EPILOGUE_RELU: u8 = 1;
#[cfg(target_arch = "aarch64")]
const EPILOGUE_RELU_MASK: u8 = 2;

#[cfg(target_arch = "aarch64")]
type Tile = [[float32x4_t; 4]; MR];

/// acc[r][q] += b[q] * a[r][LANE]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fma_lane<const LANE: i32>(acc: &mut Tile, b: &[float32x4_t; 4], a: &[float32x4_t; MR]) {
    for r in 0..MR {
        for q in 0..4 {
            acc[r][q] = vfmaq_laneq_f32::<LANE>(acc[r][q], b[q], a[r]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load4(ptr: *const f32) -> [float32x4_t; 4] {
    [
        vld1q_f32(ptr),
        vld1q_f32(ptr.add(4)),
        vld1q_f32(ptr.add(8)),
        vld1q_f32(ptr.add(12)),
    ]
}

/// C[rows_full x n] = epilogue(A[rows_full x kk] @ B[kk x n]); rows_full % MR == 0,
/// kk % 4 == 0, n % NR == 0. `aux` is the bias (length n) or the ReLU mask source
/// (rows x n), depending on the epilogue.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tiles<const EPILOGUE: u8>(
    a: *const f32,
    rows_full: usize,
    kk: usize,
    b: *const f32,
    n: usize,
    aux: *const f32,
    c: *mut f32,
) {
    let zero = vdupq_n_f32(0.0);
    for i0 in (0..rows_full).step_by(MR) {
        let a_rows = [
            a.add(i0 * kk),
            a.add((i0 + 1) * kk),
            a.add((i0 + 2) * kk),
            a.add((i0 + 3) * kk),
        ];
        for j0 in (0..n).step_by(NR) {
            let init = if EPILOGUE == EPILOGUE_RELU_MASK {
                [zero; 4]
            } else {
                load4(aux.add(j0))
            };
            let mut acc: Tile = [init; MR];
            let mut p = 0;
            while p < kk {
                let av = [
                    vld1q_f32(a_rows[0].add(p)),
                    vld1q_f32(a_rows[1].add(p)),
                    vld1q_f32(a_rows[2].add(p)),
                    vld1q_f32(a_rows[3].add(p)),
                ];
                fma_lane::<0>(&mut acc, &load4(b.add(p * n + j0)), &av);
                fma_lane::<1>(&mut acc, &load4(b.add((p + 1) * n + j0)), &av);
                fma_lane::<2>(&mut acc, &load4(b.add((p + 2) * n + j0)), &av);
                fma_lane::<3>(&mut acc, &load4(b.add((p + 3) * n + j0)), &av);
                p += 4;
            }
            for (r, acc_row) in acc.iter().enumerate() {
                let c_row = c.add((i0 + r) * n + j0);
                for (q, &value) in acc_row.iter().enumerate() {
                    let value = match EPILOGUE {
                        EPILOGUE_RELU => vmaxq_f32(value, zero),
                        EPILOGUE_RELU_MASK => {
                            let source = vld1q_f32(aux.add((i0 + r) * n + j0 + 4 * q));
                            let keep = vcgtq_f32(source, zero);
                            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(value), keep))
                        }
                        _ => value,
                    };
                    vst1q_f32(c_row.add(4 * q), value);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn dense_forward_neon(
    x: &[f32],
    rows: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: &[f32],
    activation: Activation,
    out: &mut [f32],
) {
    if k % 4 != 0 || n % NR != 0 {
        return scalar::dense_forward_scalar(x, rows, k, wt, n, bias, activation, out);
    }
    let full = rows - rows % MR;
    unsafe {
        match activation {
            Activation::Relu => gemm_tiles::<EPILOGUE_RELU>(
                x.as_ptr(),
                full,
                k,
                wt.as_ptr(),
                n,
                bias.as_ptr(),
                out.as_mut_ptr(),
            ),
            Activation::Linear => gemm_tiles::<EPILOGUE_LINEAR>(
                x.as_ptr(),
                full,
                k,
                wt.as_ptr(),
                n,
                bias.as_ptr(),
                out.as_mut_ptr(),
            ),
        }
    }
    if full < rows {
        scalar::dense_forward_scalar(
            &x[full * k..],
            rows - full,
            k,
            wt,
            n,
            bias,
            activation,
            &mut out[full * n..],
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn dense_backward_input_neon(
    dz: &[f32],
    rows: usize,
    n: usize,
    w: &[f32],
    k: usize,
    a_prev: &[f32],
    d_prev: &mut [f32],
) {
    // reduction over the n outputs, result columns are the k inputs
    if n % 4 != 0 || k % NR != 0 {
        return scalar::dense_backward_input_scalar(dz, rows, n, w, k, a_prev, d_prev);
    }
    let full = rows - rows % MR;
    unsafe {
        gemm_tiles::<EPILOGUE_RELU_MASK>(
            dz.as_ptr(),
            full,
            n,
            w.as_ptr(),
            k,
            a_prev.as_ptr(),
            d_prev.as_mut_ptr(),
        );
    }
    if full < rows {
        scalar::dense_backward_input_scalar(
            &dz[full * n..],
            rows - full,
            n,
            w,
            k,
            &a_prev[full * k..],
            &mut d_prev[full * k..],
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn dense_weight_grad_neon(
    a_prev: &[f32],
    rows: usize,
    k: usize,
    dz: &[f32],
    n: usize,
    grad_wt: &mut [f32],
    grad_b: &mut [f32],
) {
    if k % 4 != 0 || n % NR != 0 {
        return scalar::dense_weight_grad_scalar(a_prev, rows, k, dz, n, grad_wt, grad_b);
    }
    unsafe {
        let (a, d, g) = (a_prev.as_ptr(), dz.as_ptr(), grad_wt.as_mut_ptr());
        for r0 in (0..rows).step_by(GRAD_ROW_BLOCK) {
            let r1 = (r0 + GRAD_ROW_BLOCK).min(rows);
            for p0 in (0..k).step_by(4) {
                for j0 in (0..n).step_by(NR) {
                    // tile rows are the 4 inputs p0..p0+4, columns 16 outputs
                    let mut acc: Tile = [
                        load4(g.add(p0 * n + j0)),
                        load4(g.add((p0 + 1) * n + j0)),
                        load4(g.add((p0 + 2) * n + j0)),
                        load4(g.add((p0 + 3) * n + j0)),
                    ];
                    for r in r0..r1 {
                        let av = vld1q_f32(a.add(r * k + p0));
                        let dv = load4(d.add(r * n + j0));
                        for q in 0..4 {
                            acc[0][q] = vfmaq_laneq_f32::<0>(acc[0][q], dv[q], av);
                            acc[1][q] = vfmaq_laneq_f32::<1>(acc[1][q], dv[q], av);
                            acc[2][q] = vfmaq_laneq_f32::<2>(acc[2][q], dv[q], av);
                            acc[3][q] = vfmaq_laneq_f32::<3>(acc[3][q], dv[q], av);
                        }
                    }
                    for (l, acc_row) in acc.iter().enumerate() {
                        for (q, &value) in acc_row.iter().enumerate() {
                            vst1q_f32(g.add((p0 + l) * n + j0 + 4 * q), value);
                        }
                    }
                }
            }
        }
        let gb = grad_b.as_mut_ptr();
        for j0 in (0..n).step_by(4) {
            let mut sum = vld1q_f32(gb.add(j0));
            for r in 0..rows {
                sum = vaddq_f32(sum, vld1q_f32(d.add(r * n + j0)));
            }
            vst1q_f32(gb.add(j0), sum);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon(a: *const f32, w: *const f32, k: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut p = 0;
    while p + 8 <= k {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(p)), vld1q_f32(w.add(p)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(p + 4)), vld1q_f32(w.add(p + 4)));
        p += 8;
    }
    if p + 4 <= k {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(p)), vld1q_f32(w.add(p)));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn output_layer_train_neon(
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
    if k % 4 != 0 {
        return scalar::output_layer_train_scalar(
            a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b,
        );
    }
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let (ap, wp, dp, gp) = (
            a.as_ptr(),
            w.as_ptr(),
            dz_prev.as_mut_ptr(),
            grad_w.as_mut_ptr(),
        );
        for r in 0..rows {
            let a_row = ap.add(r * k);
            let logit = b + dot_neon(a_row, wp, k);
            let d = (1.0 / (1.0 + (-logit).exp()) - y[r]) * loss_scale;
            *grad_b += d;
            let dv = vdupq_n_f32(d);
            for p in (0..k).step_by(4) {
                let av = vld1q_f32(a_row.add(p));
                vst1q_f32(gp.add(p), vfmaq_f32(vld1q_f32(gp.add(p)), av, dv));
                let keep = vcgtq_f32(av, zero);
                let g = vmulq_f32(vld1q_f32(wp.add(p)), dv);
                vst1q_f32(
                    dp.add(r * k + p),
                    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(g), keep)),
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn output_layer_logits_neon(
    a: &[f32],
    rows: usize,
    k: usize,
    w: &[f32],
    b: f32,
    logits: &mut [f32],
) {
    if k % 4 != 0 {
        return scalar::output_layer_logits_scalar(a, rows, k, w, b, logits);
    }
    for (r, logit) in logits[..rows].iter_mut().enumerate() {
        *logit = b + unsafe { dot_neon(a.as_ptr().add(r * k), w.as_ptr(), k) };
    }
}
