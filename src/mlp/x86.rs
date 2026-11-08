//! AVX2 and AVX-512 dense-layer kernels (x86_64).
//!
//! The scheme of the NEON kernels: a register-blocked tile of `MR` rows x `NR` columns (two
//! vectors) of `C = A @ B` stays in registers for the whole reduction, and the epilogue (bias +
//! ReLU, bias, the ReLU mask of the previous layer, or accumulation into `C`) is fused into the
//! store. A is applied as one broadcast element per tile row, so it is never packed and may be
//! read with any stride: the weight gradient runs the same tile on the transposed activations.
//! Where the column count allows, AVX-512 uses a wider `WIDE_MR x 2 NR` tile (6 x 64: 24
//! accumulators, 10 loads per 24 FMAs instead of 10 per 16). Row tails use smaller tiles; column
//! counts that are not a multiple of `NR` fall back to the scalar code.
//!
//! [`dense_kernels`] writes the kernels once; each instruction set only supplies its vector type,
//! tile height and a handful of primitives. The SIMD backend registry selects an instruction set
//! only when the CPU supports it, which is what makes the `unsafe` calls below sound.

/// The kernels of one instruction set, written against the primitives in scope (`V`, `LANES`,
/// `MR`, `zero`, `load`, `store`, `splat`, `fma`, `add`, `mul`, `relu`, `keep_positive`, `hsum`).
/// `tails` are the tile heights below `MR` that cover row tails, largest first; 1 is implied.
/// `WIDE_MR` is the height of the 4-vector tile used when `n % (2 NR) == 0` (0: none).
macro_rules! dense_kernels {
    (feature = $feature:literal, tails = [$($tail:literal),*]) => {
        use crate::mlp::scalar;
        use crate::mlp::Activation;

        /// Columns of a GEMM tile (2 vectors).
        const NR: usize = 2 * LANES;
        /// Rows the weight-gradient kernel keeps in L1 while it sweeps all tiles.
        const GRAD_ROW_BLOCK: usize = 128;

        const EPILOGUE_LINEAR: u8 = 0;
        const EPILOGUE_RELU: u8 = 1;
        const EPILOGUE_RELU_MASK: u8 = 2;
        const EPILOGUE_ACCUMULATE: u8 = 3;

        /// One `M x NR` tile: `c[i][j] = epilogue(init[i][j] + sum_p a[i * a_row + p * a_red] * b[p * ldb + j])`.
        ///
        /// `init` is the bias `aux[j]` (linear, ReLU), zero (ReLU mask) or `c` itself (accumulate);
        /// the ReLU mask keeps `c[i][j]` where `aux[i * aux_ld + j] > 0`.
        #[inline]
        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn tile<const M: usize, const NV: usize, const EPILOGUE: u8>(
            a: *const f32,
            a_row: usize,
            a_red: usize,
            kk: usize,
            b: *const f32,
            ldb: usize,
            aux: *const f32,
            aux_ld: usize,
            c: *mut f32,
            ldc: usize,
        ) {
            let mut acc = [[zero(); NV]; M];
            for (i, acc_row) in acc.iter_mut().enumerate() {
                for (q, v) in acc_row.iter_mut().enumerate() {
                    *v = match EPILOGUE {
                        EPILOGUE_RELU_MASK => zero(),
                        EPILOGUE_ACCUMULATE => load(c.add(i * ldc + q * LANES)),
                        _ => load(aux.add(q * LANES)),
                    };
                }
            }
            for p in 0..kk {
                let b_row = b.add(p * ldb);
                let mut bv = [zero(); NV];
                for (q, v) in bv.iter_mut().enumerate() {
                    *v = load(b_row.add(q * LANES));
                }
                let a_col = a.add(p * a_red);
                for (i, acc_row) in acc.iter_mut().enumerate() {
                    let av = splat(*a_col.add(i * a_row));
                    for (q, v) in acc_row.iter_mut().enumerate() {
                        *v = fma(av, bv[q], *v);
                    }
                }
            }
            for (i, acc_row) in acc.iter().enumerate() {
                for (q, &value) in acc_row.iter().enumerate() {
                    let value = match EPILOGUE {
                        EPILOGUE_RELU => relu(value),
                        EPILOGUE_RELU_MASK => {
                            keep_positive(value, load(aux.add(i * aux_ld + q * LANES)))
                        }
                        _ => value,
                    };
                    store(c.add(i * ldc + q * LANES), value);
                }
            }
        }

        /// `C[m x n] = epilogue(A @ B)` over the reduction length `kk`, tiled `MR x NR`;
        /// `n % NR == 0`. A element `(i, p)` is at `a + i * a_row + p * a_red`, B row `p` at
        /// `b + p * ldb`. `aux` is the bias (`aux_ld == 0`) or the ReLU mask source (`[m x n]`,
        /// row stride `aux_ld`); the accumulate epilogue does not read it.
        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn gemm<const EPILOGUE: u8>(
            a: *const f32,
            a_row: usize,
            a_red: usize,
            m: usize,
            kk: usize,
            b: *const f32,
            ldb: usize,
            n: usize,
            aux: *const f32,
            aux_ld: usize,
            c: *mut f32,
            ldc: usize,
        ) {
            let mut i0 = 0;
            while i0 < m {
                let left = m - i0;
                let rows = if left >= MR {
                    MR
                } $(else if left >= $tail {
                    $tail
                })* else {
                    1
                };
                let (a_i, c_i, aux_i) =
                    (a.add(i0 * a_row), c.add(i0 * ldc), aux.wrapping_add(i0 * aux_ld));
                let wide = WIDE_MR > 0 && n % (2 * NR) == 0;
                let rows = if wide && left >= WIDE_MR { WIDE_MR } else { rows };
                let step = if wide && rows == WIDE_MR { 2 * NR } else { NR };
                for j0 in (0..n).step_by(step) {
                    let (b_j, aux_j, c_j) = (b.add(j0), aux_i.wrapping_add(j0), c_i.add(j0));
                    macro_rules! run {
                        ($m:expr, $nv:expr) => {
                            tile::<{ $m }, { $nv }, EPILOGUE>(
                                a_i, a_row, a_red, kk, b_j, ldb, aux_j, aux_ld, c_j, ldc,
                            )
                        };
                    }
                    match rows {
                        WIDE_MR if step == 2 * NR => run!(WIDE_MR, 4),
                        MR => run!(MR, 2),
                        $($tail => run!($tail, 2),)*
                        _ => run!(1, 2),
                    }
                }
                i0 += rows;
            }
        }

        /// `a . w` over `k` elements, `k % LANES == 0`.
        #[inline]
        #[target_feature(enable = $feature)]
        unsafe fn dot(a: *const f32, w: *const f32, k: usize) -> f32 {
            let (mut acc0, mut acc1) = (zero(), zero());
            let mut p = 0;
            while p + 2 * LANES <= k {
                acc0 = fma(load(a.add(p)), load(w.add(p)), acc0);
                acc1 = fma(load(a.add(p + LANES)), load(w.add(p + LANES)), acc1);
                p += 2 * LANES;
            }
            if p < k {
                acc0 = fma(load(a.add(p)), load(w.add(p)), acc0);
            }
            hsum(add(acc0, acc1))
        }

        /// `grad_b[n] += colsum(dz[rows x n])`, `n % LANES == 0`; up to 8 column vectors at a
        /// time stay in registers over all rows.
        #[target_feature(enable = $feature)]
        unsafe fn colsum(dz: *const f32, rows: usize, n: usize, grad_b: *mut f32) {
            const W: usize = 8;
            let mut j0 = 0;
            while j0 < n {
                let width = ((n - j0) / LANES).min(W);
                let mut acc = [zero(); W];
                for r in 0..rows {
                    let row = dz.add(r * n + j0);
                    for (q, v) in acc.iter_mut().enumerate().take(width) {
                        *v = add(*v, load(row.add(q * LANES)));
                    }
                }
                for (q, v) in acc.iter().enumerate().take(width) {
                    let g = grad_b.add(j0 + q * LANES);
                    store(g, add(load(g), *v));
                }
                j0 += width * LANES;
            }
        }

        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn output_layer_train_impl(
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
            let (ap, wp, dp, gp) =
                (a.as_ptr(), w.as_ptr(), dz_prev.as_mut_ptr(), grad_w.as_mut_ptr());
            for (r, &label) in y[..rows].iter().enumerate() {
                let a_row = ap.add(r * k);
                let logit = b + dot(a_row, wp, k);
                let d = (1.0 / (1.0 + (-logit).exp()) - label) * loss_scale;
                *grad_b += d;
                let dv = splat(d);
                for p in (0..k).step_by(LANES) {
                    let av = load(a_row.add(p));
                    store(gp.add(p), fma(av, dv, load(gp.add(p))));
                    store(dp.add(r * k + p), keep_positive(mul(load(wp.add(p)), dv), av));
                }
            }
        }

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
            if n % NR != 0 {
                return scalar::dense_forward_scalar(x, rows, k, wt, n, bias, activation, out);
            }
            let (x, wt, bias, out) = (x.as_ptr(), wt.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
            // SAFETY: the registry selects this instruction set only on CPUs that support it,
            // and the mlp entry points checked the slice lengths.
            unsafe {
                match activation {
                    Activation::Relu => {
                        gemm::<EPILOGUE_RELU>(x, k, 1, rows, k, wt, n, n, bias, 0, out, n)
                    }
                    Activation::Linear => {
                        gemm::<EPILOGUE_LINEAR>(x, k, 1, rows, k, wt, n, n, bias, 0, out, n)
                    }
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
            // reduction over the n outputs, result columns are the k inputs
            if k % NR != 0 {
                return scalar::dense_backward_input_scalar(dz, rows, n, w, k, a_prev, d_prev);
            }
            // SAFETY: as in dense_forward
            unsafe {
                gemm::<EPILOGUE_RELU_MASK>(
                    dz.as_ptr(),
                    n,
                    1,
                    rows,
                    n,
                    w.as_ptr(),
                    k,
                    k,
                    a_prev.as_ptr(),
                    k,
                    d_prev.as_mut_ptr(),
                    k,
                )
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
            if n % NR != 0 {
                return scalar::dense_weight_grad_scalar(a_prev, rows, k, dz, n, grad_wt, grad_b);
            }
            let (a, d) = (a_prev.as_ptr(), dz.as_ptr());
            // SAFETY: as in dense_forward
            unsafe {
                // tile rows are the k inputs (A = a_prev transposed), the reduction runs over rows
                for r0 in (0..rows).step_by(GRAD_ROW_BLOCK) {
                    let block = GRAD_ROW_BLOCK.min(rows - r0);
                    gemm::<EPILOGUE_ACCUMULATE>(
                        a.add(r0 * k),
                        1,
                        k,
                        k,
                        block,
                        d.add(r0 * n),
                        n,
                        n,
                        std::ptr::null(),
                        0,
                        grad_wt.as_mut_ptr(),
                        n,
                    );
                }
                colsum(d, rows, n, grad_b.as_mut_ptr());
            }
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
            if k % LANES != 0 {
                return scalar::output_layer_train_scalar(
                    a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b,
                );
            }
            // SAFETY: as in dense_forward
            unsafe {
                output_layer_train_impl(a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b)
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
            if k % LANES != 0 {
                return scalar::output_layer_logits_scalar(a, rows, k, w, b, logits);
            }
            for (r, logit) in logits[..rows].iter_mut().enumerate() {
                // SAFETY: as in dense_forward
                *logit = b + unsafe { dot(a.as_ptr().add(r * k), w.as_ptr(), k) };
            }
        }
    };
}

/// AVX2 + FMA: 8 lanes, a 6 x 16 tile in 12 of the 16 vector registers.
pub mod avx2 {
    use std::arch::x86_64::*;

    type V = __m256;
    const LANES: usize = 8;
    const MR: usize = 6;
    const WIDE_MR: usize = 0;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn zero() -> V {
        _mm256_setzero_ps()
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load(p: *const f32) -> V {
        _mm256_loadu_ps(p)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store(p: *mut f32, v: V) {
        _mm256_storeu_ps(p, v)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn splat(x: f32) -> V {
        _mm256_set1_ps(x)
    }
    /// `a * b + c`
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fma(a: V, b: V, c: V) -> V {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn add(a: V, b: V) -> V {
        _mm256_add_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mul(a: V, b: V) -> V {
        _mm256_mul_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn relu(v: V) -> V {
        _mm256_max_ps(v, _mm256_setzero_ps())
    }
    /// `v` where `source > 0`, else 0
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn keep_positive(v: V, source: V) -> V {
        _mm256_and_ps(v, _mm256_cmp_ps::<_CMP_GT_OQ>(source, _mm256_setzero_ps()))
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: V) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }

    dense_kernels!(feature = "avx2,fma", tails = [4, 2]);
}

/// AVX-512F: 16 lanes, an 8 x 32 tile in 16 of the 32 vector registers.
pub mod avx512 {
    use std::arch::x86_64::*;

    type V = __m512;
    const LANES: usize = 16;
    const MR: usize = 8;
    const WIDE_MR: usize = 6;

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn zero() -> V {
        _mm512_setzero_ps()
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(p: *const f32) -> V {
        _mm512_loadu_ps(p)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: *mut f32, v: V) {
        _mm512_storeu_ps(p, v)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(x: f32) -> V {
        _mm512_set1_ps(x)
    }
    /// `a * b + c`
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn fma(a: V, b: V, c: V) -> V {
        _mm512_fmadd_ps(a, b, c)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(a: V, b: V) -> V {
        _mm512_add_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul(a: V, b: V) -> V {
        _mm512_mul_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn relu(v: V) -> V {
        _mm512_max_ps(v, _mm512_setzero_ps())
    }
    /// `v` where `source > 0`, else 0
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn keep_positive(v: V, source: V) -> V {
        _mm512_maskz_mov_ps(
            _mm512_cmp_ps_mask::<_CMP_GT_OQ>(source, _mm512_setzero_ps()),
            v,
        )
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn hsum(v: V) -> f32 {
        _mm512_reduce_add_ps(v)
    }

    dense_kernels!(feature = "avx512f", tails = [4, 2]);
}
