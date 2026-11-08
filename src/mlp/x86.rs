//! AVX2 and AVX-512 dense-layer kernels (x86_64).
//!
//! One register-blocked tile computes `MR` rows x `NV` vectors of `C = A @ B`; its accumulators
//! stay in registers for the whole reduction and the epilogue (bias + ReLU, bias, the ReLU mask
//! of the previous layer, or accumulation into `C`) is fused into the store. A is applied as one
//! broadcast element per tile row, so it is never packed for the forward and input-gradient
//! passes and may be read with any stride; the weight gradient packs strips of the transposed
//! activations into a stack panel and runs the same tile on it. Row tails use smaller tiles;
//! a column tail runs the full-width tile with masked loads and stores, so every shape is
//! vectorised. The tile is unrolled explicitly ([`unroll`]), not by the optimizer, which on
//! Rust 1.89 keeps larger tiles in memory.
//!
//! [`dense_kernels`] writes the kernels once; each instruction set only supplies its vector
//! type, tile shape and a handful of primitives. The SIMD backend registry selects an
//! instruction set only when the CPU supports it, which is what makes the `unsafe` calls below
//! sound. No kernel allocates: scratch space (the weight-gradient panel) lives on the stack.

/// Expands `$body` once per index `0..16`, with `$i` a `const usize` inside it. Bodies guard
/// themselves with `if $i < BOUND` on a const generic, which the compiler folds away, so a
/// loop over a tile dimension is unrolled exactly to that dimension.
macro_rules! unroll {
    ($i:ident => $body:block) => {
        unroll!(@ $i $body 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    (@ $i:ident $body:block $($n:literal)*) => {
        $({
            const $i: usize = $n;
            $body
        })*
    };
}

/// The kernels of one instruction set, written against the primitives in scope: the vector and
/// mask types `V`, `Mask`, `LANES`, the tile shapes (`MR` rows x `NV` vectors by default and for
/// the weight-gradient panel, `WIDE_MR x WIDE_NV` for layers with at least `WIDE_NV` vectors of
/// columns) and `zero`, `load`, `store`, `tail_mask`, `load_masked`, `store_masked`, `splat`,
/// `fma`, `add`, `mul`, `relu`, `keep_positive`, `pair_sum`.
macro_rules! dense_kernels {
    (feature = $feature:literal) => {
        use crate::mlp::Activation;

        /// Columns of a full tile of the default shape.
        const NC: usize = NV * LANES;
        /// Reduction steps per iteration of the tile loop, reading A from rows or from a panel.
        const UNROLL_ROWS: usize = 1;
        const UNROLL_PANEL: usize = 4;
        /// Rows the weight-gradient kernel packs and reduces per block.
        const GRAD_ROW_BLOCK: usize = 128;

        const EPILOGUE_LINEAR: u8 = 0;
        const EPILOGUE_RELU: u8 = 1;
        const EPILOGUE_RELU_MASK: u8 = 2;
        const EPILOGUE_ACCUMULATE: u8 = 3;

        /// The masks of the `W` vectors covering the first `cols` columns of a tile.
        #[inline]
        #[target_feature(enable = $feature)]
        unsafe fn column_masks<const W: usize>(cols: usize) -> [Mask; W] {
            let mut masks = [tail_mask(0); W];
            for (q, mask) in masks.iter_mut().enumerate() {
                *mask = tail_mask(cols.saturating_sub(q * LANES).min(LANES));
            }
            masks
        }

        /// One vector of a tile row: masked in a column-tail tile.
        #[inline]
        #[target_feature(enable = $feature)]
        unsafe fn load_cols<const MASKED: bool>(p: *const f32, mask: Mask) -> V {
            if MASKED {
                load_masked(p, mask)
            } else {
                load(p)
            }
        }

        #[inline]
        #[target_feature(enable = $feature)]
        unsafe fn store_cols<const MASKED: bool>(p: *mut f32, v: V, mask: Mask) {
            if MASKED {
                store_masked(p, v, mask)
            } else {
                store(p, v)
            }
        }

        /// One `M x W`-vector tile: `c[i][j] = epilogue(init[i][j] + sum_p A[i][p] * b[p * ldb + j])`.
        ///
        /// A element `(i, p)` is at `a[i * lda + p]`, or at `a[p * lda + i]` in a weight-gradient
        /// panel (`PANEL`); either way the tile reads it with one base register per row (or one
        /// for the panel) and a shared offset, so the reduction needs no address arithmetic.
        /// `init` is the bias `aux[j]` (linear, ReLU), zero (ReLU mask) or `c` itself (accumulate);
        /// the ReLU mask keeps `c[i][j]` where `aux[i * aux_ld + j] > 0`. A masked tile touches
        /// only the columns its `masks` select.
        #[inline]
        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn tile<const M: usize, const W: usize, const PANEL: bool, const MASKED: bool, const EPILOGUE: u8>(
            a: *const f32,
            lda: usize,
            kk: usize,
            b: *const f32,
            ldb: usize,
            aux: *const f32,
            aux_ld: usize,
            c: *mut f32,
            ldc: usize,
            masks: [Mask; W],
        ) {
            let mut acc = [[zero(); W]; M];
            if EPILOGUE == EPILOGUE_LINEAR || EPILOGUE == EPILOGUE_RELU {
                let mut bias = [zero(); W];
                unroll!(Q => { if Q < W {
                    bias[Q] = load_cols::<MASKED>(aux.add(Q * LANES), masks[Q]);
                }});
                acc = [bias; M];
            }
            if EPILOGUE == EPILOGUE_ACCUMULATE {
                unroll!(I => { if I < M { unroll!(Q => { if Q < W {
                    acc[I][Q] = load_cols::<MASKED>(c.add(I * ldc + Q * LANES), masks[Q]);
                }}); }});
            }
            let mut a_rows = [a; M];
            unroll!(I => { if I < M {
                a_rows[I] = if PANEL { a.add(I) } else { a.add(I * lda) };
            }});
            let a_step = if PANEL { lda } else { 1 };
            // one reduction step: W vectors of B row p against M broadcasts of A column p
            macro_rules! step {
                ($p:expr) => {{
                    let p = $p;
                    let b_row = b.add(p * ldb);
                    let mut bv = [zero(); W];
                    unroll!(Q => { if Q < W {
                        bv[Q] = load_cols::<MASKED>(b_row.add(Q * LANES), masks[Q]);
                    }});
                    unroll!(I => { if I < M {
                        let av = splat(*a_rows[I].add(p * a_step));
                        unroll!(Q => { if Q < W {
                            acc[I][Q] = fma(av, bv[Q], acc[I][Q]);
                        }});
                    }});
                }};
            }
            let unroll = if PANEL { UNROLL_PANEL } else { UNROLL_ROWS };
            let mut p = 0;
            while p + unroll <= kk {
                unroll!(U => { if U < unroll { step!(p + U) } });
                p += unroll;
            }
            while p < kk {
                step!(p);
                p += 1;
            }
            unroll!(I => { if I < M { unroll!(Q => { if Q < W {
                let value = match EPILOGUE {
                    EPILOGUE_RELU => relu(acc[I][Q]),
                    EPILOGUE_RELU_MASK => keep_positive(
                        acc[I][Q],
                        load_cols::<MASKED>(aux.add(I * aux_ld + Q * LANES), masks[Q]),
                    ),
                    _ => acc[I][Q],
                };
                store_cols::<MASKED>(c.add(I * ldc + Q * LANES), value, masks[Q]);
            }}); }});
        }

        /// `C[m x n] = epilogue(A @ B)` over the reduction length `kk`, in `TM x TV`-vector tiles
        /// (row tails of 4, 2 and 1 rows; a masked column tail of 1 or `TV` vectors). A element
        /// `(i, p)` is at `a + i * lda + p`, or `a + p * lda + i` in a `PANEL`; B row `p` at
        /// `b + p * ldb`. `aux` is the bias (`aux_ld == 0`) or the ReLU mask source (`[m x n]`,
        /// row stride `aux_ld`); the accumulate epilogue does not read it.
        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn gemm<const TM: usize, const TV: usize, const PANEL: bool, const EPILOGUE: u8>(
            a: *const f32,
            lda: usize,
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
            let nc = TV * LANES;
            let full = n - n % nc;
            let all = column_masks::<TV>(nc);
            let mut i0 = 0;
            while i0 < m {
                let left = m - i0;
                let rows = if left >= TM {
                    TM
                } else if left >= 4 {
                    4
                } else if left >= 2 {
                    2
                } else {
                    1
                };
                let a_i = a.add(if PANEL { i0 } else { i0 * lda });
                let (c_i, aux_i) = (c.add(i0 * ldc), aux.wrapping_add(i0 * aux_ld));
                macro_rules! run {
                    ($m:expr) => {{
                        for j0 in (0..full).step_by(nc) {
                            tile::<{ $m }, TV, PANEL, false, EPILOGUE>(
                                a_i, lda, kk, b.add(j0), ldb, aux_i.wrapping_add(j0),
                                aux_ld, c_i.add(j0), ldc, all,
                            )
                        }
                        let (b_t, aux_t, c_t) = (b.add(full), aux_i.wrapping_add(full), c_i.add(full));
                        if n - full > LANES {
                            tile::<{ $m }, TV, PANEL, true, EPILOGUE>(
                                a_i, lda, kk, b_t, ldb, aux_t, aux_ld, c_t, ldc,
                                column_masks(n - full),
                            )
                        } else if n > full {
                            tile::<{ $m }, 1, PANEL, true, EPILOGUE>(
                                a_i, lda, kk, b_t, ldb, aux_t, aux_ld, c_t, ldc,
                                column_masks(n - full),
                            )
                        }
                    }};
                }
                match rows {
                    4 => run!(4),
                    2 => run!(2),
                    1 => run!(1),
                    _ => run!(TM),
                }
                i0 += rows;
            }
        }

        /// The forward / input-gradient GEMM: wide tiles when there are enough columns.
        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn gemm_rows<const EPILOGUE: u8>(
            a: *const f32,
            lda: usize,
            m: usize,
            kk: usize,
            b: *const f32,
            n: usize,
            aux: *const f32,
            aux_ld: usize,
            c: *mut f32,
        ) {
            if n >= WIDE_NV * LANES {
                gemm::<WIDE_MR, WIDE_NV, false, EPILOGUE>(a, lda, m, kk, b, n, n, aux, aux_ld, c, n)
            } else {
                gemm::<MR, NV, false, EPILOGUE>(a, lda, m, kk, b, n, n, aux, aux_ld, c, n)
            }
        }

        /// `grad_b[n] += colsum(dz[rows x n])`, `NV` vectors at a time, four rows per step into
        /// separate accumulators.
        #[target_feature(enable = $feature)]
        unsafe fn colsum(dz: *const f32, rows: usize, n: usize, grad_b: *mut f32) {
            let full = n - n % NC;
            for j0 in (0..full).step_by(NC) {
                colsum_tile::<4, NV, false>(dz.add(j0), rows, n, grad_b.add(j0), column_masks(NC));
            }
            if full < n {
                let masks = column_masks(n - full);
                colsum_tile::<4, NV, true>(dz.add(full), rows, n, grad_b.add(full), masks);
            }
        }

        /// One `W`-vector column strip of [`colsum`], `R` rows per step.
        #[inline]
        #[target_feature(enable = $feature)]
        unsafe fn colsum_tile<const R: usize, const W: usize, const MASKED: bool>(
            dz: *const f32,
            rows: usize,
            ld: usize,
            grad_b: *mut f32,
            masks: [Mask; W],
        ) {
            let mut acc = [[zero(); W]; R];
            let mut r = 0;
            while r + R <= rows {
                unroll!(U => { if U < R { unroll!(Q => { if Q < W {
                    let v = load_cols::<MASKED>(dz.add((r + U) * ld + Q * LANES), masks[Q]);
                    acc[U][Q] = add(acc[U][Q], v);
                }}); }});
                r += R;
            }
            while r < rows {
                unroll!(Q => { if Q < W {
                    let v = load_cols::<MASKED>(dz.add(r * ld + Q * LANES), masks[Q]);
                    acc[0][Q] = add(acc[0][Q], v);
                }});
                r += 1;
            }
            unroll!(Q => { if Q < W {
                let g = grad_b.add(Q * LANES);
                let mut sum = load_cols::<MASKED>(g, masks[Q]);
                unroll!(U => { if U < R {
                    sum = add(sum, acc[U][Q]);
                }});
                store_cols::<MASKED>(g, sum, masks[Q]);
            }});
        }

        /// `grad_wt[k x n] += a[rows x k]^T @ dz[rows x n]`, `rows <= GRAD_ROW_BLOCK`. Each strip
        /// of `MR` inputs is copied into a contiguous stack panel (`[rows x MR]`), so the tile
        /// broadcasts A from one unit-stride stream instead of `MR` columns `k` floats apart.
        #[target_feature(enable = $feature)]
        unsafe fn weight_grad_block(
            a: *const f32,
            rows: usize,
            k: usize,
            dz: *const f32,
            n: usize,
            grad_wt: *mut f32,
        ) {
            let mut panel = [0.0f32; GRAD_ROW_BLOCK * MR];
            let panel = panel.as_mut_ptr();
            for i0 in (0..k).step_by(MR) {
                let strip = MR.min(k - i0);
                if strip == MR {
                    for r in 0..rows {
                        std::ptr::copy_nonoverlapping(a.add(r * k + i0), panel.add(r * MR), MR);
                    }
                } else {
                    for r in 0..rows {
                        for i in 0..strip {
                            *panel.add(r * MR + i) = *a.add(r * k + i0 + i);
                        }
                    }
                }
                gemm::<MR, NV, true, EPILOGUE_ACCUMULATE>(
                    panel,
                    MR,
                    strip,
                    rows,
                    dz,
                    n,
                    n,
                    std::ptr::null(),
                    0,
                    grad_wt.add(i0 * n),
                    n,
                );
            }
        }

        /// `out[i] = bias + a[i] . w` for `count` rows of `a` (row stride `k`); `L == LANES`
        /// (generic only so that the unrolled indexing is not checked against a fixed length).
        /// Each row is reduced into its own vector, then one tree of `pair_sum`s turns the
        /// `L` vectors into the `L` dot products.
        #[target_feature(enable = $feature)]
        unsafe fn row_logits<const FULL: bool, const L: usize>(
            a: *const f32,
            count: usize,
            w: *const f32,
            k: usize,
            bias: f32,
            out: *mut f32,
        ) {
            let mut acc = [zero(); L];
            let full = k - k % LANES;
            let tail = tail_mask(k - full);
            unroll!(I => { if I < L && (FULL || I < count) {
                let row = a.add(I * k);
                let mut p = 0;
                while p < full {
                    acc[I] = fma(load(row.add(p)), load(w.add(p)), acc[I]);
                    p += LANES;
                }
                if full < k {
                    let (av, wv) = (load_masked(row.add(full), tail), load_masked(w.add(full), tail));
                    acc[I] = fma(av, wv, acc[I]);
                }
            }});
            let mut len = L;
            while len > 1 {
                len /= 2;
                for j in 0..len {
                    acc[j] = pair_sum(acc[2 * j], acc[2 * j + 1]);
                }
            }
            let logits = add(splat(bias), acc[0]);
            if FULL {
                store(out, logits)
            } else {
                store_masked(out, logits, tail_mask(count))
            }
        }

        #[target_feature(enable = $feature)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn output_layer_train_impl(
            a: *const f32,
            rows: usize,
            k: usize,
            w: *const f32,
            b: f32,
            y: *const f32,
            loss_scale: f32,
            dz_prev: *mut f32,
            grad_w: *mut f32,
            grad_b: &mut f32,
        ) -> f32 {
            let mut loss = 0.0f32;
            let full = k - k % LANES;
            let tail = tail_mask(k - full);
            for r0 in (0..rows).step_by(LANES) {
                let count = LANES.min(rows - r0);
                let a_blk = a.add(r0 * k);
                let mut logits = [0.0f32; LANES];
                if count == LANES {
                    row_logits::<true, LANES>(a_blk, count, w, k, b, logits.as_mut_ptr())
                } else {
                    row_logits::<false, LANES>(a_blk, count, w, k, b, logits.as_mut_ptr())
                }
                let mut d = [0.0f32; LANES];
                for i in 0..count {
                    let (logit, label) = (logits[i], *y.add(r0 + i));
                    // one exp for both the stable loss and the sigmoid
                    let e = (-logit.abs()).exp();
                    loss += logit.max(0.0) - logit * label + e.ln_1p();
                    let sigmoid = if logit >= 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) };
                    d[i] = (sigmoid - label) * loss_scale;
                    *grad_b += d[i];
                }
                // grad_w += sum_i d[i] a[i], dz_prev[i] = d[i] w (a[i] > 0), per vector of k
                let mut p = 0;
                while p < k {
                    let mask = if p < full { tail_mask(LANES) } else { tail };
                    let wv = load_masked(w.add(p), mask);
                    // two accumulators for the rows in turn, to halve the FMA chain
                    let (mut g0, mut g1) = (load_masked(grad_w.add(p), mask), zero());
                    macro_rules! row {
                        ($i:expr, $g:ident) => {{
                            let i = $i;
                            let (av, dv) = (load_masked(a_blk.add(i * k + p), mask), splat(d[i]));
                            $g = fma(av, dv, $g);
                            store_masked(
                                dz_prev.add((r0 + i) * k + p),
                                keep_positive(mul(wv, dv), av),
                                mask,
                            );
                        }};
                    }
                    let mut i = 0;
                    while i + 2 <= count {
                        row!(i, g0);
                        row!(i + 1, g1);
                        i += 2;
                    }
                    if i < count {
                        row!(i, g0);
                    }
                    store_masked(grad_w.add(p), add(g0, g1), mask);
                    p += LANES;
                }
            }
            loss
        }

        #[target_feature(enable = $feature)]
        unsafe fn output_layer_logits_impl(
            a: *const f32,
            rows: usize,
            k: usize,
            w: *const f32,
            b: f32,
            logits: *mut f32,
        ) {
            let mut r0 = 0;
            while r0 + LANES <= rows {
                row_logits::<true, LANES>(a.add(r0 * k), LANES, w, k, b, logits.add(r0));
                r0 += LANES;
            }
            if r0 < rows {
                row_logits::<false, LANES>(a.add(r0 * k), rows - r0, w, k, b, logits.add(r0));
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
            let (x, wt, bias, out) = (x.as_ptr(), wt.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
            // SAFETY: the registry selects this instruction set only on CPUs that support it,
            // and the mlp entry points checked the slice lengths.
            unsafe {
                match activation {
                    Activation::Relu => {
                        gemm_rows::<EPILOGUE_RELU>(x, k, rows, k, wt, n, bias, 0, out)
                    }
                    Activation::Linear => {
                        gemm_rows::<EPILOGUE_LINEAR>(x, k, rows, k, wt, n, bias, 0, out)
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
            // SAFETY: as in dense_forward
            unsafe {
                gemm_rows::<EPILOGUE_RELU_MASK>(
                    dz.as_ptr(),
                    n,
                    rows,
                    n,
                    w.as_ptr(),
                    k,
                    a_prev.as_ptr(),
                    k,
                    d_prev.as_mut_ptr(),
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
            let (a, d) = (a_prev.as_ptr(), dz.as_ptr());
            // SAFETY: as in dense_forward
            unsafe {
                for r0 in (0..rows).step_by(GRAD_ROW_BLOCK) {
                    let block = GRAD_ROW_BLOCK.min(rows - r0);
                    weight_grad_block(
                        a.add(r0 * k),
                        block,
                        k,
                        d.add(r0 * n),
                        n,
                        grad_wt.as_mut_ptr(),
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
        ) -> f32 {
            assert!(y.len() >= rows, "y has {} elements, expected at least {rows}", y.len());
            // SAFETY: as in dense_forward
            unsafe {
                output_layer_train_impl(
                    a.as_ptr(),
                    rows,
                    k,
                    w.as_ptr(),
                    b,
                    y.as_ptr(),
                    loss_scale,
                    dz_prev.as_mut_ptr(),
                    grad_w.as_mut_ptr(),
                    grad_b,
                )
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
            // SAFETY: as in dense_forward
            unsafe { output_layer_logits_impl(a.as_ptr(), rows, k, w.as_ptr(), b, logits.as_mut_ptr()) }
        }
    };
}

/// AVX2 + FMA: 8 lanes, a 6 x 16 tile in 12 of the 16 vector registers.
pub mod avx2 {
    use std::arch::x86_64::*;

    type V = __m256;
    /// Lanes selected by a mask have their sign bit set.
    type Mask = __m256i;
    const LANES: usize = 8;
    const MR: usize = 6;
    const NV: usize = 2;
    const WIDE_MR: usize = MR;
    const WIDE_NV: usize = NV;

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
    /// The first `count <= LANES` lanes
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_mask(count: usize) -> Mask {
        _mm256_cmpgt_epi32(
            _mm256_set1_epi32(count as i32),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        )
    }
    /// Unselected lanes are zero and not read.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load_masked(p: *const f32, mask: Mask) -> V {
        _mm256_maskload_ps(p, mask)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store_masked(p: *mut f32, v: V, mask: Mask) {
        _mm256_maskstore_ps(p, mask, v)
    }
    /// `[x0 + x1, x2 + x3, .., y0 + y1, y2 + y3, ..]`
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pair_sum(x: V, y: V) -> V {
        // hadd interleaves the 128-bit halves: [x01 x23 y01 y23 | x45 x67 y45 y67]
        let h = _mm256_castps_pd(_mm256_hadd_ps(x, y));
        _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(h))
    }

    dense_kernels!(feature = "avx2,fma");
}

/// AVX-512F: 16 lanes, an 8 x 32 tile in 16 of the 32 vector registers.
pub mod avx512 {
    use std::arch::x86_64::*;

    type V = __m512;
    type Mask = __mmask16;
    const LANES: usize = 16;
    const MR: usize = 12;
    const NV: usize = 2;
    const WIDE_MR: usize = 6;
    const WIDE_NV: usize = 4;

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
    /// The first `count <= LANES` lanes
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn tail_mask(count: usize) -> Mask {
        ((1u32 << count) - 1) as Mask
    }
    /// Unselected lanes are zero and not read.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_masked(p: *const f32, mask: Mask) -> V {
        _mm512_maskz_loadu_ps(mask, p)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_masked(p: *mut f32, v: V, mask: Mask) {
        _mm512_mask_storeu_ps(p, mask, v)
    }
    /// `[x0 + x1, x2 + x3, .., y0 + y1, y2 + y3, ..]`
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn pair_sum(x: V, y: V) -> V {
        let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        _mm512_add_ps(
            _mm512_permutex2var_ps(x, even, y),
            _mm512_permutex2var_ps(x, odd, y),
        )
    }

    dense_kernels!(feature = "avx512f");
}
