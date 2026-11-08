//! AVX-512F: 16 lanes; 12 x 32 tiles (24 accumulators), 6 x 64 tiles for layers with at least
//! 64 outputs (24 accumulators, fewer B loads and broadcasts per FMA).
//!
//! The kernels themselves are the shared template [`super::x86`]; this file supplies the
//! vector type, tile shapes and primitives.

use std::arch::x86_64::*;

type V = __m512;
type Mask = __mmask16;
const LANES: usize = 16;
const MR: usize = 12;
const NV: usize = 2;
const WIDE_MR: usize = 6;
const WIDE_NV: usize = 4;
/// Reduction steps per tile loop iteration: reading A from rows (unrolling makes LLVM hoist
/// all broadcasts of a 12-row tile and spill them), or from a panel.
const UNROLL_ROWS: usize = 1;
const UNROLL_PANEL: usize = 4;

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
unsafe fn min(a: V, b: V) -> V {
    _mm512_min_ps(a, b)
}
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn max(a: V, b: V) -> V {
    _mm512_max_ps(a, b)
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
/// `v` in the lanes `mask` selects, else 0
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn select(v: V, mask: Mask) -> V {
    _mm512_maskz_mov_ps(mask, v)
}
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn div(a: V, b: V) -> V {
    _mm512_div_ps(a, b)
}
/// to the nearest integer
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn round(v: V) -> V {
    _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v)
}
/// `2^n` for integral `-126 <= n <= 127`
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn pow2i(n: V) -> V {
    _mm512_scalef_ps(_mm512_set1_ps(1.0), n)
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

dense_kernels!(
    feature = "avx512f",
    dense_forward_avx512,
    dense_backward_input_avx512,
    dense_weight_grad_avx512,
    output_layer_train_avx512,
    output_layer_logits_avx512,
);
