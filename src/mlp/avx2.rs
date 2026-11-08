//! AVX2 + FMA: 8 lanes, a 6 x 16 tile in 12 of the 16 vector registers.
//!
//! The kernels themselves are the shared template [`super::x86`]; this file supplies the
//! vector type, tile shapes and primitives.

use std::arch::x86_64::*;

type V = __m256;
/// Lanes selected by a mask have their sign bit set.
type Mask = __m256i;
const LANES: usize = 8;
const MR: usize = 6;
const NV: usize = 2;
const WIDE_MR: usize = MR;
const WIDE_NV: usize = NV;
/// Reduction steps per tile loop iteration: reading A from rows, or from a panel (with 16
/// registers, unrolling either makes LLVM hoist the broadcasts and spill them).
const UNROLL_ROWS: usize = 1;
const UNROLL_PANEL: usize = 1;

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
unsafe fn min(a: V, b: V) -> V {
    _mm256_min_ps(a, b)
}
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn max(a: V, b: V) -> V {
    _mm256_max_ps(a, b)
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
/// `v` in the lanes `mask` selects, else 0
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn select(v: V, mask: Mask) -> V {
    _mm256_and_ps(v, _mm256_castsi256_ps(mask))
}
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn div(a: V, b: V) -> V {
    _mm256_div_ps(a, b)
}
/// to the nearest integer
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn round(v: V) -> V {
    _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v)
}
/// `2^n` for integral `-126 <= n <= 127`
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pow2i(n: V) -> V {
    let biased = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
    _mm256_castsi256_ps(_mm256_slli_epi32::<23>(biased))
}
/// `[x0 + x1, x2 + x3, .., y0 + y1, y2 + y3, ..]`
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_sum(x: V, y: V) -> V {
    // hadd interleaves the 128-bit halves: [x01 x23 y01 y23 | x45 x67 y45 y67]
    let h = _mm256_castps_pd(_mm256_hadd_ps(x, y));
    _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(h))
}

dense_kernels!(
    feature = "avx2,fma",
    dense_forward_avx2,
    dense_backward_input_avx2,
    dense_weight_grad_avx2,
    output_layer_train_avx2,
    output_layer_logits_avx2,
);
