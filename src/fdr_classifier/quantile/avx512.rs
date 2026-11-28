//! [`FeatureMap::apply_column`] with AVX-512 gathers. The binary searches of `LANES` values run in
//! lockstep, and the interpolation uses 8-wide vectors. Values outside the interpolated segments
//! take the scalar path.

use super::{FeatureMap, LANES};
use std::arch::x86_64::*;

/// Vectors of 8 values searched in lockstep, to hide the latency of the dependent gathers.
const VECTORS: usize = LANES / 8;

/// `FeatureMap::apply`: the binary searches as gathers of 8 knots, and the interpolation of the
/// values for which `FeatureMap::interpolates` is true. The other values take the scalar path.
///
/// # Safety
/// The CPU must support AVX-512F. Every gather index is a knot index below `quantiles.len()`,
/// which is also the length of `knot_scores` and `slopes`.
#[target_feature(enable = "avx512f")]
unsafe fn apply(map: &FeatureMap, x: [f32; LANES]) -> [f32; LANES] {
    let q = map.quantiles.as_ptr();
    let last = map.quantiles.len() - 1;
    let xv: [__m512d; VECTORS] =
        std::array::from_fn(|v| _mm512_cvtps_pd(_mm256_loadu_ps(x.as_ptr().add(8 * v))));
    let mut base = [_mm512_setzero_si512(); VECTORS];
    let mut len = last + 1;
    while len > 1 {
        let half = _mm512_set1_epi64((len / 2) as i64);
        for v in 0..VECTORS {
            let probe = _mm512_add_epi64(base[v], half);
            let qv = _mm512_i64gather_pd::<8>(probe, q.cast());
            let le = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(qv, xv[v]);
            base[v] = _mm512_mask_mov_epi64(base[v], le, probe);
        }
        len -= len / 2;
    }
    let (lo, hi) = (_mm512_set1_pd(*q), _mm512_set1_pd(*q.add(last)));
    let (first, final_) = (_mm512_set1_epi64(1), _mm512_set1_epi64(last as i64 - 3));
    let mut out = [0.0f32; LANES];
    let mut js = [0u64; LANES];
    let mut rest = 0u64;
    for v in 0..VECTORS {
        // The same condition as `FeatureMap::interpolates`.
        let j = base[v];
        let qj = _mm512_i64gather_pd::<8>(j, q.cast());
        let inner = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(xv[v], qj)
            | (_mm512_cmp_pd_mask::<_CMP_GT_OQ>(xv[v], lo)
                & _mm512_cmp_pd_mask::<_CMP_LT_OQ>(xv[v], hi)
                & _mm512_cmpge_epi64_mask(j, first)
                & _mm512_cmple_epi64_mask(j, final_));
        let ks = _mm512_i64gather_pd::<8>(j, map.knot_scores.as_ptr().cast());
        let sl = _mm512_i64gather_pd::<8>(j, map.slopes.as_ptr().cast());
        let scores = _mm512_add_pd(ks, _mm512_mul_pd(_mm512_sub_pd(xv[v], qj), sl));
        _mm256_storeu_ps(out.as_mut_ptr().add(8 * v), _mm512_cvtpd_ps(scores));
        _mm512_storeu_si512(js.as_mut_ptr().add(8 * v).cast(), base[v]);
        rest |= u64::from(!inner) << (8 * v);
    }
    while rest != 0 {
        let i = rest.trailing_zeros() as usize;
        out[i] = map.score(f64::from(x[i]), js[i] as usize);
        rest &= rest - 1;
    }
    out
}

/// The normal scores of a column, as [`FeatureMap::apply_column`].
///
/// # Safety
/// The CPU must support AVX-512F (the registry selects the AVX-512 backend only then).
pub(crate) unsafe fn apply_column(map: &FeatureMap, x: &[f32], out: &mut [f32]) {
    // SAFETY: the caller guarantees AVX-512F
    map.apply_blocks(x, out, |map, x| unsafe { apply(map, x) })
}
