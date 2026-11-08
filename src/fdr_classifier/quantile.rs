//! Quantile transform to a standard normal output, matching sklearn's
//! `QuantileTransformer(output_distribution="normal")` up to float rounding (mean deviation
//! ~3e-5 on alphaDIA feature matrices).
//!
//! Fit: numpy "linear" percentiles of each feature, from all rows or a random subsample.
//! Apply: a branchless binary search finds the quantile segment of a value, and the normal score
//! is interpolated linearly between the scores of the segment's two knots. Only the two outer
//! segments, where the normal quantile function bends sharply, use the exact score
//! (sklearn's averaged forward/backward interpolation followed by the normal ppf).

use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

/// sklearn's BOUNDS_THRESHOLD: inputs this close to the outer quantiles map to the clip value.
const BOUNDS_THRESHOLD: f64 = 1e-7;
/// Rows per task: a chunk (160 KB at 40 features) and its transposes stay in L2. Not a power of
/// two, so that the columns of a transposed chunk do not all map to the same L1 cache sets.
const TRANSFORM_CHUNK_ROWS: usize = 1000;
/// Values searched in lockstep: the steps of one binary search depend on each other, so it is the
/// independent searches that keep the core busy.
const LANES: usize = 32;
/// Lockstep searches of the scalar code, as many as its index registers hold.
const SCALAR_LANES: usize = 8;

/// Quantiles `[n_features x n_quantiles]`, row-major, at the evenly spaced references 0..1.
pub fn fit(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    n_quantiles: usize,
    subsample: Option<usize>,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if n_rows == 0 || n_features == 0 {
        return Err("cannot fit a quantile transform on an empty matrix".to_string());
    }
    if n_quantiles < 2 {
        return Err(format!("n_quantiles must be at least 2, got {n_quantiles}"));
    }
    if x.len() != n_rows * n_features {
        return Err(format!(
            "x has {} values, expected {n_rows} x {n_features}",
            x.len()
        ));
    }
    if subsample == Some(0) {
        return Err("the quantile subsample needs at least one row".to_string());
    }
    let rows = match subsample {
        Some(n) if n < n_rows => {
            let mut rng = StdRng::seed_from_u64(seed);
            Some(sample(&mut rng, n_rows, n).into_vec())
        }
        _ => None,
    };
    let rows = rows.as_deref();
    let n = rows.map_or(n_rows, <[usize]>::len);
    let references = references(n_quantiles);
    // a group of features at a time: one parallel sweep over the rows extracts their columns
    // (reading the matrix once per group rather than once per feature), then each is sorted
    let group = rayon::current_num_threads().clamp(1, n_features);
    let mut quantiles = Vec::with_capacity(n_features * n_quantiles);
    for first in (0..n_features).step_by(group) {
        let mut columns: Vec<Vec<u32>> = (first..(first + group).min(n_features))
            .map(|_| vec![0; n])
            .collect();
        // the same row block of every column
        let mut blocks: Vec<Vec<&mut [u32]>> = (0..n.div_ceil(TRANSFORM_CHUNK_ROWS))
            .map(|_| Vec::with_capacity(group))
            .collect();
        for column in &mut columns {
            for (block, piece) in blocks.iter_mut().zip(column.chunks_mut(TRANSFORM_CHUNK_ROWS)) {
                block.push(piece);
            }
        }
        blocks.into_par_iter().enumerate().for_each(|(b, mut block)| {
            for i in 0..block[0].len() {
                let r = b * TRANSFORM_CHUNK_ROWS + i;
                let row = &x[rows.map_or(r, |rows| rows[r]) * n_features + first..];
                for (column, &v) in block.iter_mut().zip(row) {
                    column[i] = order_key(v);
                }
            }
        });
        let per_feature: Vec<Vec<f64>> = columns
            .into_par_iter()
            .map(|keys| percentiles_from_sorted(&radix_sort(keys), &references))
            .collect();
        quantiles.extend(per_feature.into_iter().flatten());
    }
    Ok(quantiles)
}

/// Normal scores of `x` under the fitted `quantiles`.
///
/// `mid_rank_bounds` maps values at the lowest or highest quantile to the middle of their tie
/// group, as every other tie group is mapped. sklearn instead clamps values within
/// `BOUNDS_THRESHOLD` of the outer quantiles to the clip value (about -5.2 / +5.2), which puts a
/// feature that is 0 in most rows almost entirely at -5.2; and in float32 the threshold vanishes
/// for values of magnitude above about 1, so whether a tie group is clamped depends on its value.
pub fn transform(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    quantiles: &[f64],
    n_quantiles: usize,
    mid_rank_bounds: bool,
) -> Result<Vec<f32>, String> {
    if quantiles.len() != n_features * n_quantiles {
        return Err(format!(
            "quantiles have {} values, expected {n_features} features x {n_quantiles}",
            quantiles.len()
        ));
    }
    if x.len() != n_rows * n_features {
        return Err(format!(
            "x has {} values, expected {n_rows} x {n_features}",
            x.len()
        ));
    }
    let clip = clip_bounds();
    let maps: Vec<FeatureMap> = quantiles
        .chunks_exact(n_quantiles)
        .map(|q| FeatureMap::new(q.to_vec(), clip, mid_rank_bounds))
        .collect();
    let mut out = vec![0.0f32; x.len()];
    let chunk_len = TRANSFORM_CHUNK_ROWS * n_features;
    out.par_chunks_mut(chunk_len)
        .zip(x.par_chunks(chunk_len))
        .for_each_init(
            || (vec![0.0f32; chunk_len], vec![0.0f32; chunk_len]),
            |(columns, scores), (out_chunk, input)| {
                let rows = input.len() / n_features;
                // feature by feature on contiguous columns, so one feature's tables stay in L1
                transpose(input, rows, n_features, columns);
                for (f, map) in maps.iter().enumerate() {
                    let column = f * rows..(f + 1) * rows;
                    map.apply_column(&columns[column.clone()], &mut scores[column]);
                }
                transpose(&scores[..input.len()], n_features, rows, out_chunk);
            },
        );
    Ok(out)
}

/// `out[cols x rows] = a[rows x cols]^T`
fn transpose(a: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for (r, row) in a.chunks_exact(cols).take(rows).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[c * rows + r] = v;
        }
    }
}

fn references(n_quantiles: usize) -> Vec<f64> {
    (0..n_quantiles)
        .map(|i| i as f64 / (n_quantiles - 1) as f64)
        .collect()
}

/// An integer key of `v` in `f32::total_cmp` order: the bits of a positive float are in order,
/// those of a negative one in reverse order.
fn order_key(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}

fn from_order_key(key: u32) -> f32 {
    f32::from_bits(if key >> 31 == 1 {
        key & !(1 << 31)
    } else {
        !key
    })
}

/// LSD radix sort, several times faster than a comparison sort on the millions of keys of a
/// feature.
fn radix_sort(mut keys: Vec<u32>) -> Vec<u32> {
    const BITS: usize = 11;
    const BUCKETS: usize = 1 << BITS;
    const PASSES: usize = 32usize.div_ceil(BITS);
    let digit = |key: u32, pass: usize| (key >> (pass * BITS)) as usize & (BUCKETS - 1);
    let mut counts = vec![[0usize; BUCKETS]; PASSES];
    for &key in &keys {
        for (pass, counts) in counts.iter_mut().enumerate() {
            counts[digit(key, pass)] += 1;
        }
    }
    let mut sorted = vec![0u32; keys.len()];
    for (pass, counts) in counts.iter_mut().enumerate() {
        if counts.contains(&keys.len()) {
            continue; // all keys share this digit
        }
        let mut offset = 0;
        for count in counts.iter_mut() {
            (*count, offset) = (offset, offset + *count);
        }
        for &key in &keys {
            let slot = &mut counts[digit(key, pass)];
            sorted[*slot] = key;
            *slot += 1;
        }
        std::mem::swap(&mut keys, &mut sorted);
    }
    keys
}

/// numpy "linear" percentiles of sorted values (as `order_key`s), made monotone like sklearn does.
fn percentiles_from_sorted(sorted: &[u32], references: &[f64]) -> Vec<f64> {
    let last = sorted.len() - 1;
    let value = |i: usize| f64::from(from_order_key(sorted[i]));
    let mut out: Vec<f64> = references
        .iter()
        .map(|&r| {
            let pos = r * last as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(last);
            let frac = pos - lo as f64;
            value(lo) + frac * (value(hi) - value(lo))
        })
        .collect();
    for i in 1..out.len() {
        out[i] = out[i].max(out[i - 1]);
    }
    out
}

/// Inverse standard normal CDF, Wichura's AS241 (PPND16), relative error ~1e-16.
/// The coefficients are copied as published.
#[allow(clippy::inconsistent_digit_grouping, clippy::excessive_precision)]
pub(crate) fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((2509.080_928_730_122_7 * r + 33_430.575_583_588_13) * r
                + 67_265.770_927_008_7)
                * r
                + 45_921.953_931_549_87)
                * r
                + 13_731.693_765_509_46)
                * r
                + 1971.590_950_306_551_4)
                * r
                + 133.141_667_891_784_38)
                * r
                + 3.387_132_872_796_366_6)
            / (((((((5226.495_278_852_546 * r + 28_729.085_735_721_94) * r
                + 39_307.895_800_092_71)
                * r
                + 21_213.794_301_586_6)
                * r
                + 5394.196_021_424_751)
                * r
                + 687.187_007_492_057_9)
                * r
                + 42.313_330_701_600_91)
                * r
                + 1.0);
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        (((((((7.745_450_142_783_414e-4 * r + 0.022_723_844_989_269_184) * r
            + 0.241_780_725_177_450_6)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_5)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_545)
            * r
            + 1.423_437_110_749_683_5)
            / (((((((1.050_750_071_644_416_8e-9 * r + 5.475_938_084_995_345e-4) * r
                + 0.015_198_666_563_616_457)
                * r
                + 0.148_103_976_427_480_08)
                * r
                + 0.689_767_334_985_1)
                * r
                + 1.676_384_830_183_803_8)
                * r
                + 2.053_191_626_637_759)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 0.001_242_660_947_388_078_4)
            * r
            + 0.026_532_189_526_576_124)
            * r
            + 0.296_560_571_828_504_9)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103)
            / (((((((2.044_263_103_389_939_7e-15 * r + 1.421_511_758_316_446e-7) * r
                + 1.846_318_317_510_054_8e-5)
                * r
                + 7.868_691_311_456_133e-4)
                * r
                + 0.014_875_361_290_850_615)
                * r
                + 0.136_929_880_922_735_8)
                * r
                + 0.599_832_206_555_888)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -value
    } else {
        value
    }
}

/// sklearn clips the normal scores at ppf(BOUNDS_THRESHOLD - spacing(1)) and its mirror.
fn clip_bounds() -> (f64, f64) {
    let eps = f64::EPSILON;
    (
        norm_ppf(BOUNDS_THRESHOLD - eps),
        norm_ppf(1.0 - (BOUNDS_THRESHOLD - eps)),
    )
}

struct FeatureMap {
    quantiles: Vec<f64>,
    knot_scores: Vec<f64>,
    /// score per unit of input along the segment from each knot (0 for an empty segment and for
    /// the last knot)
    slopes: Vec<f64>,
    clip: (f64, f64),
    mid_rank_bounds: bool,
    /// search and interpolate with AVX-512 gathers (the SIMD backend registry selected AVX-512)
    #[cfg(target_arch = "x86_64")]
    avx512: bool,
}

impl FeatureMap {
    fn new(quantiles: Vec<f64>, clip: (f64, f64), mid_rank_bounds: bool) -> Self {
        let mut map = FeatureMap {
            quantiles,
            knot_scores: Vec::new(),
            slopes: Vec::new(),
            clip,
            mid_rank_bounds,
            #[cfg(target_arch = "x86_64")]
            avx512: crate::simd::get_backend().priority() == crate::simd::Rank::Avx512,
        };
        map.knot_scores = map.quantiles.iter().map(|&q| map.exact(q)).collect();
        map.slopes = (0..map.quantiles.len())
            .map(|j| {
                let width = map.quantiles.get(j + 1).map_or(0.0, |&q| q - map.quantiles[j]);
                if width > 0.0 {
                    (map.knot_scores[j + 1] - map.knot_scores[j]) / width
                } else {
                    0.0
                }
            })
            .collect();
        map
    }

    /// sklearn `_transform_col` for output_distribution="normal" (with `mid_rank_bounds`, without
    /// its clamp at the outer quantiles: a value below all quantiles still gets rank 0).
    fn exact(&self, x: f64) -> f64 {
        let n = self.quantiles.len() - 1;
        let p = if self.mid_rank_bounds {
            0.5 * (self.forward_rank(x) + self.backward_rank(x)) / n as f64
        } else if x - BOUNDS_THRESHOLD < self.quantiles[0] {
            0.0
        } else if x + BOUNDS_THRESHOLD > self.quantiles[n] {
            1.0
        } else {
            0.5 * (self.forward_rank(x) + self.backward_rank(x)) / n as f64
        };
        norm_ppf(p).clamp(self.clip.0, self.clip.1)
    }

    /// np.interp(x, quantiles, 0..n): rank through the rightmost knot at or below x.
    fn forward_rank(&self, x: f64) -> f64 {
        let last_segment = self.quantiles.len() - 2;
        if x >= self.quantiles[last_segment + 1] {
            // at or above the highest knot, as np.interp: the last rank, also for a tie group that
            // ends at the highest knot (otherwise the segment clamp gives the rank before it)
            return (last_segment + 1) as f64;
        }
        let j = self
            .quantiles
            .partition_point(|&q| q <= x)
            .saturating_sub(1)
            .min(last_segment);
        j as f64 + segment_fraction(self.quantiles[j], self.quantiles[j + 1], x)
    }

    /// The mirrored interpolation of sklearn: rank through the leftmost knot at or above x.
    fn backward_rank(&self, x: f64) -> f64 {
        let n = self.quantiles.len() - 1;
        let j = self.quantiles.partition_point(|&q| q < x).clamp(1, n);
        j as f64 - segment_fraction(self.quantiles[j - 1], self.quantiles[j], x).mul_add(-1.0, 1.0)
    }

    /// Index of the last knot at or below each value (0 below all knots or for NaN): binary
    /// searches of the same fixed length, whose steps compile to conditional moves as the values
    /// come in random order.
    fn segments<const N: usize>(&self, x: &[f64; N]) -> [usize; N] {
        let q = &self.quantiles;
        let (mut base, mut len) = ([0usize; N], q.len());
        while len > 1 {
            let half = len / 2;
            for (b, &x) in base.iter_mut().zip(x) {
                *b = std::hint::select_unpredictable(q[*b + half] <= x, *b + half, *b);
            }
            len -= half;
        }
        base
    }

    fn apply_column(&self, x: &[f32], out: &mut [f32]) {
        let mut x_blocks = x.chunks_exact(LANES);
        let mut out_blocks = out.chunks_exact_mut(LANES);
        for (x, out) in (&mut x_blocks).zip(&mut out_blocks) {
            out.copy_from_slice(&self.apply(x.try_into().unwrap()));
        }
        let (x, out) = (x_blocks.remainder(), out_blocks.into_remainder());
        if !x.is_empty() {
            let mut padded = [0.0f32; LANES];
            padded[..x.len()].copy_from_slice(x);
            out.copy_from_slice(&self.apply(padded)[..x.len()]);
        }
    }

    fn apply(&self, x: [f32; LANES]) -> [f32; LANES] {
        #[cfg(target_arch = "x86_64")]
        if self.avx512 {
            // SAFETY: `avx512` is set only when the CPU supports AVX-512F
            return unsafe { avx512::apply(self, x) };
        }
        let mut out = [0.0f32; LANES];
        for (x, out) in x.chunks_exact(SCALAR_LANES).zip(out.chunks_exact_mut(SCALAR_LANES)) {
            let x: [f64; SCALAR_LANES] = std::array::from_fn(|i| f64::from(x[i]));
            let segments = self.segments(&x);
            for ((out, &x), &j) in out.iter_mut().zip(&x).zip(&segments) {
                *out = self.score(x, j);
            }
        }
        out
    }

    /// The normal score of `x` in segment `j` (from `segments`).
    fn score(&self, x: f64, j: usize) -> f32 {
        if self.interpolates(x, j) {
            return (self.knot_scores[j] + (x - self.quantiles[j]) * self.slopes[j]) as f32;
        }
        if x.is_nan() {
            return f32::NAN;
        }
        self.exact(x) as f32
    }

    /// Whether the score of `x` in segment `j` is interpolated: at a knot (a tie group, often a
    /// feature's 0 in most rows, has the knot's exact score) and in the inner segments; the outer
    /// segments, where the normal quantile function bends sharply, are exact.
    fn interpolates(&self, x: f64, j: usize) -> bool {
        let q = &self.quantiles;
        let last = q.len() - 1;
        x == q[j] || (x > q[0] && x < q[last] && j > 0 && j + 2 < last)
    }
}

fn segment_fraction(q0: f64, q1: f64, x: f64) -> f64 {
    if q1 > q0 {
        ((x - q0) / (q1 - q0)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
pub(crate) mod test_access {
    pub fn exact_scores(x: &[f32], quantiles: &[f64]) -> Vec<f32> {
        let map = super::FeatureMap::new(quantiles.to_vec(), super::clip_bounds(), false);
        x.iter().map(|&v| map.exact(v as f64) as f32).collect()
    }

    pub fn radix_sort(x: &[f32]) -> Vec<f32> {
        let keys = x.iter().map(|&v| super::order_key(v)).collect();
        super::radix_sort(keys)
            .into_iter()
            .map(super::from_order_key)
            .collect()
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{FeatureMap, LANES};
    use std::arch::x86_64::*;

    /// Vectors of 8 values searched in lockstep, to hide the latency of the dependent gathers.
    const VECTORS: usize = LANES / 8;

    /// `FeatureMap::apply`: the binary searches as gathers of 8 knots, and the interpolation of the
    /// values for which `FeatureMap::interpolates` holds; the others take the scalar path.
    ///
    /// # Safety
    /// The CPU must support AVX-512F. Every gather index is a knot index below `quantiles.len()`,
    /// which is also the length of `knot_scores` and `slopes`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn apply(map: &FeatureMap, x: [f32; LANES]) -> [f32; LANES] {
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
            // as `FeatureMap::interpolates`
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
}
