//! Quantile transform to a standard normal output, matching sklearn's
//! `QuantileTransformer(output_distribution="normal")` up to float rounding (mean deviation
//! ~3e-5 on alphaDIA feature matrices).
//!
//! Fit: numpy "linear" percentiles of each feature, from all rows or a random subsample.
//! Apply: a bucket table jumps to the quantile segment of a value, and the normal score is
//! interpolated linearly between the scores of the segment's two knots. Only the two outer
//! segments, where the normal quantile function bends sharply, use the exact score
//! (sklearn's averaged forward/backward interpolation followed by the normal ppf).

use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

/// sklearn's BOUNDS_THRESHOLD: inputs this close to the outer quantiles map to the clip value.
const BOUNDS_THRESHOLD: f64 = 1e-7;
/// Buckets of the table that finds the quantile segment of a value.
const LOOKUP_BUCKETS: usize = 4096;
const TRANSFORM_CHUNK_ROWS: usize = 8192;

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
    let rows: Vec<usize> = match subsample {
        Some(n) if n < n_rows => {
            let mut rng = StdRng::seed_from_u64(seed);
            sample(&mut rng, n_rows, n).into_vec()
        }
        _ => (0..n_rows).collect(),
    };
    let references = references(n_quantiles);
    let per_feature: Vec<Vec<f64>> = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let mut values: Vec<f32> = rows.iter().map(|&r| x[r * n_features + f]).collect();
            values.sort_unstable_by(f32::total_cmp);
            percentiles_from_sorted(&values, &references)
        })
        .collect();
    Ok(per_feature.concat())
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
    out.par_chunks_mut(TRANSFORM_CHUNK_ROWS * n_features)
        .enumerate()
        .for_each(|(c, out_chunk)| {
            let input = &x[c * TRANSFORM_CHUNK_ROWS * n_features..][..out_chunk.len()];
            let rows = out_chunk.len() / n_features;
            // feature by feature, so one feature's tables stay in L1 for the whole chunk
            for (f, map) in maps.iter().enumerate() {
                for r in 0..rows {
                    out_chunk[r * n_features + f] = map.apply(input[r * n_features + f]);
                }
            }
        });
    Ok(out)
}

fn references(n_quantiles: usize) -> Vec<f64> {
    (0..n_quantiles)
        .map(|i| i as f64 / (n_quantiles - 1) as f64)
        .collect()
}

/// numpy "linear" percentiles of sorted values, made monotone like sklearn does.
fn percentiles_from_sorted(sorted: &[f32], references: &[f64]) -> Vec<f64> {
    let last = sorted.len() - 1;
    let mut out: Vec<f64> = references
        .iter()
        .map(|&r| {
            let pos = r * last as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(last);
            let frac = pos - lo as f64;
            sorted[lo] as f64 + frac * (sorted[hi] as f64 - sorted[lo] as f64)
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
    bucket_start: Vec<u32>,
    bucket_scale: f64,
    clip: (f64, f64),
    mid_rank_bounds: bool,
}

impl FeatureMap {
    fn new(quantiles: Vec<f64>, clip: (f64, f64), mid_rank_bounds: bool) -> Self {
        let (lo, hi) = (quantiles[0], quantiles[quantiles.len() - 1]);
        let bucket_scale = if hi > lo {
            LOOKUP_BUCKETS as f64 / (hi - lo)
        } else {
            0.0
        };
        let last_segment = quantiles.len() - 2;
        let bucket_start = (0..LOOKUP_BUCKETS)
            .map(|b| {
                let x = if bucket_scale > 0.0 {
                    lo + b as f64 / bucket_scale
                } else {
                    lo
                };
                (quantiles.partition_point(|&q| q <= x).saturating_sub(1)).min(last_segment) as u32
            })
            .collect();
        let mut map = FeatureMap {
            quantiles,
            knot_scores: Vec::new(),
            bucket_start,
            bucket_scale,
            clip,
            mid_rank_bounds,
        };
        map.knot_scores = (0..map.quantiles.len())
            .map(|i| map.exact(map.quantiles[i]))
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

    fn apply(&self, x: f32) -> f32 {
        let x = x as f64;
        let last = self.quantiles.len() - 1;
        if x.is_nan() {
            return f32::NAN;
        }
        if x <= self.quantiles[0] || x >= self.quantiles[last] {
            return self.exact(x) as f32;
        }
        let bucket =
            (((x - self.quantiles[0]) * self.bucket_scale) as usize).min(LOOKUP_BUCKETS - 1);
        let mut j = self.bucket_start[bucket] as usize;
        while j + 1 < last && self.quantiles[j + 1] <= x {
            j += 1;
        }
        if j == 0 || j + 2 >= last {
            return self.exact(x) as f32;
        }
        let t = segment_fraction(self.quantiles[j], self.quantiles[j + 1], x);
        (self.knot_scores[j] + t * (self.knot_scores[j + 1] - self.knot_scores[j])) as f32
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
}
