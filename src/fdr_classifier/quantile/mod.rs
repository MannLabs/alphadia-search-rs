//! Quantile transform to a standard normal output, matching sklearn's
//! `QuantileTransformer(output_distribution="normal")` up to float rounding (mean deviation
//! ~3e-5 on alphaDIA feature matrices).
//!
//! Fit: numpy "linear" percentiles of each feature, from all rows or a random subsample.
//!
//! Apply: a branchless binary search finds the quantile segment of a value. The normal score is
//! the linear interpolation between the scores of the two knots of the segment. The first segment
//! and the last two segments, where the normal quantile function bends sharply, use the exact
//! score (sklearn's averaged forward/backward interpolation followed by the normal ppf). Values
//! outside the fitted quantiles also use the exact score.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
mod normal;
mod sort;

use normal::clip_bounds;
pub(crate) use normal::norm_ppf;
use sort::{from_order_key, order_key, radix_sort};

use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

/// sklearn's BOUNDS_THRESHOLD: inputs this close to the outer quantiles map to the clip value.
const BOUNDS_THRESHOLD: f64 = 1e-7;
/// Rows per task: a chunk (160 KB at 40 features) and its transposes stay in L2. Not a power of
/// two, so that the columns of a transposed chunk do not all map to the same L1 cache sets.
const TRANSFORM_CHUNK_ROWS: usize = 1000;
/// Values searched in lockstep. The steps of one binary search depend on each other, so only
/// independent searches keep the core busy.
const LANES: usize = 32;
/// Lockstep searches of the scalar code, as many as the index registers can hold.
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
    if x.iter().any(|v| v.is_nan()) {
        // sklearn ignores NaN in the fit. Sorted with the values, NaN would shift every quantile.
        return Err("cannot fit a quantile transform on NaN values".to_string());
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
    // The fit takes a group of features at a time. One parallel pass over the rows extracts their
    // columns, so the fit reads the matrix once per group and not once per feature.
    let group = rayon::current_num_threads().clamp(1, n_features);
    let mut quantiles = Vec::with_capacity(n_features * n_quantiles);
    for first in (0..n_features).step_by(group) {
        let mut columns: Vec<Vec<u32>> = (first..(first + group).min(n_features))
            .map(|_| vec![0; n])
            .collect();
        // The same row block of every column.
        let mut blocks: Vec<Vec<&mut [u32]>> = (0..n.div_ceil(TRANSFORM_CHUNK_ROWS))
            .map(|_| Vec::with_capacity(group))
            .collect();
        for column in &mut columns {
            for (block, piece) in blocks
                .iter_mut()
                .zip(column.chunks_mut(TRANSFORM_CHUNK_ROWS))
            {
                block.push(piece);
            }
        }
        blocks
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut block)| {
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
pub fn transform(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    quantiles: &[f64],
    n_quantiles: usize,
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
    let backend = crate::simd::get_backend();
    let maps: Vec<FeatureMap> = quantiles
        .chunks_exact(n_quantiles)
        .map(|q| FeatureMap::new(q.to_vec(), clip))
        .collect();
    let mut out = vec![0.0f32; x.len()];
    let chunk_len = TRANSFORM_CHUNK_ROWS * n_features;
    out.par_chunks_mut(chunk_len)
        .zip(x.par_chunks(chunk_len))
        .for_each_init(
            || (vec![0.0f32; chunk_len], vec![0.0f32; chunk_len]),
            |(columns, scores), (out_chunk, input)| {
                let rows = input.len() / n_features;
                // Feature by feature on contiguous columns, so the tables of one feature stay in
                // L1.
                transpose(input, rows, n_features, columns);
                for (f, map) in maps.iter().enumerate() {
                    let column = f * rows..(f + 1) * rows;
                    backend.quantile_transform_column(
                        map,
                        &columns[column.clone()],
                        &mut scores[column],
                    );
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

/// numpy "linear" percentiles of sorted values (as `order_key`s), made monotone as in sklearn.
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

/// The fitted quantiles of one feature and the normal scores at and between them.
pub(crate) struct FeatureMap {
    quantiles: Vec<f64>,
    knot_scores: Vec<f64>,
    /// The score per unit of input along the segment from each knot (0 for an empty segment and
    /// for the last knot).
    slopes: Vec<f64>,
    clip: (f64, f64),
}

impl FeatureMap {
    fn new(quantiles: Vec<f64>, clip: (f64, f64)) -> Self {
        let mut map = FeatureMap {
            quantiles,
            knot_scores: Vec::new(),
            slopes: Vec::new(),
            clip,
        };
        map.knot_scores = map.quantiles.iter().map(|&q| map.exact(q)).collect();
        map.slopes = (0..map.quantiles.len())
            .map(|j| {
                let width = map
                    .quantiles
                    .get(j + 1)
                    .map_or(0.0, |&q| q - map.quantiles[j]);
                if width > 0.0 {
                    (map.knot_scores[j + 1] - map.knot_scores[j]) / width
                } else {
                    0.0
                }
            })
            .collect();
        map
    }

    /// sklearn `_transform_col` for output_distribution="normal".
    fn exact(&self, x: f64) -> f64 {
        let n = self.quantiles.len() - 1;
        let p = if x - BOUNDS_THRESHOLD < self.quantiles[0] {
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
            // At or above the highest knot, the rank is the last rank, as in np.interp. This also
            // applies to a tie group that ends at the highest knot. Without this check, the
            // segment clamp gives the rank before it.
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

    /// Index of the last knot at or below each value (0 below all knots or for NaN). All binary
    /// searches have the same fixed length. Their steps compile to conditional moves, because the
    /// values come in random order.
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

    /// The normal scores of a column (the scalar default of the SIMD backends).
    pub(crate) fn apply_column(&self, x: &[f32], out: &mut [f32]) {
        self.apply_blocks(x, out, Self::apply)
    }

    /// `apply` on every block of `LANES` values of a column, with a padded tail.
    fn apply_blocks(
        &self,
        x: &[f32],
        out: &mut [f32],
        apply: impl Fn(&Self, [f32; LANES]) -> [f32; LANES],
    ) {
        let mut x_blocks = x.chunks_exact(LANES);
        let mut out_blocks = out.chunks_exact_mut(LANES);
        for (x, out) in (&mut x_blocks).zip(&mut out_blocks) {
            out.copy_from_slice(&apply(self, x.try_into().unwrap()));
        }
        let (x, out) = (x_blocks.remainder(), out_blocks.into_remainder());
        if !x.is_empty() {
            let mut padded = [0.0f32; LANES];
            padded[..x.len()].copy_from_slice(x);
            out.copy_from_slice(&apply(self, padded)[..x.len()]);
        }
    }

    fn apply(&self, x: [f32; LANES]) -> [f32; LANES] {
        let mut out = [0.0f32; LANES];
        for (x, out) in x
            .chunks_exact(SCALAR_LANES)
            .zip(out.chunks_exact_mut(SCALAR_LANES))
        {
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

    /// Whether the score of `x` in segment `j` is interpolated. This is true at a knot, where the
    /// interpolation gives the exact knot score to a tie group (often the 0 of a feature in most
    /// rows). It is also true in the inner segments. The first segment, the last two segments and
    /// values outside the quantiles get the exact score, because there the normal quantile
    /// function bends sharply.
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
        let map = super::FeatureMap::new(quantiles.to_vec(), super::clip_bounds());
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
