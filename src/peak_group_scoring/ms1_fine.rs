//! MS1 features at the MS1 scans' own times.
//!
//! The cycle-binned MS1 XIC sums every MS1 scan of a cycle into one point. When a cycle holds several MS1 scans (Astral:
//! ~3 per 1.5 s cycle of 300 windows), that throws away the time resolution in which a low-input peak is sampled best:
//! the precursor's MS2 window holds only 2-4 points of the peak, its MS1 trace 6-12. Here every MS1 observation keeps
//! its own point, at the cycle's RT plus its position within the cycle, and the fragment profile is placed at the time
//! of the precursor's own MS2 window, so MS1 and MS2 are compared at the times they were measured.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::utils::calculate_correlation_safe;

/// FWHM to Gaussian standard deviation.
const FWHM_TO_SIGMA: f32 = 1.0 / 2.354_82;
/// The expected MS1 peak is never narrower than this many cycles: the fragment FWHM of a 2-4 point peak is coarse.
const MIN_SIGMA_CYCLES: f32 = 0.5;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ms1FineFeatures {
    /// Pearson correlation of the M0 points with the fragment profile interpolated at their times
    pub ms1_fine_frag_corr: f32,
    /// Pearson correlation of the M0 points with a Gaussian of the fragment FWHM at the fragment apex time
    pub ms1_fine_shape: f32,
    /// M0 apex time minus fragment apex time in RT units (parabolic apex refinement); 0 without M0 signal
    pub ms1_fine_apex_shift: f32,
    /// Pearson correlation of the M0 and M1 points
    pub ms1_fine_iso_corr: f32,
    /// M0 points with signal within one fragment FWHM of the fragment apex
    pub ms1_fine_n_points: f32,
}

impl Ms1FineFeatures {
    /// Feature values for a candidate without MS1 data.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// Apex time of `values` sampled at the evenly spaced `times` (spacing `step`), refined by a parabola through the
/// maximum and its neighbours.
fn refined_apex(times: &[f32], values: &[f32], step: f32) -> Option<f32> {
    let (apex, &top) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if top <= 0.0 {
        return None;
    }
    if apex == 0 || apex + 1 == values.len() {
        return Some(times[apex]);
    }
    let (left, right) = (values[apex - 1], values[apex + 1]);
    let curvature = left - 2.0 * top + right;
    let offset = if curvature < 0.0 {
        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some(times[apex] + offset * step)
}

fn interpolate(times: &[f32], values: &[f32], t: f32) -> f32 {
    if t <= times[0] {
        return values[0];
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return values[last];
    }
    let i = times.partition_point(|&x| x <= t) - 1;
    let w = (t - times[i]) / (times[i + 1] - times[i]);
    values[i] * (1.0 - w) + values[i + 1] * w
}

/// MS1 features from per-observation M0 / M1 XICs.
///
/// `m0`, `m1`: [MS1 observation, cycle] over the candidate's cycles; `ms1_position`: each MS1 observation's position
/// within the cycle as a fraction of the cycle (0..1); `cycle_rt`: RT of the candidate's cycles; `fragment_position`:
/// the precursor's MS2 window position within the cycle (fraction); `fragment_profile`: the median fragment profile
/// over the same cycles; `fwhm`: its FWHM in RT units.
pub fn ms1_fine_features(
    m0: &Array2<f32>,
    m1: &Array2<f32>,
    ms1_position: &[f32],
    cycle_rt: &[f32],
    fragment_position: f32,
    fragment_profile: &[f32],
    fwhm: f32,
) -> Ms1FineFeatures {
    let n_cycles = cycle_rt.len();
    let n_obs = ms1_position.len();
    if n_cycles < 2
        || n_obs == 0
        || m0.dim() != (n_obs, n_cycles)
        || m1.dim() != (n_obs, n_cycles)
        || fragment_profile.len() != n_cycles
    {
        return Ms1FineFeatures::missing();
    }
    let cycle_time = (cycle_rt[n_cycles - 1] - cycle_rt[0]) / (n_cycles - 1) as f32;
    if cycle_time <= 0.0 {
        return Ms1FineFeatures::missing();
    }

    let fragment_times: Vec<f32> = cycle_rt
        .iter()
        .map(|&t| t + fragment_position * cycle_time)
        .collect();
    let Some(fragment_apex) = refined_apex(&fragment_times, fragment_profile, cycle_time) else {
        return Ms1FineFeatures::missing();
    };

    let mut order: Vec<usize> = (0..n_obs).collect();
    order.sort_by(|&a, &b| ms1_position[a].total_cmp(&ms1_position[b]));
    let mut times = Vec::with_capacity(n_obs * n_cycles);
    let mut m0_points = Vec::with_capacity(n_obs * n_cycles);
    let mut m1_points = Vec::with_capacity(n_obs * n_cycles);
    for c in 0..n_cycles {
        for &j in &order {
            times.push(cycle_rt[c] + ms1_position[j] * cycle_time);
            m0_points.push(m0[[j, c]]);
            m1_points.push(m1[[j, c]]);
        }
    }

    let fragment_at_ms1: Vec<f32> = times
        .iter()
        .map(|&t| interpolate(&fragment_times, fragment_profile, t))
        .collect();
    let sigma = (fwhm * FWHM_TO_SIGMA).max(MIN_SIGMA_CYCLES * cycle_time);
    let expected: Vec<f32> = times
        .iter()
        .map(|&t| (-0.5 * ((t - fragment_apex) / sigma).powi(2)).exp())
        .collect();
    let half_width = fwhm.max(cycle_time);
    let n_points = times
        .iter()
        .zip(&m0_points)
        .filter(|(&t, &v)| v > 0.0 && (t - fragment_apex).abs() <= half_width)
        .count();
    let ms1_step = cycle_time / n_obs as f32;
    let apex_shift = refined_apex(&times, &m0_points, ms1_step)
        .map(|t| t - fragment_apex)
        .unwrap_or(0.0);

    Ms1FineFeatures {
        ms1_fine_frag_corr: calculate_correlation_safe(&m0_points, &fragment_at_ms1),
        ms1_fine_shape: calculate_correlation_safe(&m0_points, &expected),
        ms1_fine_apex_shift: apex_shift,
        ms1_fine_iso_corr: calculate_correlation_safe(&m0_points, &m1_points),
        ms1_fine_n_points: n_points as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(t: f32, centre: f32, sigma: f32) -> f32 {
        (-0.5 * ((t - centre) / sigma).powi(2)).exp()
    }

    /// 3 MS1 scans per 1.5 s cycle at positions 0, 1/3, 2/3; the precursor's MS2 window at position 0.5.
    fn traces(centre: f32, sigma: f32, cycle_rt: &[f32]) -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        let positions = vec![0.0, 1.0 / 3.0, 2.0 / 3.0];
        let m0 = Array2::from_shape_fn((3, cycle_rt.len()), |(j, c)| {
            100.0 * gaussian(cycle_rt[c] + positions[j] * 1.5, centre, sigma)
        });
        let profile = cycle_rt
            .iter()
            .map(|&t| gaussian(t + 0.75, centre, sigma))
            .collect();
        (m0, positions, profile)
    }

    #[test]
    fn coeluting_ms1_peak_scores_well_and_is_on_time() {
        let cycle_rt: Vec<f32> = (0..7).map(|c| 100.0 + 1.5 * c as f32).collect();
        let (m0, positions, profile) = traces(104.6, 1.3, &cycle_rt);
        let m1 = m0.mapv(|v| 0.6 * v);

        let f = ms1_fine_features(&m0, &m1, &positions, &cycle_rt, 0.5, &profile, 3.0);

        assert!(f.ms1_fine_frag_corr > 0.95);
        assert!(f.ms1_fine_shape > 0.95);
        assert!(f.ms1_fine_apex_shift.abs() < 0.4);
        assert!(f.ms1_fine_iso_corr > 0.99);
        assert!(f.ms1_fine_n_points >= 6.0);
    }

    #[test]
    fn ms1_peak_off_the_fragment_apex_is_shifted_and_uncorrelated() {
        let cycle_rt: Vec<f32> = (0..7).map(|c| 100.0 + 1.5 * c as f32).collect();
        let (_, positions, profile) = traces(103.0, 1.0, &cycle_rt);
        let (m0, _, _) = traces(109.0, 1.0, &cycle_rt);

        let f = ms1_fine_features(&m0, &m0, &positions, &cycle_rt, 0.5, &profile, 2.4);

        assert!(f.ms1_fine_apex_shift > 4.0);
        assert!(f.ms1_fine_frag_corr < 0.2);
        assert!(f.ms1_fine_shape < 0.2);
    }

    #[test]
    fn no_signal_or_mismatched_shapes_give_missing_or_zero_values() {
        let cycle_rt: Vec<f32> = (0..5).map(|c| 1.5 * c as f32).collect();
        let empty = Array2::<f32>::zeros((3, 5));
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0];
        let f = ms1_fine_features(
            &empty,
            &empty,
            &[0.0, 0.3, 0.6],
            &cycle_rt,
            0.5,
            &profile,
            2.0,
        );
        assert_eq!(f.ms1_fine_apex_shift, 0.0);
        assert_eq!(f.ms1_fine_n_points, 0.0);
        assert_eq!(f.ms1_fine_frag_corr, 0.0);

        let wrong = Array2::<f32>::zeros((2, 5));
        assert_eq!(
            ms1_fine_features(
                &wrong,
                &wrong,
                &[0.0, 0.3, 0.6],
                &cycle_rt,
                0.5,
                &profile,
                2.0
            ),
            Ms1FineFeatures::missing()
        );
    }
}
