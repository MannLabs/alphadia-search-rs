//! MS1 accurate-mass part of the MS1 block (Tomas Jelinek, alphadia-search-rs exp/ms1-accuracy 2337a1e).
//!
//! At low input a chance peak lies within the MS1 extraction tolerance (4 residual sd of the calibrated error) at a
//! random peptide m/z in ~40 % of the apex windows, anywhere in the window, so presence and co-elution are weak
//! evidence. A real precursor's M0 sits close to the calibrated m/z. Two features count, per MS1 scan kept apart and
//! per apex cycle, the points with an M0 / M+1 peak within `ACCURATE_PPM` of the calibrated MS1 m/z.
//!
//! A candidate without library isotopes (or with only M0), or a run without MS1 scans, gets 0. All values are finite.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::ms1::APEX_HALF_WIDTH;

/// Tolerance (ppm) of the accurate-mass traces, about 1.7 residual sd of the calibrated MS1 error at 5 ng Astral
/// (0.9 ppm). The extraction tolerance (4 sd) admits a chance peak at a random peptide m/z in ~40 % of the apex
/// windows, anywhere in the window; a real precursor's M0 sits close to the calibrated m/z, so the share of apex scans
/// with a peak this close separates the two better than presence and co-elution.
pub const ACCURATE_PPM: f32 = 1.5;

/// The accurate-mass features of one candidate, both 0 without MS1 data or without library isotopes M0 and M+1.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ms1AccuracyFeatures {
    /// share of the apex points (MS1 scan x apex cycle) with an M0 peak within ACCURATE_PPM
    pub ms1_m0_accurate_share: f32,
    /// share of the apex points with an M+1 peak within ACCURATE_PPM
    pub ms1_m1_accurate_share: f32,
}

/// Apex cycle (maximum of the fragment profile) and the half-open cycle range around it that counts as the apex, as
/// in the MS1 block.
pub fn apex_window(fragment_profile: &[f32]) -> (usize, usize, usize) {
    let apex = fragment_profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let start = apex.saturating_sub(APEX_HALF_WIDTH);
    let stop = (apex + APEX_HALF_WIDTH + 1).min(fragment_profile.len());
    (apex, start, stop)
}

/// Shares of the apex points with M0 and with M+1 signal in the accurate-mass traces of each MS1 scan.
///
/// `scan_xics` holds one [2, cycle] array per MS1 scan of the cycle (rows M0, M+1, extracted at ACCURATE_PPM over the
/// candidate's cycles); the scans are kept apart because the cycle sum would count a peak once per cycle however many
/// scans carry it.
pub fn calculate_ms1_accuracy_features(
    scan_xics: &[Array2<f32>],
    fragment_profile: &[f32],
) -> Ms1AccuracyFeatures {
    if scan_xics.is_empty() || fragment_profile.is_empty() {
        return Ms1AccuracyFeatures::default();
    }
    let (_, start, stop) = apex_window(fragment_profile);
    let mut hits = [0usize; 2];
    let mut points = 0usize;
    for xic in scan_xics {
        if xic.nrows() < 2 || xic.ncols() != fragment_profile.len() {
            continue;
        }
        for c in start..stop {
            points += 1;
            for (k, hit) in hits.iter_mut().enumerate() {
                if xic[[k, c]] > 0.0 {
                    *hit += 1;
                }
            }
        }
    }
    if points == 0 {
        return Ms1AccuracyFeatures::default();
    }
    Ms1AccuracyFeatures {
        ms1_m0_accurate_share: hits[0] as f32 / points as f32,
        ms1_m1_accurate_share: hits[1] as f32 / points as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    #[test]
    fn accurate_shares_count_apex_scans_with_signal() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0];
        let scan_a = arr2(&[[0.0, 5.0, 5.0, 0.0, 9.0], [0.0, 0.0, 2.0, 0.0, 0.0]]);
        let scan_b = arr2(&[[0.0, 0.0, 5.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

        let f = calculate_ms1_accuracy_features(&[scan_a, scan_b], &profile);

        // apex cycle 2, apex window cycles 1..=3: M0 in 4 of 6 points, M+1 in 1 of 6; cycle 4 is outside
        assert!((f.ms1_m0_accurate_share - 4.0 / 6.0).abs() < 1e-6);
        assert!((f.ms1_m1_accurate_share - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn accurate_shares_without_scans_are_zero() {
        assert_eq!(
            calculate_ms1_accuracy_features(&[], &[1.0, 2.0]),
            Ms1AccuracyFeatures::default()
        );
    }

    #[test]
    fn apex_window_is_the_ms1_blocks_apex() {
        assert_eq!(apex_window(&[0.0, 1.0, 3.0, 1.0, 0.0]), (2, 1, 4));
        assert_eq!(apex_window(&[5.0, 1.0]), (0, 0, 2));
    }
}
