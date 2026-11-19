//! Accurate-mass part of the MS1 block.
//!
//! At low input a chance peak lies within the MS1 extraction tolerance (4 residual sd of the calibrated error) at a
//! random peptide m/z in ~40 % of the apex windows, anywhere in the window, so presence and co-elution are weak
//! evidence. A real precursor's M0 sits close to the calibrated m/z. Two features count, per MS1 scan kept apart and
//! per apex cycle, the points with an M0 / M+1 peak within `ACCURATE_PPM` of the calibrated MS1 m/z.
//!
//! A candidate without library isotopes (or with only M0), or a run without MS1 scans, gets 0. All values are finite.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::ms1::{Ms1Features, APEX_HALF_WIDTH};
use crate::peak_group_scoring::utils::apex_window;

/// Tolerance (ppm) of the accurate-mass traces, about 1.7 residual sd of the calibrated MS1 error at 5 ng Astral
/// (0.9 ppm). The extraction tolerance (4 sd) admits a chance peak at a random peptide m/z in ~40 % of the apex
/// windows, anywhere in the window; a real precursor's M0 sits close to the calibrated m/z, so the share of apex scans
/// with a peak this close separates the two better than presence and co-elution.
pub const ACCURATE_PPM: f32 = 1.5;

/// Sets `ms1_m0_accurate_share` and `ms1_m1_accurate_share`: the shares of the apex points with M0 and with M+1 signal
/// in the accurate-mass traces of each MS1 scan. Leaves them unchanged without scans or apex points.
///
/// `scan_xics` holds one [2, cycle] array per MS1 scan of the cycle (rows M0, M+1, extracted at ACCURATE_PPM over the
/// candidate's cycles); the scans are kept apart because the cycle sum would count a peak once per cycle however many
/// scans carry it.
pub fn set_accuracy_features(
    features: &mut Ms1Features,
    scan_xics: &[Array2<f32>],
    fragment_profile: &[f32],
) {
    if scan_xics.is_empty() || fragment_profile.is_empty() {
        return;
    }
    let (start, stop) = apex_window(fragment_profile, APEX_HALF_WIDTH);
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
        return;
    }
    features.ms1_m0_accurate_share = hits[0] as f32 / points as f32;
    features.ms1_m1_accurate_share = hits[1] as f32 / points as f32;
}
