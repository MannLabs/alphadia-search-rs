//! MS1 elution-profile part of the MS1 block: does the precursor's isotope envelope co-elute with the candidate's
//! fragments over the whole candidate window?
//!
//! Each isotope XIC (traces M-1 .. M+(n-1) as in `ms1::isotope_mz`) is compared with the median fragment profile over
//! all the candidate's cycles, and its intensity is the least-squares scale of that profile rather than the sum over
//! the apex cycles:
//! - `ms1_isotope_correlation`: the M+1/M+2 XICs follow the fragment profile,
//! - `ms1_isotope_cosine`: the profile-fitted M0..M+2 intensities follow the library isotope distribution,
//! - `ms1_m_minus_1_frac`: no co-eluting signal one isotope below M0 (else M0 is another precursor's M+1),
//! - `ms1_signal_at_apex`, `ms1_log_intensity`: M0 signal at the fragment apex and its profile-fitted height.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::ms1::{finite_or_zero, ISOTOPES_BELOW};
use crate::peak_group_scoring::utils::calculate_correlation_safe;

/// Isotopes M0, M+1, M+2 compared with the library isotope distribution.
pub const NUM_PROFILE_ISOTOPES: usize = 3;

/// The elution-profile features of one candidate, all 0 for a candidate without MS1 data.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ms1ProfileFeatures {
    pub ms1_signal_at_apex: f32,
    pub ms1_isotope_correlation: f32,
    pub ms1_isotope_cosine: f32,
    pub ms1_m_minus_1_frac: f32,
    pub ms1_log_intensity: f32,
}

/// Least-squares scale `a >= 0` of `profile` that fits `xic` (`xic ~ a * profile`).
fn profile_scale(xic: &[f32], profile: &[f32], profile_norm: f32) -> f32 {
    if profile_norm <= 0.0 {
        return 0.0;
    }
    let dot: f32 = xic.iter().zip(profile).map(|(x, m)| x * m).sum();
    (dot / profile_norm).max(0.0)
}

/// Elution-profile features from the isotope XICs (`dense_xic`: [trace, cycle], traces M-1 .. M+(n-1)).
///
/// `fragment_profile` is the candidate's median fragment profile over the same cycles and `apex` its apex column
/// (`window_share::apex_column`). Only the first `NUM_PROFILE_ISOTOPES` library abundances are used; with fewer, the cosine and the
/// isotope correlation use those there are.
pub fn calculate_ms1_profile_features(
    dense_xic: &Array2<f32>,
    isotope_intensity: &[f32],
    fragment_profile: &[f32],
    apex: usize,
) -> Ms1ProfileFeatures {
    let num_isotopes = isotope_intensity.len().min(NUM_PROFILE_ISOTOPES);
    let n_cycles = dense_xic.ncols();
    if num_isotopes == 0
        || n_cycles == 0
        || dense_xic.nrows() != ISOTOPES_BELOW + isotope_intensity.len()
        || fragment_profile.len() != n_cycles
    {
        return Ms1ProfileFeatures::default();
    }

    // trace of isotope k (M-1 is k = -1)
    let trace =
        |k: i32| -> Vec<f32> { dense_xic.row((k + ISOTOPES_BELOW as i32) as usize).to_vec() };
    let mono = trace(0);

    let profile_norm: f32 = fragment_profile.iter().map(|m| m * m).sum();
    let profile_max = fragment_profile.iter().cloned().fold(0.0f32, f32::max);
    let scale = |k: i32| profile_scale(&trace(k), fragment_profile, profile_norm);

    let ms1_signal_at_apex = if apex < n_cycles && mono[apex] > 0.0 {
        1.0
    } else {
        0.0
    };
    let ms1_isotope_correlation = if num_isotopes > 1 {
        (1..num_isotopes as i32)
            .map(|k| calculate_correlation_safe(&trace(k), fragment_profile))
            .sum::<f32>()
            / (num_isotopes - 1) as f32
    } else {
        0.0
    };

    let observed: Vec<f32> = (0..num_isotopes as i32).map(scale).collect();
    let library = &isotope_intensity[..num_isotopes];
    let dot: f32 = observed.iter().zip(library).map(|(o, t)| o * t).sum();
    let norm_observed = observed.iter().map(|o| o * o).sum::<f32>().sqrt();
    let norm_library = library.iter().map(|t| t * t).sum::<f32>().sqrt();
    let ms1_isotope_cosine = if norm_observed > 0.0 && norm_library > 0.0 {
        dot / (norm_observed * norm_library)
    } else {
        0.0
    };

    let (scale_minus1, scale_mono) = (scale(-1), observed[0]);
    let ms1_m_minus_1_frac = if scale_minus1 + scale_mono > 0.0 {
        scale_minus1 / (scale_minus1 + scale_mono)
    } else {
        0.0
    };

    Ms1ProfileFeatures {
        ms1_signal_at_apex,
        ms1_isotope_correlation: finite_or_zero(ms1_isotope_correlation),
        ms1_isotope_cosine: finite_or_zero(ms1_isotope_cosine),
        ms1_m_minus_1_frac: finite_or_zero(ms1_m_minus_1_frac),
        ms1_log_intensity: finite_or_zero((1.0 + scale_mono * profile_max).log10()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    /// M-1, M0..M+3 traces of a co-eluting envelope with library abundances 1, 0.6, 0.2, 0.05
    fn coeluting() -> (Array2<f32>, [f32; 4], [f32; 7]) {
        let profile = [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 10.0, 30.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 18.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 6.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.5, 0.5, 0.0, 0.0],
        ]);
        (xic, [1.0, 0.6, 0.2, 0.05], profile)
    }

    #[test]
    fn coeluting_envelope_scores_well() {
        let (xic, abundance, profile) = coeluting();

        let f = calculate_ms1_profile_features(&xic, &abundance, &profile, 3);

        assert_eq!(f.ms1_signal_at_apex, 1.0);
        assert!(f.ms1_isotope_correlation > 0.999);
        assert!(f.ms1_isotope_cosine > 0.999);
        assert_eq!(f.ms1_m_minus_1_frac, 0.0);
        // M0 = 10 x profile, profile maximum 3
        assert!((f.ms1_log_intensity - 31.0f32.log10()).abs() < 1e-5);
    }

    #[test]
    fn only_m0_to_m2_enter_the_cosine() {
        let (mut xic, abundance, profile) = coeluting();
        // an M+3 far off the library abundance changes nothing
        xic.row_mut(4).assign(&numpy::ndarray::arr1(&[
            0.0, 0.0, 50.0, 150.0, 50.0, 0.0, 0.0,
        ]));

        let f = calculate_ms1_profile_features(&xic, &abundance, &profile, 3);

        assert!(f.ms1_isotope_cosine > 0.999);
    }

    #[test]
    fn signal_off_the_profile_does_not_count() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
        ]);

        let f = calculate_ms1_profile_features(&xic, &[1.0, 0.6], &profile, 2);

        assert_eq!(f.ms1_signal_at_apex, 0.0);
        assert!(f.ms1_isotope_correlation < 0.0);
        assert_eq!(f.ms1_isotope_cosine, 0.0);
        assert_eq!(f.ms1_log_intensity, 0.0);
    }

    #[test]
    fn another_precursors_isotope_raises_the_m_minus_1_frac() {
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[30.0, 90.0, 30.0], [10.0, 30.0, 10.0], [6.0, 18.0, 6.0]]);

        let f = calculate_ms1_profile_features(&xic, &[1.0, 0.6], &profile, 1);

        assert!((f.ms1_m_minus_1_frac - 0.75).abs() < 1e-5);
    }

    #[test]
    fn no_isotopes_mismatched_or_empty_traces_give_zero() {
        let xic = Array2::<f32>::zeros((1, 3));
        let profile = [1.0, 2.0, 1.0];
        let zero = Ms1ProfileFeatures::default();
        assert_eq!(calculate_ms1_profile_features(&xic, &[], &profile, 1), zero);
        assert_eq!(
            calculate_ms1_profile_features(&xic, &[1.0, 0.5], &profile, 1),
            zero
        );
        let empty = Array2::<f32>::zeros((3, 5));
        assert_eq!(
            calculate_ms1_profile_features(&empty, &[1.0, 0.5], &[0.0; 5], 0),
            zero
        );
    }
}
