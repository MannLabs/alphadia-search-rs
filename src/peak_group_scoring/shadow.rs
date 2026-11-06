//! Fragment M-1 "shadow" features: is a matched fragment peak really the M+1 isotope of another ion?
//!
//! A chance fragment match lands on isotope peaks about as often as they occur in the spectrum, while a real fragment
//! sits on a monoisotopic peak. So for each fragment the XIC one isotope spacing below it (for a 1+ and a 2+
//! interfering ion) is extracted over the same observations, cycles and tolerance. A fragment is "shadowed" when it
//! is coherent with the peak group and its shadow is coherent too and intense enough to explain the fragment as that
//! ion's M+1. This is m/z evidence: it does not need several points across the peak.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::ms1::C13_MASS_DELTA;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, AGREEMENT_MIN_CORRELATION, LIB_COSINE_APEX_HALF_WIDTH,
};
use crate::utils::create_ranked_mask;

/// Charges of the interfering ion whose M+1 could sit on the fragment.
pub const SHADOW_CHARGES: [u8; 2] = [1, 2];
/// A shadow counts when its XIC correlates with the median fragment profile above this.
const SHADOW_MIN_CORRELATION: f32 = 0.5;
/// A shadow counts when its apex intensity is at least this fraction of the M0 that would explain the fragment as M+1.
const SHADOW_MIN_EXPLAINED: f32 = 0.5;
/// Averagine M1/M0 ratio per Dalton of ion mass.
const AVERAGINE_M1_PER_DA: f32 = 5.6e-4;
/// The library's strongest fragments whose shadow correlations are summed.
const TOP_FRAGMENTS: usize = 6;
const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShadowFeatures {
    /// library-weighted share of the coherent fragments that are shadowed
    pub shadow_lib_share: f32,
    /// sum over the coherent, not shadowed fragments of library share x correlation
    pub shadow_clean_libw_corr: f32,
    /// sum over the top library fragments of the positive part of their shadow correlation
    pub shadow_corr_sum6: f32,
}

impl ShadowFeatures {
    /// Feature values for a candidate whose traces do not fit together.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// m/z of the shadow traces: all fragments one 1+ isotope spacing below, then all fragments one 2+ spacing below.
pub fn shadow_mz(fragment_mz: &[f32]) -> Vec<f32> {
    SHADOW_CHARGES
        .iter()
        .flat_map(|&z| {
            fragment_mz
                .iter()
                .map(move |&mz| mz - C13_MASS_DELTA / z as f32)
        })
        .collect()
}

/// Averagine M0/M1 intensity ratio of an ion of mass `mass` (Da).
fn m0_over_m1(mass: f32) -> f32 {
    1.0 / (AVERAGINE_M1_PER_DA * mass).max(EPSILON)
}

/// Shadow features of one candidate.
///
/// `xic`: [fragment, cycle] over the candidate window, `shadow_xic`: [trace, cycle] over the same cycles with traces
/// as in `shadow_mz`, `median_profile`: the median fragment profile, `correlations`: each fragment's correlation with
/// it, `library_intensity` and `fragment_mz`: per fragment.
pub fn shadow_features(
    xic: &Array2<f32>,
    shadow_xic: &Array2<f32>,
    median_profile: &[f32],
    correlations: &[f32],
    library_intensity: &[f32],
    fragment_mz: &[f32],
) -> ShadowFeatures {
    let (n_fragments, n_cycles) = xic.dim();
    if n_fragments == 0
        || n_cycles == 0
        || shadow_xic.dim() != (SHADOW_CHARGES.len() * n_fragments, n_cycles)
        || median_profile.len() != n_cycles
        || correlations.len() != n_fragments
        || library_intensity.len() != n_fragments
        || fragment_mz.len() != n_fragments
    {
        return ShadowFeatures::missing();
    }
    let total: f32 = library_intensity.iter().sum();
    if total <= 0.0 {
        return ShadowFeatures::missing();
    }

    let apex = median_profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let start = apex.saturating_sub(LIB_COSINE_APEX_HALF_WIDTH);
    let stop = (apex + LIB_COSINE_APEX_HALF_WIDTH + 1).min(n_cycles);
    let apex_sum =
        |m: &Array2<f32>, row: usize| -> f32 { (start..stop).map(|c| m[[row, c]]).sum() };

    let top = create_ranked_mask(library_intensity, 0, TOP_FRAGMENTS);
    let mut shadowed_lib = 0.0f32;
    let mut coherent_lib = 0.0f32;
    let mut clean_corr = 0.0f32;
    let mut corr_sum6 = 0.0f32;
    for f in 0..n_fragments {
        let (row, z) = SHADOW_CHARGES
            .iter()
            .enumerate()
            .map(|(k, &z)| (k * n_fragments + f, z))
            .max_by(|a, b| apex_sum(shadow_xic, a.0).total_cmp(&apex_sum(shadow_xic, b.0)))
            .unwrap();
        let shadow_trace = shadow_xic.row(row).to_vec();
        let shadow_corr = calculate_correlation_safe(&shadow_trace, median_profile);
        let shadow_ratio = apex_sum(shadow_xic, row) / apex_sum(xic, f).max(EPSILON);
        let shadow_mass = (fragment_mz[f] - C13_MASS_DELTA / z as f32) * z as f32;

        let share = library_intensity[f] / total;
        let coherent = correlations[f] > AGREEMENT_MIN_CORRELATION;
        let shadowed = coherent
            && shadow_corr > SHADOW_MIN_CORRELATION
            && shadow_ratio > SHADOW_MIN_EXPLAINED * m0_over_m1(shadow_mass);
        if coherent {
            coherent_lib += share;
            if shadowed {
                shadowed_lib += share;
            } else {
                clean_corr += share * correlations[f];
            }
        }
        if top[f] {
            corr_sum6 += shadow_corr.max(0.0);
        }
    }

    ShadowFeatures {
        shadow_lib_share: shadowed_lib / coherent_lib.max(EPSILON),
        shadow_clean_libw_corr: clean_corr,
        shadow_corr_sum6: corr_sum6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;
    const CYCLES: usize = 7;
    const PEAK: [f32; CYCLES] = [0.0, 1.0, 4.0, 10.0, 4.0, 1.0, 0.0];

    fn fragments() -> Array2<f32> {
        let mut xic = Array2::zeros((N, CYCLES));
        for f in 0..N {
            for c in 0..CYCLES {
                xic[[f, c]] = PEAK[c] * 100.0;
            }
        }
        xic
    }

    fn library() -> Vec<f32> {
        vec![1.0, 0.8, 0.6, 0.4]
    }

    fn mz() -> Vec<f32> {
        vec![500.0, 700.0, 900.0, 1100.0]
    }

    fn score(shadow: &Array2<f32>) -> ShadowFeatures {
        shadow_features(&fragments(), shadow, &PEAK, &[0.99; N], &library(), &mz())
    }

    #[test]
    fn shadow_mz_lists_1plus_then_2plus_spacings() {
        let out = shadow_mz(&[500.0, 800.0]);
        assert_eq!(out.len(), 4);
        assert!((out[0] - (500.0 - C13_MASS_DELTA)).abs() < 1e-4);
        assert!((out[1] - (800.0 - C13_MASS_DELTA)).abs() < 1e-4);
        assert!((out[2] - (500.0 - C13_MASS_DELTA / 2.0)).abs() < 1e-4);
    }

    #[test]
    fn clean_fragments_are_not_shadowed() {
        let out = score(&Array2::zeros((2 * N, CYCLES)));
        assert_eq!(out.shadow_lib_share, 0.0);
        assert_eq!(out.shadow_corr_sum6, 0.0);
        assert!((out.shadow_clean_libw_corr - 0.99).abs() < 1e-5);
    }

    #[test]
    fn a_coeluting_intense_m_minus_1_shadows_the_fragment() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        // fragment 0 (500 m/z, library share 0.357): a 1+ ion at M-1 three times as intense
        for c in 0..CYCLES {
            shadow[[0, c]] = PEAK[c] * 300.0;
        }
        let out = score(&shadow);
        assert!((out.shadow_lib_share - 1.0 / 2.8).abs() < 1e-5);
        assert!((out.shadow_clean_libw_corr - 0.99 * 1.8 / 2.8).abs() < 1e-5);
        assert!((out.shadow_corr_sum6 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_weak_or_non_coeluting_shadow_does_not_count() {
        let mut weak = Array2::zeros((2 * N, CYCLES));
        let mut off_peak = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            // M0/M1 of a 500 Da ion is ~3.6, so 0.5 x that needs a shadow of > 1.8 x the fragment
            weak[[0, c]] = PEAK[c] * 100.0;
            off_peak[[0, c]] = PEAK[(c + 3) % CYCLES] * 1000.0;
        }
        assert_eq!(score(&weak).shadow_lib_share, 0.0);
        assert!(score(&weak).shadow_corr_sum6 > 0.9);
        assert_eq!(score(&off_peak).shadow_lib_share, 0.0);
    }

    #[test]
    fn the_stronger_shadow_charge_is_kept() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            // fragment 3 (1100 m/z): a weak 1+ shadow and an intense 2+ shadow (a 2+ ion of ~2200 Da: M0/M1 ~0.8)
            shadow[[3, c]] = PEAK[c] * 10.0;
            shadow[[N + 3, c]] = PEAK[c] * 100.0;
        }
        let out = score(&shadow);
        assert!((out.shadow_lib_share - 0.4 / 2.8).abs() < 1e-5);
    }

    #[test]
    fn incoherent_fragments_are_never_shadowed() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            shadow[[0, c]] = PEAK[c] * 1000.0;
        }
        let out = shadow_features(
            &fragments(),
            &shadow,
            &PEAK,
            &[0.1, 0.99, 0.99, 0.99],
            &library(),
            &mz(),
        );
        assert_eq!(out.shadow_lib_share, 0.0);
        assert!((out.shadow_clean_libw_corr - 0.99 * 1.8 / 2.8).abs() < 1e-5);
    }

    #[test]
    fn mismatched_inputs_give_missing_values() {
        assert_eq!(
            score(&Array2::zeros((N, CYCLES))),
            ShadowFeatures::missing()
        );
        assert_eq!(
            shadow_features(
                &fragments(),
                &Array2::zeros((2 * N, CYCLES)),
                &PEAK[..3],
                &[0.99; N],
                &library(),
                &mz(),
            ),
            ShadowFeatures::missing()
        );
    }
}
