//! Fragment M-1 "shadow" feature of the detection block: is a matched fragment peak really the M+1 isotope of another
//! ion?
//!
//! A chance fragment match lands on isotope peaks about as often as they occur in the spectrum, while a real fragment
//! sits on a monoisotopic peak. So for each fragment the XIC one isotope spacing below it (for a 1+ and a 2+
//! interfering ion) is extracted over the same observations, cycles and tolerance. A fragment is "shadowed" when it
//! is coherent with the peak group and its shadow is coherent too and intense enough to explain the fragment as that
//! ion's M+1. This is m/z evidence: it does not need several points across the peak.
//!
//! The detection block keeps `shadow_lib_share`, the library-weighted share of the coherent fragments that are
//! shadowed. It is finite, 0 when the inputs do not fit together.

use numpy::ndarray::Array2;
use std::cmp::Ordering;

use crate::peak_group_scoring::library_agreement::LIB_COSINE_APEX_HALF_WIDTH;
use crate::peak_group_scoring::ms1::C13_MASS_DELTA;
use crate::peak_group_scoring::utils::{
    apex_window, calculate_correlation_safe, COELUTION_MIN_CORRELATION,
};

/// Charges of the interfering ion whose M+1 could sit on the fragment.
pub const SHADOW_CHARGES: [u8; 2] = [1, 2];
/// A shadow counts when its XIC correlates with the median fragment profile above this.
const SHADOW_MIN_CORRELATION: f32 = 0.5;
/// A shadow counts when its apex intensity is at least this fraction of the M0 that would explain the fragment as M+1.
const SHADOW_MIN_EXPLAINED: f32 = 0.5;
/// Averagine M1/M0 ratio per Dalton of ion mass.
const AVERAGINE_M1_PER_DA: f32 = 5.6e-4;
const EPSILON: f32 = 1e-6;

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

/// Library-weighted share of the coherent fragments (correlation above `COELUTION_MIN_CORRELATION`) that are shadowed;
/// 0 if the inputs do not fit together.
///
/// `xic`: [fragment, cycle] over the candidate window, `shadow_xic`: [trace, cycle] over the same cycles with traces
/// as in `shadow_mz`, `median_profile`: the median fragment profile, `correlations`: each fragment's correlation with
/// it, `library_intensity` and `fragment_mz`: per fragment.
pub fn shadow_lib_share(
    xic: &Array2<f32>,
    shadow_xic: &Array2<f32>,
    median_profile: &[f32],
    correlations: &[f32],
    library_intensity: &[f32],
    fragment_mz: &[f32],
) -> f32 {
    let (n_fragments, n_cycles) = xic.dim();
    if n_fragments == 0
        || n_cycles == 0
        || shadow_xic.dim() != (SHADOW_CHARGES.len() * n_fragments, n_cycles)
        || median_profile.len() != n_cycles
        || correlations.len() != n_fragments
        || library_intensity.len() != n_fragments
        || fragment_mz.len() != n_fragments
    {
        return 0.0;
    }
    let total: f32 = library_intensity.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let (start, stop) = apex_window(median_profile, LIB_COSINE_APEX_HALF_WIDTH);
    let apex_sum =
        |m: &Array2<f32>, row: usize| -> f32 { (start..stop).map(|c| m[[row, c]]).sum() };

    let mut shadowed_lib = 0.0f32;
    let mut coherent_lib = 0.0f32;
    for f in 0..n_fragments {
        // the shadow trace with the most apex signal; on a tie the later charge, as `Iterator::max_by`
        let (row, z) = SHADOW_CHARGES
            .iter()
            .enumerate()
            .map(|(k, &z)| (k * n_fragments + f, z))
            .fold((f, SHADOW_CHARGES[0]), |best, next| {
                if apex_sum(shadow_xic, best.0).total_cmp(&apex_sum(shadow_xic, next.0))
                    == Ordering::Greater
                {
                    best
                } else {
                    next
                }
            });
        let shadow_trace = shadow_xic.row(row).to_vec();
        let shadow_corr = calculate_correlation_safe(&shadow_trace, median_profile);
        let shadow_ratio = apex_sum(shadow_xic, row) / apex_sum(xic, f).max(EPSILON);
        let shadow_mass = (fragment_mz[f] - C13_MASS_DELTA / z as f32) * z as f32;

        let share = library_intensity[f] / total;
        let coherent = correlations[f] > COELUTION_MIN_CORRELATION;
        let shadowed = coherent
            && shadow_corr > SHADOW_MIN_CORRELATION
            && shadow_ratio > SHADOW_MIN_EXPLAINED * m0_over_m1(shadow_mass);
        if coherent {
            coherent_lib += share;
            if shadowed {
                shadowed_lib += share;
            }
        }
    }

    shadowed_lib / coherent_lib.max(EPSILON)
}
