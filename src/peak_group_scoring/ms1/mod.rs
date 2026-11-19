//! MS1 precursor-isotope features of a candidate peak group (final feature set only).
//!
//! The isotope XICs M-1 .. M+(n-1) are extracted from the MS1 survey scans over the candidate's cycles at the MS1 m/z
//! the library sets per precursor (alphaDIA: every precursor at its own m/z, shifted by its MS1 calibration), at the
//! calibrated MS1 tolerance. The block combines four views on them:
//! - the whole candidate window against the median fragment profile (`profile`),
//! - the isotope ratio M+1/M0 at the fragment apex (`ms1_iso_m1_err`, `calculate_ms1_apex_features` here),
//! - the share of apex scans with an M0 / M+1 peak close to the calibrated m/z (`accuracy`),
//! - the M0 / M+1 points of each MS1 scan at its own time within the cycle (`fine`).
//!
//! The ten features are the ones a feature selection kept from a larger pool of MS1 features on HeLa 2-12 min
//! gradients, plasma and ProteoBench.
//!
//! A candidate without library isotopes (`SpecLibFlat::set_isotopes` not called) or a run without MS1 scans gets 0 in
//! every feature. All values are finite, never NaN, because NaN rows are dropped before the classifier.

use numpy::ndarray::{s, Array2};

use crate::candidate::Candidate;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::observation::PeakGroupObservation;
use crate::peak_group_scoring::utils::{apex_window, finite_or_zero};
use crate::peak_group_scoring::window_share::apex_column;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;

pub mod accuracy;
pub mod fine;
pub mod profile;

#[cfg(test)]
mod tests;

use accuracy::{set_accuracy_features, ACCURATE_PPM};
use fine::set_fine_features;
use profile::set_profile_features;

/// Mass difference between the 13C and 12C isotopes in Dalton.
pub const C13_MASS_DELTA: f32 = 1.003_354_8;
/// Isotope traces extracted below the monoisotopic one (M-1: another precursor's isotope on our M0).
pub const ISOTOPES_BELOW: usize = 1;
/// Cycles on each side of the fragment apex that count as the apex.
pub const APEX_HALF_WIDTH: usize = 1;
/// Added before logarithms so that an empty trace stays finite.
const LOG_EPSILON: f32 = 1.0;

feature_block! {
    /// The MS1 features of one candidate.
    pub struct Ms1Features {
        /// 1 if the M0 XIC has signal at the apex of the fragment profile, else 0
        pub ms1_signal_at_apex: f32,
        /// mean Pearson correlation of the M+1 and M+2 XICs with the median fragment profile
        pub ms1_isotope_correlation: f32,
        /// cosine of the profile-fitted M0..M+2 intensities with the library isotope abundances
        pub ms1_isotope_cosine: f32,
        /// profile-fitted M-1 intensity over that of M-1 and M0 (M0 is another precursor's M+1)
        pub ms1_m_minus_1_frac: f32,
        /// log10(1 + profile-fitted M0 intensity at the profile maximum)
        pub ms1_log_intensity: f32,
        /// log2 of the observed apex M+1/M0 ratio minus log2 of the library ratio
        pub ms1_iso_m1_err: f32,
        /// share of the apex points (MS1 scan x apex cycle) with an M0 peak within `ACCURATE_PPM`
        pub ms1_m0_accurate_share: f32,
        /// share of the apex points with an M+1 peak within `ACCURATE_PPM`
        pub ms1_m1_accurate_share: f32,
        /// Pearson correlation of the per-scan M0 points with a Gaussian of the fragment FWHM at the fragment apex time
        pub ms1_fine_shape: f32,
        /// Pearson correlation of the per-scan M0 and M+1 points
        pub ms1_fine_iso_corr: f32,
    }
}

impl Ms1Features {
    /// Feature values for a candidate without MS1 data or without library isotopes.
    pub fn missing() -> Self {
        Self::default()
    }
}

feature_block! {
    /// The apex view on the isotope XICs, the isotopes summed over the apex cycles of the fragment profile: what
    /// alphaDIA's MS1 calibration pre-pass (`PeakGroupScoring::score_ms1_calibration`) needs to pick confident
    /// precursors and their mass error. All 0 without MS1 data or without library isotopes.
    pub struct Ms1ApexFeatures {
        /// cosine of the apex isotope intensities M0.. with the library isotope abundances
        pub ms1_iso_cosine: f32,
        /// number of isotopes M0.. with signal at the apex
        pub ms1_n_isotopes: f32,
        /// log10 of the monoisotopic intensity summed over the apex cycles
        pub ms1_log_m0_apex: f32,
        /// intensity-weighted signed mass error of M0 at the apex in ppm; 0 without M0 signal
        pub ms1_ppm_signed: f32,
    }
}

/// m/z of the isotope traces M-ISOTOPES_BELOW .. M+(num_isotopes-1) of a precursor.
pub fn isotope_mz(mono_mz: f32, charge: u8, num_isotopes: usize) -> Vec<f32> {
    (0..ISOTOPES_BELOW + num_isotopes)
        .map(|k| mono_mz + (k as f32 - ISOTOPES_BELOW as f32) * C13_MASS_DELTA / charge as f32)
        .collect()
}

/// The apex features from the isotope XICs (`dense_xic`, `dense_mz`: [trace, cycle], traces as in `isotope_mz`), and
/// the MS1 block's `ms1_iso_m1_err` from the same apex isotopes: log2 of the observed apex M+1/M0 ratio minus log2 of
/// the library ratio, 0 without M0 or library M+1. All 0 if the traces do not fit.
///
/// `fragment_profile` is the candidate's median fragment profile over the same cycles.
pub fn calculate_ms1_apex_features(
    dense_xic: &Array2<f32>,
    dense_mz: &Array2<f32>,
    mono_mz: f32,
    isotope_intensity: &[f32],
    fragment_profile: &[f32],
) -> (Ms1ApexFeatures, f32) {
    let num_isotopes = isotope_intensity.len();
    let n_cycles = dense_xic.ncols();
    if num_isotopes == 0
        || n_cycles == 0
        || dense_xic.nrows() != ISOTOPES_BELOW + num_isotopes
        || fragment_profile.len() != n_cycles
    {
        return (Ms1ApexFeatures::default(), 0.0);
    }

    let (start, stop) = apex_window(fragment_profile, APEX_HALF_WIDTH);

    let apex_sum = |trace: usize| -> f32 { (start..stop).map(|c| dense_xic[[trace, c]]).sum() };
    let apex_isotopes: Vec<f32> = (0..num_isotopes)
        .map(|k| apex_sum(ISOTOPES_BELOW + k))
        .collect();
    let m0 = apex_isotopes[0];

    let dot: f32 = apex_isotopes
        .iter()
        .zip(isotope_intensity)
        .map(|(a, i)| a * i)
        .sum();
    let norm = apex_isotopes.iter().map(|a| a * a).sum::<f32>().sqrt()
        * isotope_intensity.iter().map(|i| i * i).sum::<f32>().sqrt();
    let iso_cosine = if norm > 0.0 { dot / norm } else { 0.0 };

    let n_isotopes = apex_isotopes.iter().filter(|&&a| a > 0.0).count() as f32;

    let iso_m1_err =
        if m0 > 0.0 && num_isotopes > 1 && isotope_intensity[0] > 0.0 && isotope_intensity[1] > 0.0
        {
            ((apex_isotopes[1] + LOG_EPSILON) / (m0 + LOG_EPSILON)).log2()
                - (isotope_intensity[1] / isotope_intensity[0]).log2()
        } else {
            0.0
        };

    let mono_row = ISOTOPES_BELOW;
    let mut error_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for c in start..stop {
        let intensity = dense_xic[[mono_row, c]];
        if intensity > 0.0 {
            error_sum += (dense_mz[[mono_row, c]] - mono_mz) / mono_mz * 1e6 * intensity;
            weight_sum += intensity;
        }
    }
    let ppm_signed = if weight_sum > 0.0 {
        error_sum / weight_sum
    } else {
        0.0
    };

    let apex = Ms1ApexFeatures {
        ms1_iso_cosine: finite_or_zero(iso_cosine),
        ms1_n_isotopes: n_isotopes,
        ms1_log_m0_apex: finite_or_zero((m0 + LOG_EPSILON).log10()),
        ms1_ppm_signed: finite_or_zero(ppm_signed),
    };
    (apex, finite_or_zero(iso_m1_err))
}

/// The MS1 isotope traces of a candidate's precursor and their XICs.
struct Ms1IsotopeXic<'a> {
    charge: u8,
    ms1_mz: f32,
    /// library isotope abundances M0..
    abundance: &'a [f32],
    /// m/z of the traces M-1 .. M+(n-1)
    traces: Vec<f32>,
    xic: DenseXICMZObservation,
}

/// The isotope XICs M-1 .. M+(n-1) of a candidate's precursor over the MS1 scans and the candidate's cycles, at the MS1
/// m/z and `ms1_mass_tolerance`; None without MS1 scans or without library isotopes for the precursor.
fn ms1_isotope_xic<'a, T: DIADataTrait>(
    dia_data: &T,
    lib: &'a SpecLibFlat,
    candidate: &Candidate,
    ms1_observations: &[usize],
    ms1_mass_tolerance: f32,
) -> Option<Ms1IsotopeXic<'a>> {
    if ms1_observations.is_empty() {
        return None;
    }
    let (charge, ms1_mz, abundance) = lib.get_isotopes_by_idx(candidate.precursor_idx)?;
    let traces = isotope_mz(ms1_mz, charge, abundance.len());
    let xic = DenseXICMZObservation::from_observations(
        dia_data,
        ms1_observations.to_vec(),
        candidate.cycle_start,
        candidate.cycle_stop,
        ms1_mass_tolerance,
        &traces,
    );
    Some(Ms1IsotopeXic {
        charge,
        ms1_mz,
        abundance,
        traces,
        xic,
    })
}

/// The apex features of one candidate for the MS1 calibration, at `ms1_mass_tolerance` over the MS1 scans
/// `ms1_observations`; all 0 without them or without library isotopes for the precursor.
pub fn calculate_ms1_apex_calibration<T: DIADataTrait>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidate: &Candidate,
    observation: &PeakGroupObservation,
    ms1_observations: &[usize],
    ms1_mass_tolerance: f32,
) -> Ms1ApexFeatures {
    ms1_isotope_xic(
        dia_data,
        lib,
        candidate,
        ms1_observations,
        ms1_mass_tolerance,
    )
    .map_or_else(Ms1ApexFeatures::default, |isotopes| {
        calculate_ms1_apex_features(
            &isotopes.xic.dense_xic,
            &isotopes.xic.dense_mz,
            isotopes.ms1_mz,
            isotopes.abundance,
            &observation.median_profile_filtered,
        )
        .0
    })
}

/// The MS1 block of one candidate, all at the precursor's MS1 m/z over the candidate's cycles: the isotope XICs summed
/// over the MS1 scans of a cycle, against the whole fragment profile (`profile`) and at its apex (`ms1_iso_m1_err`);
/// the M0 / M+1 traces of each MS1 scan apart at ACCURATE_PPM (`accuracy`) and at `ms1_mass_tolerance`, at the scans'
/// own times (`fine`). `ms1_observations` are the MS1 survey scans; without them, or without library isotopes for the
/// precursor, the block takes its missing values.
pub fn calculate_ms1_features<T: DIADataTrait>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidate: &Candidate,
    observation: &PeakGroupObservation,
    fwhm_rt: f32,
    ms1_observations: &[usize],
    ms1_mass_tolerance: f32,
) -> Ms1Features {
    let Some(Ms1IsotopeXic {
        charge,
        ms1_mz,
        abundance: isotope_intensity,
        traces,
        xic: ms1_xic,
    }) = ms1_isotope_xic(
        dia_data,
        lib,
        candidate,
        ms1_observations,
        ms1_mass_tolerance,
    )
    else {
        return Ms1Features::missing();
    };
    let mut features = Ms1Features::missing();
    let fragment_profile = &observation.median_profile_filtered;
    let apex = apex_column(
        fragment_profile,
        candidate.cycle_center.saturating_sub(candidate.cycle_start),
    );
    set_profile_features(
        &mut features,
        &ms1_xic.dense_xic,
        isotope_intensity,
        fragment_profile,
        apex,
    );
    features.ms1_iso_m1_err = calculate_ms1_apex_features(
        &ms1_xic.dense_xic,
        &ms1_xic.dense_mz,
        ms1_mz,
        isotope_intensity,
        fragment_profile,
    )
    .1;

    // M0 and M+1 of each MS1 scan apart
    let scan_xic = |obs: usize, tolerance: f32, mz: &[f32]| {
        DenseXICMZObservation::from_observations(
            dia_data,
            vec![obs],
            candidate.cycle_start,
            candidate.cycle_stop,
            tolerance,
            mz,
        )
        .dense_xic
    };
    let accurate_mz = &traces[ISOTOPES_BELOW..(ISOTOPES_BELOW + 2).min(traces.len())];
    if accurate_mz.len() == 2 {
        let scan_xics: Vec<_> = ms1_observations
            .iter()
            .map(|&obs| scan_xic(obs, ACCURATE_PPM, accurate_mz))
            .collect();
        set_accuracy_features(&mut features, &scan_xics, fragment_profile);
    }

    let fine_mz = [ms1_mz, ms1_mz + C13_MASS_DELTA / charge.max(1) as f32];
    let n_cycles = candidate.cycle_stop - candidate.cycle_start;
    let mut m0 = Array2::<f32>::zeros((ms1_observations.len(), n_cycles));
    let mut m1 = m0.clone();
    for (row, &obs) in ms1_observations.iter().enumerate() {
        let xic = scan_xic(obs, ms1_mass_tolerance, &fine_mz);
        m0.row_mut(row).assign(&xic.row(0));
        m1.row_mut(row).assign(&xic.row(1));
    }
    let ms1_position: Vec<f32> = ms1_observations
        .iter()
        .map(|&obs| dia_data.cycle_position(obs))
        .collect();
    let windows = &observation.contributing_obs_indices;
    let fragment_position = if windows.is_empty() {
        0.0
    } else {
        windows
            .iter()
            .map(|&obs| dia_data.cycle_position(obs))
            .sum::<f32>()
            / windows.len() as f32
    };
    let cycle_rt = dia_data
        .rt_index()
        .rt
        .slice(s![candidate.cycle_start..candidate.cycle_stop])
        .to_vec();
    set_fine_features(
        &mut features,
        &m0,
        &m1,
        &ms1_position,
        &cycle_rt,
        fragment_position,
        fragment_profile,
        fwhm_rt,
    );

    features
}
