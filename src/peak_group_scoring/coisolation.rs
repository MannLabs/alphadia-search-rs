//! Isotope co-isolation features: does the next DIA window show the candidate's fragments as its isotope envelope
//! predicts?
//!
//! With narrow windows (e.g. 2 Th on Astral) a precursor's isotope envelope (M+k at +k * 1.00335 / z) often reaches the
//! neighbouring window, so a present peptide fragments there too, in the same cycle. Fragment f (share s of the
//! precursor mass) then appears in window W at its monoisotopic m/z (j = 0) and at its first isotope (j = 1) with gain
//! g_Wj(f) = sum_k I_k T_W(m_k) Binom(j; k, s), with averagine isotope abundances I_k and a quadrupole transmission T_W
//! with logistic edges. Relative to the channel the search reads (the precursor's own window(s), j = 0) the expected
//! signal is g_Wj(f) / g_P0(f). A chance match in the primary window has no reason to show the same fragments in the
//! neighbour at that ratio.
//!
//! The evidence is scored as more hit/miss terms of the fragment detection likelihood (`detection`): each expected
//! neighbour channel is hit or missed anywhere in apex +-1 (the neighbour is acquired up to half a cycle away from the
//! primary window) against its own flank background, with the signal of fragment f scaled by the gain ratio. Decoys
//! are scored the same way at their own m/z, which is their target's precursor m/z for runtime decoys.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::detection::{self, scales, FLANK_MAX, FLANK_MIN, PRIOR_WEIGHT};
use crate::peak_group_scoring::ms1::C13_MASS_DELTA;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};

/// Isotopes M+0 .. M+(N_ISOTOPES-1) of the precursor envelope.
const N_ISOTOPES: usize = 6;
/// Fragment isotopes read in the neighbour windows (monoisotopic and M+1).
const N_FRAGMENT_ISOTOPES: usize = 2;
/// Logistic width of the quadrupole window edges (Th); 0.1-0.2 fit 5 ng Astral data best.
const EDGE_WIDTH: f32 = 0.15;
/// A neighbour channel is scored when it expects at least this share of the primary channel's signal.
const MIN_GAIN_RATIO: f32 = 0.1;
/// Neighbour windows are considered when they lie within this many edge widths of the envelope.
const EDGE_REACH: f32 = 10.0;
/// Cycles on each side of the apex in which a neighbour channel counts as hit.
const APEX_HALF_WIDTH: usize = 1;
const PROTON_MASS: f32 = 1.007_276;
/// Averagine expected count of +1 Da heavy isotopes (13C, 2H, 15N, 17O, 33S) per Da of peptide mass.
const SINGLE_HEAVY_PER_DA: f32 = 5.359e-4;
/// Averagine expected count of +2 Da heavy isotopes (18O, 34S) per Da of peptide mass.
const DOUBLE_HEAVY_PER_DA: f32 = 4.335e-5;
const MIN_PROBABILITY: f32 = 1e-6;
const EPSILON: f32 = 1e-12;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoisolationFeatures {
    /// library-weighted expected neighbour signal relative to the primary channel (window geometry only)
    pub coiso_expect: f32,
    /// number of neighbour channels scored
    pub coiso_n_channels: f32,
    /// det_llr with the neighbour channels added, minus det_llr (both maximised over the signal scale)
    pub coiso_det_gain: f32,
    /// the neighbour terms alone, maximised over their own signal scale
    pub coiso_det_nb_own: f32,
    /// the neighbour terms alone at det_llr's best signal scale (the abundance the primary window implies)
    pub coiso_det_nb_prim: f32,
}

impl CoisolationFeatures {
    /// Feature values for a candidate without an expected neighbour channel.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// One fragment read in a neighbour window.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourChannel {
    /// index of the neighbour window's quadrupole observation
    pub observation: usize,
    /// index of the fragment in the precursor's fragment list
    pub fragment: usize,
    /// m/z read (the fragment's monoisotopic m/z or its first isotope)
    pub mz: f32,
    /// expected signal relative to the fragment's primary channel
    pub gain_ratio: f32,
}

/// Averagine abundances of M+0 .. M+(N_ISOTOPES-1) of a peptide of neutral mass `mass` (Da), normalised to sum 1:
/// a Poisson count of +1 Da isotopes convolved with a Poisson count of +2 Da isotopes.
pub fn isotope_abundances(mass: f32) -> [f32; N_ISOTOPES] {
    let poisson = |lambda: f32| {
        let mut p = [0.0f32; N_ISOTOPES];
        p[0] = (-lambda).exp();
        for k in 1..N_ISOTOPES {
            p[k] = p[k - 1] * lambda / k as f32;
        }
        p
    };
    let single = poisson(mass * SINGLE_HEAVY_PER_DA);
    let double = poisson(mass * DOUBLE_HEAVY_PER_DA);
    let mut out = [0.0f32; N_ISOTOPES];
    for (d, &pd) in double.iter().enumerate() {
        for (s, &ps) in single.iter().enumerate() {
            if s + 2 * d < N_ISOTOPES {
                out[s + 2 * d] += ps * pd;
            }
        }
    }
    let total: f32 = out.iter().sum();
    out.map(|v| v / total.max(EPSILON))
}

/// Quadrupole transmission of `mz` through the window [lower, upper] with logistic edges.
pub fn transmission(mz: f32, lower: f32, upper: f32) -> f32 {
    let logistic = |x: f32| 1.0 / (1.0 + (-x / EDGE_WIDTH).exp());
    (logistic(mz - lower) - logistic(mz - upper)).max(0.0)
}

/// Gains of the fragment isotopes j = 0, 1 of a fragment with mass share `share` in a window [lower, upper].
fn window_gains(
    isotope_mz: &[f32; N_ISOTOPES],
    abundance: &[f32; N_ISOTOPES],
    share: f32,
    window: [f32; 2],
) -> [f32; N_FRAGMENT_ISOTOPES] {
    let mut gains = [0.0f32; N_FRAGMENT_ISOTOPES];
    for k in 0..N_ISOTOPES {
        let t = transmission(isotope_mz[k], window[0], window[1]);
        if t <= 0.0 {
            continue;
        }
        let w = abundance[k] * t;
        // Binom(j; k, share): the k extra neutrons of the precursor split between fragment and complement
        gains[0] += w * (1.0 - share).powi(k as i32);
        if k >= 1 {
            gains[1] += w * k as f32 * share * (1.0 - share).powi(k as i32 - 1);
        }
    }
    gains
}

/// Neighbour channels of a precursor and its expected neighbour / primary signal ratio (library-weighted).
///
/// `windows`: every MS2 observation as (index, isolation window); `primary`: the observations the search reads the
/// precursor from (its m/z inside the window).
#[allow(clippy::too_many_arguments)]
pub fn plan_channels(
    precursor_mz: f32,
    charge: u8,
    fragment_mz: &[f32],
    fragment_charge: &[u8],
    library_intensity: &[f32],
    windows: &[(usize, [f32; 2])],
    primary: &[usize],
) -> (Vec<NeighbourChannel>, f32) {
    if charge == 0 || primary.is_empty() {
        return (Vec::new(), 0.0);
    }
    let z = charge as f32;
    let mass = (precursor_mz - PROTON_MASS) * z;
    let abundance = isotope_abundances(mass);
    let isotope_mz: [f32; N_ISOTOPES] =
        std::array::from_fn(|k| precursor_mz + k as f32 * C13_MASS_DELTA / z);
    let reach = EDGE_REACH * EDGE_WIDTH;
    let (envelope_lo, envelope_hi) = (isotope_mz[0] - reach, isotope_mz[N_ISOTOPES - 1] + reach);
    let neighbours: Vec<&(usize, [f32; 2])> = windows
        .iter()
        .filter(|(obs, w)| {
            !primary.contains(obs) && w[1] >= 0.0 && w[1] >= envelope_lo && w[0] <= envelope_hi
        })
        .collect();

    let mut channels = Vec::new();
    let (mut expected, mut weight) = (0.0f32, 0.0f32);
    for f in 0..fragment_mz.len() {
        let fz = fragment_charge.get(f).copied().unwrap_or(1).max(1) as f32;
        let share = ((fragment_mz[f] - PROTON_MASS) * fz / mass).clamp(0.0, 1.0);
        let primary_gain: f32 = windows
            .iter()
            .filter(|(obs, _)| primary.contains(obs))
            .map(|(_, w)| window_gains(&isotope_mz, &abundance, share, *w)[0])
            .sum();
        if primary_gain <= EPSILON || library_intensity[f] <= 0.0 {
            continue;
        }
        weight += library_intensity[f] * primary_gain;
        for &&(obs, w) in &neighbours {
            let gains = window_gains(&isotope_mz, &abundance, share, w);
            for (j, &g) in gains.iter().enumerate() {
                expected += library_intensity[f] * g;
                let gain_ratio = g / primary_gain;
                if gain_ratio >= MIN_GAIN_RATIO {
                    channels.push(NeighbourChannel {
                        observation: obs,
                        fragment: f,
                        mz: fragment_mz[f] + j as f32 * C13_MASS_DELTA / fz,
                        gain_ratio,
                    });
                }
            }
        }
    }
    let expect = if weight > 0.0 { expected / weight } else { 0.0 };
    (channels, expect)
}

/// XICs [channel, cycle] of the neighbour channels over the cycles [start, stop).
pub fn channel_xic<T: DIADataTrait>(
    dia_data: &T,
    channels: &[NeighbourChannel],
    start: usize,
    stop: usize,
    mass_tolerance: f32,
) -> Array2<f32> {
    let mut xic = Array2::zeros((channels.len(), stop - start));
    for (c, channel) in channels.iter().enumerate() {
        dia_data.quadrupole_observations()[channel.observation].fill_xic_slice(
            dia_data.mz_index(),
            &mut xic.row_mut(c),
            start,
            stop,
            mass_tolerance,
            channel.mz,
        );
    }
    xic
}

fn clamp(p: f32) -> f32 {
    p.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}

/// Hit/miss log-likelihood ratio of the neighbour channels per signal scale of `detection::scales()`.
///
/// `channel_xic`: [channel, cycle] over the detection window with the apex in column `apex_col`.
fn neighbour_llr_by_scale(
    channel_xic: &Array2<f32>,
    apex_col: usize,
    channels: &[NeighbourChannel],
    library_intensity: &[f32],
    b0: f32,
) -> Vec<f32> {
    let top = library_intensity.iter().cloned().fold(0.0f32, f32::max);
    let ncols = channel_xic.ncols();
    let apex_lo = apex_col.saturating_sub(APEX_HALF_WIDTH);
    let apex_hi = (apex_col + APEX_HALF_WIDTH + 1).min(ncols);
    let flank: Vec<usize> = (0..ncols)
        .filter(|&c| (FLANK_MIN..=FLANK_MAX).contains(&c.abs_diff(apex_col)))
        .collect();
    let apex_scans = (apex_hi - apex_lo) as i32;
    let terms: Vec<(bool, f32, f32)> = channels
        .iter()
        .enumerate()
        .map(|(c, channel)| {
            let flank_hits = flank
                .iter()
                .filter(|&&col| channel_xic[[c, col]] > 0.0)
                .count();
            let per_scan =
                (flank_hits as f32 + PRIOR_WEIGHT * b0) / (flank.len() as f32 + PRIOR_WEIGHT);
            // a hit anywhere in apex +-1: the background rate over those scans
            let b = clamp(1.0 - (1.0 - clamp(per_scan)).powi(apex_scans));
            let hit = (apex_lo..apex_hi).any(|col| channel_xic[[c, col]] > 0.0);
            let relative =
                library_intensity[channel.fragment] / top.max(EPSILON) * channel.gain_ratio;
            (hit, b, relative)
        })
        .collect();
    scales()
        .iter()
        .map(|&scale| {
            terms
                .iter()
                .map(|&(hit, b, relative)| {
                    let p1 = clamp(1.0 - (1.0 - b) * (-scale * relative).exp());
                    if hit {
                        (p1 / b).ln()
                    } else {
                        ((1.0 - p1) / (1.0 - b)).ln()
                    }
                })
                .sum()
        })
        .collect()
}

/// Co-isolation features of one candidate.
///
/// `primary_xic`: the detection XIC of the primary window ([fragment, cycle], apex in column `apex_col`),
/// `channel_xic`: the neighbour channels over the same cycles, `expect`: from `plan_channels`, `b0`: the run's mean
/// flank hit rate.
pub fn coisolation_features(
    primary_xic: &Array2<f32>,
    channel_xic: &Array2<f32>,
    apex_col: usize,
    channels: &[NeighbourChannel],
    library_intensity: &[f32],
    expect: f32,
    b0: f32,
) -> CoisolationFeatures {
    let primary = detection::llr_by_scale(primary_xic, apex_col, library_intensity, b0);
    if channels.is_empty()
        || primary.is_empty()
        || channel_xic.nrows() != channels.len()
        || apex_col >= channel_xic.ncols()
    {
        return CoisolationFeatures {
            coiso_expect: expect,
            ..CoisolationFeatures::missing()
        };
    }
    let neighbour = neighbour_llr_by_scale(channel_xic, apex_col, channels, library_intensity, b0);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 {
                    (i, x)
                } else {
                    best
                }
            })
    };
    let (best_primary, primary_max) = argmax(&primary);
    let joint: Vec<f32> = primary.iter().zip(&neighbour).map(|(a, b)| a + b).collect();
    let (_, joint_max) = argmax(&joint);
    let (_, neighbour_max) = argmax(&neighbour);
    CoisolationFeatures {
        coiso_expect: expect,
        coiso_n_channels: channels.len() as f32,
        coiso_det_gain: joint_max - primary_max,
        coiso_det_nb_own: neighbour_max,
        coiso_det_nb_prim: neighbour[best_primary],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOWS: [(usize, [f32; 2]); 3] = [
        (0, [498.0, 500.0]),
        (1, [500.0, 502.0]),
        (2, [502.0, 504.0]),
    ];

    fn fragments() -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mz: Vec<f32> = (0..12).map(|f| 300.0 + 60.0 * f as f32).collect();
        let charge = vec![1u8; 12];
        let lib: Vec<f32> = (0..12).map(|f| 1.0 - f as f32 * 0.07).collect();
        (mz, charge, lib)
    }

    #[test]
    fn isotope_abundances_follow_the_peptide_mass() {
        let small = isotope_abundances(800.0);
        let large = isotope_abundances(3000.0);
        assert!((small.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(small[0] > small[1]);
        assert!(large[1] > 0.9 * large[0]);
        // averagine M+1 / M0 of a 1 kDa peptide is about 0.55
        let one_kda = isotope_abundances(1000.0);
        assert!((one_kda[1] / one_kda[0] - 0.54).abs() < 0.03);
    }

    #[test]
    fn transmission_is_one_inside_and_zero_far_outside() {
        assert!(transmission(501.0, 500.0, 502.0) > 0.99);
        assert!((transmission(500.0, 500.0, 502.0) - 0.5).abs() < 0.01);
        assert!(transmission(503.5, 500.0, 502.0) < 1e-3);
    }

    #[test]
    fn an_envelope_at_the_upper_edge_reaches_the_next_window_only() {
        let (mz, charge, lib) = fragments();
        let (channels, expect) = plan_channels(501.8, 2, &mz, &charge, &lib, &WINDOWS, &[1]);
        assert!(expect > 0.3);
        assert!(!channels.is_empty());
        assert!(channels.iter().all(|c| c.observation == 2));
        // heavy fragments carry the extra neutron: their M+1 channel is expected, light fragments read M0
        assert!(channels
            .iter()
            .any(|c| c.fragment == 0 && (c.mz - mz[0]).abs() < 1e-4));
        assert!(channels
            .iter()
            .any(|c| c.fragment == 11 && (c.mz - mz[11] - C13_MASS_DELTA).abs() < 1e-4));
    }

    #[test]
    fn an_envelope_in_the_window_centre_has_no_neighbour_channel() {
        let (mz, charge, lib) = fragments();
        let (channels, expect) = plan_channels(500.6, 4, &mz, &charge, &lib, &WINDOWS, &[1]);
        assert!(channels.is_empty());
        assert!(expect < MIN_GAIN_RATIO);
        let (none, zero) = plan_channels(501.8, 0, &mz, &charge, &lib, &WINDOWS, &[1]);
        assert!(none.is_empty());
        assert_eq!(zero, 0.0);
    }

    fn detection_xic(hits: &[usize]) -> Array2<f32> {
        let mut xic = Array2::zeros((12, 2 * FLANK_MAX + 1));
        for &f in hits {
            xic[[f, FLANK_MAX]] = 300.0;
        }
        xic
    }

    #[test]
    fn neighbour_hits_where_expected_raise_the_likelihood() {
        let (mz, charge, lib) = fragments();
        let (channels, expect) = plan_channels(501.8, 2, &mz, &charge, &lib, &WINDOWS, &[1]);
        let primary = detection_xic(&[0, 1, 2, 3, 4, 5]);
        let mut present = Array2::zeros((channels.len(), 2 * FLANK_MAX + 1));
        for (c, channel) in channels.iter().enumerate() {
            if channel.fragment < 6 {
                present[[c, FLANK_MAX]] = 100.0;
            }
        }
        let absent = Array2::zeros((channels.len(), 2 * FLANK_MAX + 1));
        let with =
            coisolation_features(&primary, &present, FLANK_MAX, &channels, &lib, expect, 0.09);
        let without =
            coisolation_features(&primary, &absent, FLANK_MAX, &channels, &lib, expect, 0.09);
        assert!(with.coiso_det_gain > 0.0);
        assert!(with.coiso_det_nb_prim > without.coiso_det_nb_prim);
        assert!(without.coiso_det_nb_prim < 0.0);
        assert!(with.coiso_det_nb_own > without.coiso_det_nb_own);
        assert_eq!(with.coiso_n_channels, channels.len() as f32);
        assert!((with.coiso_expect - expect).abs() < 1e-6);
    }

    #[test]
    fn no_channels_give_missing_values_with_the_geometry() {
        let (_, _, lib) = fragments();
        let primary = detection_xic(&[0, 1]);
        let f = coisolation_features(
            &primary,
            &Array2::zeros((0, 19)),
            FLANK_MAX,
            &[],
            &lib,
            0.05,
            0.09,
        );
        assert_eq!(f.coiso_det_gain, 0.0);
        assert_eq!(f.coiso_n_channels, 0.0);
        assert!((f.coiso_expect - 0.05).abs() < 1e-6);
    }
}
