//! Where the fragments of a precursor are expected in the raw data, given its isotope envelope.
//!
//! The quadrupole isolates every isotope of a precursor that falls into a window. With narrow windows
//! (2 Th on the Astral) the envelope of a doubly charged precursor (0.5 Th spacing) often reaches into the
//! next window, so part of the fragment signal is recorded there.
//!
//! Fragments of a heavy precursor isotope carry its extra neutrons in part: each one sits in a fragment
//! with a probability equal to the fragment's share `f` of the precursor mass, so a fragment of precursor
//! isotope k is at fragment isotope j with probability Binom(j; k, f).
//!
//! An extraction channel (window w, fragment isotope j) of a fragment therefore expects
//! `E_wj = sum_k I_k T_w(m_k) Binom(j; k, f)` of the precursor signal, with `I_k` the library isotope
//! abundances and `T_w(m_k)` the transmission of window w at the m/z of isotope k. Its gain
//! `g_wj = E_wj / E_ref` compares it with the monoisotopic fragment of a fully isolated envelope,
//! `E_ref = sum_k I_k (1 - f)^k`, which is what a library intensity describes.
//! With channels `x_wj = s g_wj + noise`, the least-squares estimate of the fragment signal is
//! `s = sum g_wj x_wj / sum g_wj^2`; each channel's weight is its term of that sum.
//! An envelope inside one window, extracted at the monoisotopic fragment only, gives g_w0 = 1 and the
//! plain monoisotopic XIC.

use crate::constants::{
    ISOTOPE_SPACING, MAX_FRAGMENT_ISOTOPE, MIN_CHANNEL_GAIN, PROTON_MASS, QUADRUPOLE_EDGE_WIDTH,
};
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};

#[cfg(test)]
mod tests;

/// One fragment m/z to extract from one quadrupole observation.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentChannel {
    pub fragment_idx: usize,
    pub observation_idx: usize,
    /// m/z to extract
    pub mz: f32,
    /// Offset of `mz` from the monoisotopic fragment m/z
    pub mz_shift: f32,
    /// Expected signal relative to the monoisotopic fragment of a fully isolated envelope
    pub gain: f32,
    /// Factor applied to the extracted intensity before it is added to the fragment's XIC
    pub weight: f32,
}

/// The extraction channels of all fragments of a precursor.
#[derive(Debug, Clone)]
pub struct ExtractionPlan {
    pub num_fragments: usize,
    pub channels: Vec<FragmentChannel>,
    /// Observations used by at least one channel, ascending
    pub observation_indices: Vec<usize>,
}

impl ExtractionPlan {
    /// The isotope-envelope plan if the precursor carries usable library isotopes, otherwise the monoisotopic one.
    pub fn new<T: DIADataTrait>(dia_data: &T, precursor: &Precursor) -> Self {
        let isotopes_usable = precursor.isotope_intensity.iter().all(|x| x.is_finite())
            && precursor.isotope_intensity.iter().sum::<f32>() > 0.0;
        if precursor.charge == 0 || !isotopes_usable {
            return Self::monoisotopic(dia_data, precursor.mz, &precursor.fragment_mz);
        }
        Self::isotope_envelope(
            dia_data,
            precursor.mz,
            precursor.charge,
            &precursor.isotope_intensity,
            &precursor.fragment_mz,
            &precursor.fragment_charge,
        )
    }

    /// Monoisotopic fragments, summed over every window that contains the precursor m/z.
    pub fn monoisotopic<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        fragment_mz: &[f32],
    ) -> Self {
        let observation_indices = dia_data.get_valid_observations(precursor_mz);
        let channels = observation_indices
            .iter()
            .flat_map(|&observation_idx| {
                fragment_mz
                    .iter()
                    .enumerate()
                    .map(move |(fragment_idx, &mz)| FragmentChannel {
                        fragment_idx,
                        observation_idx,
                        mz,
                        mz_shift: 0.0,
                        gain: 1.0,
                        weight: 1.0,
                    })
            })
            .collect();
        Self {
            num_fragments: fragment_mz.len(),
            channels,
            observation_indices,
        }
    }

    /// Fragment isotopes up to `MAX_FRAGMENT_ISOTOPE` in every window the isotope envelope reaches,
    /// weighted by their expected share of the fragment signal (see the module documentation).
    pub fn isotope_envelope<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        charge: u8,
        isotope_intensity: &[f32],
        fragment_mz: &[f32],
        fragment_charge: &[u8],
    ) -> Self {
        let isotope_mz: Vec<f32> = (0..isotope_intensity.len())
            .map(|k| precursor_mz + k as f32 * ISOTOPE_SPACING / charge as f32)
            .collect();

        // Beyond ten edge widths from a window the transmission is below e^-10.
        let reach = 10.0 * QUADRUPOLE_EDGE_WIDTH;
        let envelope_lower = isotope_mz[0] - reach;
        let envelope_upper = isotope_mz[isotope_mz.len() - 1] + reach;

        // (observation, transmitted abundance I_k T_w(m_k) per precursor isotope)
        let windows: Vec<(usize, Vec<f32>)> = dia_data
            .quadrupole_observations()
            .iter()
            .enumerate()
            .filter_map(|(observation_idx, observation)| {
                let [lower, upper] = observation.isolation_window();
                if upper <= lower || upper < envelope_lower || lower > envelope_upper {
                    return None;
                }
                let transmitted = isotope_mz
                    .iter()
                    .zip(isotope_intensity)
                    .map(|(&mz, &intensity)| intensity * transmission(mz, lower, upper))
                    .collect();
                Some((observation_idx, transmitted))
            })
            .collect();

        let precursor_mass = (precursor_mz - PROTON_MASS) * charge as f32;
        let mut channels = Vec::new();

        for (fragment_idx, (&mz, &fragment_z)) in
            fragment_mz.iter().zip(fragment_charge).enumerate()
        {
            let fragment_z = fragment_z.max(1) as f32;
            let mass_share = ((mz - PROTON_MASS) * fragment_z / precursor_mass).clamp(0.0, 1.0);
            let reference: f32 = isotope_intensity
                .iter()
                .enumerate()
                .map(|(k, &intensity)| intensity * binomial(k, 0, mass_share))
                .sum();
            if reference <= 0.0 {
                continue;
            }

            let first_channel = channels.len();
            let mut sum_squared_gain = 0.0;
            for (observation_idx, transmitted) in &windows {
                for fragment_isotope in 0..=MAX_FRAGMENT_ISOTOPE {
                    let expected: f32 = transmitted
                        .iter()
                        .enumerate()
                        .skip(fragment_isotope)
                        .map(|(k, &abundance)| {
                            abundance * binomial(k, fragment_isotope, mass_share)
                        })
                        .sum();
                    let gain = expected / reference;
                    if gain < MIN_CHANNEL_GAIN {
                        continue;
                    }
                    let mz_shift = fragment_isotope as f32 * ISOTOPE_SPACING / fragment_z;
                    channels.push(FragmentChannel {
                        fragment_idx,
                        observation_idx: *observation_idx,
                        mz: mz + mz_shift,
                        mz_shift,
                        gain,
                        weight: gain,
                    });
                    sum_squared_gain += gain * gain;
                }
            }
            for channel in &mut channels[first_channel..] {
                channel.weight /= sum_squared_gain;
            }
        }

        let mut observation_indices: Vec<usize> = channels
            .iter()
            .map(|channel| channel.observation_idx)
            .collect();
        observation_indices.sort_unstable();
        observation_indices.dedup();

        Self {
            num_fragments: fragment_mz.len(),
            channels,
            observation_indices,
        }
    }
}

/// Quadrupole transmission at `mz` for the window [lower, upper]: a rectangle with logistic edges.
pub fn transmission(mz: f32, lower: f32, upper: f32) -> f32 {
    let logistic = |x: f32| 1.0 / (1.0 + (-x / QUADRUPOLE_EDGE_WIDTH).exp());
    (logistic(mz - lower) - logistic(mz - upper)).max(0.0)
}

/// Probability that j of k extra neutrons sit in a fragment holding the share p of the precursor mass.
fn binomial(k: usize, j: usize, p: f32) -> f32 {
    if j > k {
        return 0.0;
    }
    let coefficient = (0..j).fold(1.0, |acc, i| acc * (k - i) as f32 / (i + 1) as f32);
    coefficient * p.powi(j as i32) * (1.0 - p).powi((k - j) as i32)
}
