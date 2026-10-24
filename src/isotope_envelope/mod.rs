//! Where the fragments of a precursor are expected in the raw data, given its isotope envelope.
//!
//! The quadrupole isolates every isotope of a precursor that falls into a window. With narrow windows
//! (2 Th on the Astral) the envelope of a doubly charged precursor (0.5 Th spacing) often reaches into the
//! next window, so the fragment signal is split between two windows.
//!
//! Fragments of a heavy precursor isotope carry its extra neutrons in part: each one sits in a fragment
//! with a probability equal to the fragment's share `f` of the precursor mass, so a fragment of precursor
//! isotope k is at fragment isotope j with probability Binom(j; k, f).
//!
//! Fragment isotope j of a fragment in window w therefore expects `E_wj = sum_k I_k T_w(m_k) Binom(j; k, f)`
//! of the precursor signal, with `I_k` the library isotope abundances and `T_w(m_k)` the transmission of
//! window w at the m/z of isotope k. Its gain `g_wj = E_wj / E_ref` compares it with the monoisotopic
//! fragment of a fully isolated envelope, `E_ref = sum_k I_k (1 - f)^k`, which is what a library
//! intensity describes.
//!
//! The fragments are extracted from one window only, the primary one, which transmits the largest share
//! of the envelope. Adding the signal of further windows to the same XIC gives decoys as many chance
//! matches as it gives real precursors signal (e2e on 200 ng HeLa: -2 % at matched entrapment FDR). Each
//! fragment is extracted at its most intense isotope in that window, and the library intensity is scaled
//! by its gain, so the library pattern matches what the window records. The window with the second
//! largest share is described as a secondary window for features of its own.

use numpy::ndarray::ArrayView4;

use crate::constants::{
    ISOTOPE_SPACING, MAX_FRAGMENT_ISOTOPE, PRIMARY_WINDOW_TOLERANCE, PROTON_MASS,
    QUADRUPOLE_EDGE_WIDTH, SECONDARY_MIN_SIGNAL_RATIO,
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
}

/// The extraction channels of one isolation window, one per fragment and observation of that window.
#[derive(Debug, Clone)]
pub struct WindowChannels {
    pub channels: Vec<FragmentChannel>,
    /// Expected signal of each fragment's channel relative to the library intensity
    pub fragment_gain: Vec<f32>,
    /// Share of the precursor's isotope envelope the window transmits
    pub precursor_share: f32,
}

impl WindowChannels {
    /// Library intensities scaled to what this window is expected to record.
    pub fn expected_intensity(&self, library_intensity: &[f32]) -> Vec<f32> {
        library_intensity
            .iter()
            .zip(&self.fragment_gain)
            .map(|(&intensity, &gain)| intensity * gain)
            .collect()
    }

    /// Observations used by at least one channel, ascending
    pub fn observation_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.channels.iter().map(|c| c.observation_idx).collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

/// The extraction channels of all fragments of a precursor.
#[derive(Debug, Clone)]
pub struct ExtractionPlan {
    pub primary: WindowChannels,
    /// The window with the second largest share of the envelope, if it holds at least
    /// `SECONDARY_MIN_SIGNAL_RATIO` of the primary window's share
    pub secondary: Option<WindowChannels>,
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
                    })
            })
            .collect();
        Self {
            primary: WindowChannels {
                channels,
                fragment_gain: vec![1.0; fragment_mz.len()],
                precursor_share: 1.0,
            },
            secondary: None,
        }
    }

    /// Fragments in the primary window of the isotope envelope, and in the secondary window if it holds enough
    /// of the envelope (see the module documentation).
    pub fn isotope_envelope<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        charge: u8,
        isotope_intensity: &[f32],
        fragment_mz: &[f32],
        fragment_charge: &[u8],
    ) -> Self {
        let isotope_mz = isotope_mz(precursor_mz, charge, isotope_intensity.len());
        let windows: Vec<[f32; 2]> = dia_data
            .quadrupole_observations()
            .iter()
            .map(|observation| observation.isolation_window())
            .collect();
        let shares = transmitted_shares(&isotope_mz, isotope_intensity, &windows);
        let Some((primary, secondary)) = rank_windows(&shares) else {
            return Self::monoisotopic(dia_data, precursor_mz, fragment_mz);
        };

        let precursor_mass = (precursor_mz - PROTON_MASS) * charge as f32;
        let window_channels = |observations: &[usize]| {
            let best = observations[0];
            let [lower, upper] = windows[best];
            let transmitted: Vec<f32> = isotope_mz
                .iter()
                .zip(isotope_intensity)
                .map(|(&mz, &intensity)| intensity * transmission(mz, lower, upper))
                .collect();
            let mut channels = Vec::with_capacity(observations.len() * fragment_mz.len());
            let mut fragment_gain = Vec::with_capacity(fragment_mz.len());
            for (fragment_idx, (&mz, &fragment_z)) in
                fragment_mz.iter().zip(fragment_charge).enumerate()
            {
                let fragment_z = fragment_z.max(1) as f32;
                let mass_share = ((mz - PROTON_MASS) * fragment_z / precursor_mass).clamp(0.0, 1.0);
                let (fragment_isotope, gain) =
                    strongest_fragment_isotope(&transmitted, isotope_intensity, mass_share);
                let mz_shift = fragment_isotope as f32 * ISOTOPE_SPACING / fragment_z;
                channels.extend(observations.iter().map(|&observation_idx| FragmentChannel {
                    fragment_idx,
                    observation_idx,
                    mz: mz + mz_shift,
                    mz_shift,
                }));
                fragment_gain.push(gain);
            }
            WindowChannels {
                channels,
                fragment_gain,
                precursor_share: shares[best],
            }
        };

        Self {
            primary: window_channels(&primary),
            secondary: secondary.map(|observations| window_channels(&observations)),
        }
    }
}

/// m/z of the first `n` isotopes of a precursor.
fn isotope_mz(precursor_mz: f32, charge: u8, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| precursor_mz + k as f32 * ISOTOPE_SPACING / charge as f32)
        .collect()
}

/// Share of the isotope envelope each window transmits; 0 for windows without a valid range or too far away.
fn transmitted_shares(
    isotope_mz: &[f32],
    isotope_intensity: &[f32],
    windows: &[[f32; 2]],
) -> Vec<f32> {
    let total: f32 = isotope_intensity.iter().sum();
    // Beyond ten edge widths from a window the transmission is below e^-10.
    let reach = 10.0 * QUADRUPOLE_EDGE_WIDTH;
    let envelope_lower = isotope_mz[0] - reach;
    let envelope_upper = isotope_mz[isotope_mz.len() - 1] + reach;
    windows
        .iter()
        .map(|&[lower, upper]| {
            if upper <= lower || upper < envelope_lower || lower > envelope_upper {
                return 0.0;
            }
            isotope_mz
                .iter()
                .zip(isotope_intensity)
                .map(|(&mz, &intensity)| intensity * transmission(mz, lower, upper))
                .sum::<f32>()
                / total
        })
        .collect()
}

/// The primary windows, those within `PRIMARY_WINDOW_TOLERANCE` of the largest share, best first, and the
/// secondary windows ranked the same way among the rest, if they reach `SECONDARY_MIN_SIGNAL_RATIO` of the
/// primary share. Overlapping windows can hold the same share, so a rank can have several windows.
/// None if no window transmits the envelope.
fn rank_windows(shares: &[f32]) -> Option<(Vec<usize>, Option<Vec<usize>>)> {
    let mut order: Vec<usize> = (0..shares.len()).filter(|&w| shares[w] > 0.0).collect();
    order.sort_by(|&a, &b| shares[b].total_cmp(&shares[a]).then(a.cmp(&b)));
    let best = *order.first()?;
    let primary_floor = shares[best] * (1.0 - PRIMARY_WINDOW_TOLERANCE);
    let (primary, rest): (Vec<usize>, Vec<usize>) =
        order.into_iter().partition(|&w| shares[w] >= primary_floor);

    let secondary = rest.first().and_then(|&second| {
        if shares[second] < shares[best] * SECONDARY_MIN_SIGNAL_RATIO {
            return None;
        }
        let floor = shares[second] * (1.0 - PRIMARY_WINDOW_TOLERANCE);
        Some(
            rest.iter()
                .copied()
                .filter(|&w| shares[w] >= floor)
                .collect(),
        )
    });
    Some((primary, secondary))
}

/// The fragment isotope with the largest gain in a window, and that gain, given the window's transmitted
/// isotope abundances. The monoisotopic fragment wins ties.
fn strongest_fragment_isotope(
    transmitted: &[f32],
    isotope_intensity: &[f32],
    mass_share: f32,
) -> (usize, f32) {
    let reference: f32 = isotope_intensity
        .iter()
        .enumerate()
        .map(|(k, &intensity)| intensity * binomial(k, 0, mass_share))
        .sum();
    if reference <= 0.0 {
        return (0, 1.0);
    }
    (0..=MAX_FRAGMENT_ISOTOPE)
        .map(|fragment_isotope| {
            let expected: f32 = transmitted
                .iter()
                .enumerate()
                .skip(fragment_isotope)
                .map(|(k, &abundance)| abundance * binomial(k, fragment_isotope, mass_share))
                .sum();
            (fragment_isotope, expected / reference)
        })
        .fold((0, f32::NEG_INFINITY), |best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
}

/// Lower and upper m/z of each window of a DIA cycle of shape `(1, n_windows, n_scans, 2)`, over all scans.
pub fn cycle_windows(cycle: ArrayView4<'_, f32>) -> Vec<[f32; 2]> {
    let shape = cycle.shape();
    (0..shape[1])
        .map(|window| {
            (0..shape[2]).fold(
                [f32::INFINITY, f32::NEG_INFINITY],
                |[lower, upper], scan| {
                    [
                        lower.min(cycle[[0, window, scan, 0]]),
                        upper.max(cycle[[0, window, scan, 1]]),
                    ]
                },
            )
        })
        .collect()
}

/// An m/z inside the primary window of the precursor's isotope envelope: the centre of that window, or
/// `precursor_mz` if the isotopes are unusable or no window transmits the envelope.
pub fn primary_window_mz(
    precursor_mz: f32,
    charge: u8,
    isotope_intensity: &[f32],
    windows: &[[f32; 2]],
) -> f32 {
    let isotopes_usable = isotope_intensity.iter().all(|x| x.is_finite())
        && isotope_intensity.iter().sum::<f32>() > 0.0;
    if charge == 0 || !isotopes_usable {
        return precursor_mz;
    }
    let isotope_mz = isotope_mz(precursor_mz, charge, isotope_intensity.len());
    let shares = transmitted_shares(&isotope_mz, isotope_intensity, windows);
    match rank_windows(&shares) {
        Some((primary, _)) => {
            let [lower, upper] = windows[primary[0]];
            0.5 * (lower + upper)
        }
        None => precursor_mz,
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
