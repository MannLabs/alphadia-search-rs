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
//! fragment of a fully isolated envelope, `E_ref = sum_k I_k (1 - f)^k`.
//!
//! Each fragment is extracted from the one channel (window, fragment isotope) that expects the most signal.
//! Summing the channels of several windows into one XIC gives decoys as many chance matches as it gives
//! real precursors signal (e2e on 200 ng HeLa: -2 % at matched entrapment FDR), and reading every fragment
//! from the window with the largest share of the envelope picks a weaker channel for long fragments, whose
//! signal that window splits over their isotopes (local A/B: -8 % where the window moved). The library
//! intensity of a fragment is scaled by the gain of its channel relative to its monoisotopic channel in the
//! window of the precursor m/z, so a fragment that stays there keeps the library intensity, which that
//! channel was found to match better than `E_wj` does. The best channel of each fragment in another window
//! is described as a secondary channel for features of its own.

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

/// One extraction channel per fragment of a precursor, or several for overlapping windows.
#[derive(Debug, Clone)]
pub struct ChannelSet {
    pub channels: Vec<FragmentChannel>,
    /// Expected signal of each fragment's channel relative to its monoisotopic channel in the window of
    /// the precursor m/z
    pub fragment_gain: Vec<f32>,
}

impl ChannelSet {
    fn with_capacity(n_fragments: usize) -> Self {
        Self {
            channels: Vec::with_capacity(n_fragments),
            fragment_gain: Vec::with_capacity(n_fragments),
        }
    }

    /// Library intensities scaled to what the channels are expected to record.
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

    fn push(&mut self, fragment_idx: usize, mz: f32, channel: &BestChannel, fragment_z: f32) {
        let mz_shift = channel.isotope as f32 * ISOTOPE_SPACING / fragment_z;
        self.channels.extend(
            channel
                .windows
                .iter()
                .map(|&observation_idx| FragmentChannel {
                    fragment_idx,
                    observation_idx,
                    mz: mz + mz_shift,
                    mz_shift,
                }),
        );
        self.fragment_gain.push(channel.gain);
    }
}

/// The extraction channels of all fragments of a precursor.
#[derive(Debug, Clone)]
pub struct ExtractionPlan {
    /// The channel of each fragment that expects the most signal
    pub primary: ChannelSet,
    /// The best channel of each fragment outside its primary windows, if together they expect at least
    /// `SECONDARY_MIN_SIGNAL_RATIO` of the library-weighted signal of the primary channels
    pub secondary: Option<ChannelSet>,
}

impl ExtractionPlan {
    /// The isotope-envelope plan if the precursor carries usable library isotopes, otherwise the monoisotopic one.
    pub fn new<T: DIADataTrait>(dia_data: &T, precursor: &Precursor) -> Self {
        if precursor.charge == 0 || !isotopes_usable(&precursor.isotope_intensity) {
            return Self::monoisotopic(dia_data, precursor.mz, &precursor.fragment_mz);
        }
        Self::isotope_envelope(
            dia_data,
            precursor.mz,
            precursor.charge,
            &precursor.isotope_intensity,
            &precursor.fragment_mz,
            &precursor.fragment_charge,
            &precursor.fragment_intensity,
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
            primary: ChannelSet {
                channels,
                fragment_gain: vec![1.0; fragment_mz.len()],
            },
            secondary: None,
        }
    }

    /// The best channel of every fragment over the windows the isotope envelope reaches, and its best
    /// channel elsewhere as the secondary one (see the module documentation).
    pub fn isotope_envelope<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        charge: u8,
        isotope_intensity: &[f32],
        fragment_mz: &[f32],
        fragment_charge: &[u8],
        fragment_intensity: &[f32],
    ) -> Self {
        let windows: Vec<[f32; 2]> = dia_data
            .quadrupole_observations()
            .iter()
            .map(|observation| observation.isolation_window())
            .collect();
        let Some(gains) = ChannelGains::new(
            precursor_mz,
            charge,
            isotope_intensity,
            fragment_mz,
            fragment_charge,
            &windows,
        ) else {
            return Self::monoisotopic(dia_data, precursor_mz, fragment_mz);
        };
        let Some(reference) = gains.reference_gain(precursor_mz, &windows) else {
            return Self::monoisotopic(dia_data, precursor_mz, fragment_mz);
        };

        let mut primary = ChannelSet::with_capacity(fragment_mz.len());
        let mut secondary = ChannelSet::with_capacity(fragment_mz.len());
        for (fragment_idx, (&mz, &fragment_z)) in
            fragment_mz.iter().zip(fragment_charge).enumerate()
        {
            let fragment_z = fragment_z.max(1) as f32;
            let mut best = gains.best_channel(fragment_idx, &[]);
            let mut other = gains.best_channel(fragment_idx, &best.windows);
            best.gain /= reference[fragment_idx];
            other.gain /= reference[fragment_idx];
            primary.push(fragment_idx, mz, &best, fragment_z);
            secondary.push(fragment_idx, mz, &other, fragment_z);
        }

        let weighted_signal = |set: &ChannelSet| {
            set.expected_intensity(fragment_intensity)
                .iter()
                .sum::<f32>()
        };
        let primary_signal = weighted_signal(&primary);
        let secondary_signal = weighted_signal(&secondary);
        let secondary = (secondary_signal > 0.0
            && secondary_signal >= SECONDARY_MIN_SIGNAL_RATIO * primary_signal)
            .then_some(secondary);
        Self { primary, secondary }
    }
}

/// The strongest channel of one fragment: its windows, those within `PRIMARY_WINDOW_TOLERANCE` of the
/// largest gain at the best fragment isotope (overlapping windows), that isotope and the summed gain.
struct BestChannel {
    windows: Vec<usize>,
    isotope: usize,
    gain: f32,
}

/// The gain of every channel (window, fragment isotope) of a precursor's fragments.
struct ChannelGains {
    /// Observation index of each window that transmits part of the envelope
    windows: Vec<usize>,
    /// `gain[fragment][window position][fragment isotope]`
    gain: Vec<Vec<[f32; MAX_FRAGMENT_ISOTOPE + 1]>>,
}

impl ChannelGains {
    /// None if no window transmits the envelope.
    fn new(
        precursor_mz: f32,
        charge: u8,
        isotope_intensity: &[f32],
        fragment_mz: &[f32],
        fragment_charge: &[u8],
        windows: &[[f32; 2]],
    ) -> Option<Self> {
        let isotope_mz = isotope_mz(precursor_mz, charge, isotope_intensity.len());
        let shares = transmitted_shares(&isotope_mz, isotope_intensity, windows);
        let reached: Vec<usize> = (0..windows.len()).filter(|&w| shares[w] > 0.0).collect();
        if reached.is_empty() {
            return None;
        }
        let transmitted: Vec<Vec<f32>> = reached
            .iter()
            .map(|&w| {
                let [lower, upper] = windows[w];
                isotope_mz
                    .iter()
                    .zip(isotope_intensity)
                    .map(|(&mz, &intensity)| intensity * transmission(mz, lower, upper))
                    .collect()
            })
            .collect();

        let precursor_mass = (precursor_mz - PROTON_MASS) * charge as f32;
        let gain = fragment_mz
            .iter()
            .zip(fragment_charge)
            .map(|(&mz, &fragment_z)| {
                let fragment_z = fragment_z.max(1) as f32;
                let mass_share = ((mz - PROTON_MASS) * fragment_z / precursor_mass).clamp(0.0, 1.0);
                let reference: f32 = isotope_intensity
                    .iter()
                    .enumerate()
                    .map(|(k, &intensity)| intensity * binomial(k, 0, mass_share))
                    .sum();
                transmitted
                    .iter()
                    .map(|abundances| {
                        let mut gains = [0.0; MAX_FRAGMENT_ISOTOPE + 1];
                        if reference > 0.0 {
                            for (fragment_isotope, gain) in gains.iter_mut().enumerate() {
                                *gain = abundances
                                    .iter()
                                    .enumerate()
                                    .skip(fragment_isotope)
                                    .map(|(k, &abundance)| {
                                        abundance * binomial(k, fragment_isotope, mass_share)
                                    })
                                    .sum::<f32>()
                                    / reference;
                            }
                        }
                        gains
                    })
                    .collect()
            })
            .collect();
        Some(Self {
            windows: reached,
            gain,
        })
    }

    /// The gain of each fragment's monoisotopic channel in the windows containing `precursor_mz`, which the
    /// monoisotopic plan reads. None if there is no such window or a fragment expects nothing there.
    fn reference_gain(&self, precursor_mz: f32, windows: &[[f32; 2]]) -> Option<Vec<f32>> {
        let own: Vec<usize> = (0..self.windows.len())
            .filter(|&position| {
                let [lower, upper] = windows[self.windows[position]];
                lower <= precursor_mz && precursor_mz <= upper
            })
            .collect();
        let reference: Vec<f32> = self
            .gain
            .iter()
            .map(|gains| own.iter().map(|&position| gains[position][0]).sum())
            .collect();
        (!own.is_empty() && reference.iter().all(|&gain| gain > 0.0)).then_some(reference)
    }

    /// The strongest channel of a fragment outside the `excluded` windows; no windows and gain 0 if none
    /// expects signal. The lower fragment isotope and the earlier window win ties.
    fn best_channel(&self, fragment_idx: usize, excluded: &[usize]) -> BestChannel {
        let gains = &self.gain[fragment_idx];
        let allowed = |position: &usize| !excluded.contains(&self.windows[*position]);
        let (isotope, top) = (0..self.windows.len())
            .filter(allowed)
            .flat_map(|position| {
                (0..=MAX_FRAGMENT_ISOTOPE).map(move |isotope| (isotope, gains[position][isotope]))
            })
            .fold((0, 0.0f32), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            });
        if top <= 0.0 {
            return BestChannel {
                windows: Vec::new(),
                isotope: 0,
                gain: 0.0,
            };
        }
        let floor = top * (1.0 - PRIMARY_WINDOW_TOLERANCE);
        let positions: Vec<usize> = (0..self.windows.len())
            .filter(allowed)
            .filter(|&position| gains[position][isotope] >= floor)
            .collect();
        BestChannel {
            windows: positions
                .iter()
                .map(|&position| self.windows[position])
                .collect(),
            isotope,
            gain: positions
                .iter()
                .map(|&position| gains[position][isotope])
                .sum(),
        }
    }

    /// The window that expects the most library-weighted signal when every fragment reads its best channel
    /// there.
    fn strongest_window(&self, library_intensity: &[f32]) -> usize {
        let signal = |position: usize| -> f32 {
            self.gain
                .iter()
                .zip(library_intensity)
                .map(|(gains, &intensity)| {
                    intensity * gains[position].iter().cloned().fold(0.0, f32::max)
                })
                .sum()
        };
        let best = (0..self.windows.len())
            .fold((0, f32::NEG_INFINITY), |best, position| {
                let candidate = signal(position);
                if candidate > best.1 {
                    (position, candidate)
                } else {
                    best
                }
            })
            .0;
        self.windows[best]
    }
}

fn isotopes_usable(isotope_intensity: &[f32]) -> bool {
    isotope_intensity.iter().all(|x| x.is_finite()) && isotope_intensity.iter().sum::<f32>() > 0.0
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

/// An m/z inside the window that expects the most library-weighted fragment signal of the precursor, the
/// one most of its fragments are extracted from: the centre of that window, or `precursor_mz` if the
/// isotopes are unusable or the precursor falls back to the monoisotopic extraction.
pub fn primary_window_mz(
    precursor_mz: f32,
    charge: u8,
    isotope_intensity: &[f32],
    fragment_mz: &[f32],
    fragment_charge: &[u8],
    fragment_intensity: &[f32],
    windows: &[[f32; 2]],
) -> f32 {
    if charge == 0 || !isotopes_usable(isotope_intensity) {
        return precursor_mz;
    }
    let Some(gains) = ChannelGains::new(
        precursor_mz,
        charge,
        isotope_intensity,
        fragment_mz,
        fragment_charge,
        windows,
    ) else {
        return precursor_mz;
    };
    if gains.reference_gain(precursor_mz, windows).is_none() {
        return precursor_mz;
    }
    let [lower, upper] = windows[gains.strongest_window(fragment_intensity)];
    0.5 * (lower + upper)
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
