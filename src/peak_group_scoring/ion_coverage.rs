//! Which of a candidate's fragments co-elute with its peak, and which ions they are.
//!
//! Short ions (b2/b3, y2/y3) match by chance in crowded spectra, while a real peptide also lights up its long ions.
//! Summaries over all fragments (counts, mean correlations) lose which ions carry the signal; these features keep it.
//!
//! A decoy's fragments are its target's (copied intensities, ion types and numbers) at the m/z of its own sequence,
//! so every feature measures the evidence of a target and of its decoy the same way.
//!
//! `ion_fisher_lib` needs a noise model fitted over all scored candidates (`IonNoiseModel`); the other features
//! depend on the candidate alone.

use rayon::prelude::*;

use crate::constants::FragmentType;
use crate::peak_group_scoring::utils::{shares, COELUTION_MIN_CORRELATION};

/// Fragments with at most this ion number are short ions
pub const SHORT_ION_MAX_NUMBER: u8 = 3;
/// Fragments with at least this ion number are long ions; the ones between are mid ions
pub const LONG_ION_MIN_NUMBER: u8 = 8;
/// Fragment charge of the *_z2 features
const Z2_FRAGMENT_CHARGE: u8 = 2;

/// Ion-number bucket edges of the noise model's ion classes (bucket = number of edges <= ion number). They are finer
/// than short / mid / long, because the chance of a hit changes most among the short ions.
const NOISE_ION_NUMBER_EDGES: [u8; 3] = [3, 5, 8];
/// Library intensity ranks per rank group of the noise model's ion classes
const INTENSITY_RANK_GROUP: usize = 3;
/// Fragment charges above this share the noise model's ion class of this charge
const NOISE_MAX_FRAGMENT_CHARGE: u8 = 2;
/// Correlation histogram bins of the noise model on [0, 1]
const CORRELATION_BINS: usize = 100;
/// The chance of a fragment hit is floored here: one fragment adds at most -ln(MIN_CHANCE) surprise
const MIN_CHANCE: f64 = 1e-4;
/// Number of ion kinds: b, y, other
const ION_KINDS: usize = 3;
const BASE_CLASSES: usize =
    ION_KINDS * (NOISE_ION_NUMBER_EDGES.len() + 1) * (NOISE_MAX_FRAGMENT_CHARGE as usize + 1);

enum IonLength {
    Short,
    Mid,
    Long,
}

fn ion_length(number: u8) -> IonLength {
    if number <= SHORT_ION_MAX_NUMBER {
        IonLength::Short
    } else if number >= LONG_ION_MIN_NUMBER {
        IonLength::Long
    } else {
        IonLength::Mid
    }
}

/// The ion-coverage features of one candidate that depend on the candidate alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IonCoverageFeatures {
    /// mean ion number relative to the peptide length over the co-eluting fragments
    pub ion_position_mean: f32,
    /// share of the observed fragment intensity in short ions
    pub short_ion_intensity_share: f32,
    /// co-eluting b and y ions, short, mid and long; mid b ions are left out
    pub num_over_50_b_short: f32,
    pub num_over_50_b_long: f32,
    pub num_over_50_y_short: f32,
    pub num_over_50_y_mid: f32,
    pub num_over_50_y_long: f32,
    /// co-eluting doubly charged fragments
    pub num_over_50_z2: f32,
    /// share of the library intensity in co-eluting fragments
    pub lib_share_over_50: f32,
    /// fragment correlations averaged with the library intensity shares as weights
    pub xic_libweighted: f32,
    /// fragment correlations averaged with the observed intensity shares as weights
    pub xic_obsweighted: f32,
}

/// What the noise model needs of one scored fragment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonEvidence {
    /// ion class: ion kind x ion-number bucket x fragment charge x library intensity-rank group
    pub class: u16,
    /// correlation histogram bin, meaningful if `positive`
    pub bin: u8,
    /// observed intensity > 0
    pub hit: bool,
    /// hit with correlation > 0
    pub positive: bool,
    /// share of the candidate's library intensity
    pub library_share: f32,
}

/// Number of ion classes for candidates scored with up to `top_k_fragments` fragments.
pub fn num_ion_classes(top_k_fragments: usize) -> usize {
    BASE_CLASSES * top_k_fragments.div_ceil(INTENSITY_RANK_GROUP).max(1)
}

fn number_bucket(number: u8, edges: &[u8]) -> usize {
    edges.iter().filter(|&&edge| edge <= number).count()
}

/// Bin of `correlation` on CORRELATION_BINS equal bins of [0, 1], the last one closed.
fn correlation_bin(correlation: f64) -> u8 {
    let edge = |i: usize| i as f64 / CORRELATION_BINS as f64;
    let mut bin =
        ((correlation * CORRELATION_BINS as f64).floor().max(0.0) as usize).min(CORRELATION_BINS);
    // the floor can be one off at the edges
    while bin < CORRELATION_BINS && edge(bin + 1) <= correlation {
        bin += 1;
    }
    while bin > 0 && edge(bin) > correlation {
        bin -= 1;
    }
    bin.min(CORRELATION_BINS - 1) as u8
}

fn ion_class(
    fragment_type: u8,
    number: u8,
    charge: u8,
    intensity_rank: usize,
    n_rank_groups: usize,
) -> u16 {
    let kind = match fragment_type {
        FragmentType::B => 0,
        FragmentType::Y => 1,
        _ => 2,
    };
    let bucket = number_bucket(number, &NOISE_ION_NUMBER_EDGES);
    let charge = charge.min(NOISE_MAX_FRAGMENT_CHARGE) as usize;
    let base = (kind * (NOISE_ION_NUMBER_EDGES.len() + 1) + bucket)
        * (NOISE_MAX_FRAGMENT_CHARGE as usize + 1)
        + charge;
    (base * n_rank_groups + intensity_rank / INTENSITY_RANK_GROUP) as u16
}

/// The ion-coverage features of one candidate and the evidence of its fragments for the noise model.
///
/// The slices hold one value per scored fragment: library intensity, observed intensity (summed over the
/// candidate's cycles), correlation with the candidate's median profile, ion type, ion number and fragment charge.
/// `naa` is the peptide length. `top_k_fragments` is the maximum number of scored fragments per candidate, at least
/// the length of the slices.
#[allow(clippy::too_many_arguments)]
pub fn calculate_ion_coverage_features(
    library_intensities: &[f32],
    observation_intensities: &[f32],
    correlations: &[f32],
    fragment_types: &[u8],
    fragment_numbers: &[u8],
    fragment_charges: &[u8],
    naa: u8,
    top_k_fragments: usize,
) -> (IonCoverageFeatures, Vec<IonEvidence>) {
    let n = library_intensities.len();
    if n == 0
        || observation_intensities.len() != n
        || correlations.len() != n
        || fragment_types.len() != n
        || fragment_numbers.len() != n
        || fragment_charges.len() != n
    {
        return (IonCoverageFeatures::default(), Vec::new());
    }
    let coeluting: Vec<bool> = correlations
        .iter()
        .map(|&c| c > COELUTION_MIN_CORRELATION)
        .collect();
    let mut features = IonCoverageFeatures::default();

    let mut position_sum = 0.0f32;
    let mut n_coeluting = 0u32;
    let mut short_intensity = 0.0f32;
    let mut total_intensity = 0.0f32;
    for i in 0..n {
        let length = ion_length(fragment_numbers[i]);
        total_intensity += observation_intensities[i];
        if matches!(length, IonLength::Short) {
            short_intensity += observation_intensities[i];
        }
        if !coeluting[i] {
            continue;
        }
        if naa > 0 {
            position_sum += fragment_numbers[i] as f32 / naa as f32;
        }
        n_coeluting += 1;
        if fragment_charges[i] == Z2_FRAGMENT_CHARGE {
            features.num_over_50_z2 += 1.0;
        }
        let count = match (fragment_types[i], length) {
            (FragmentType::B, IonLength::Short) => &mut features.num_over_50_b_short,
            (FragmentType::B, IonLength::Mid) => continue,
            (FragmentType::B, IonLength::Long) => &mut features.num_over_50_b_long,
            (FragmentType::Y, IonLength::Short) => &mut features.num_over_50_y_short,
            (FragmentType::Y, IonLength::Mid) => &mut features.num_over_50_y_mid,
            (FragmentType::Y, IonLength::Long) => &mut features.num_over_50_y_long,
            _ => continue,
        };
        *count += 1.0;
    }
    if n_coeluting > 0 {
        features.ion_position_mean = position_sum / n_coeluting as f32;
    }
    if total_intensity > 0.0 {
        features.short_ion_intensity_share = short_intensity / total_intensity;
    }

    let library_total: f32 = library_intensities.iter().sum();
    let library_coeluting: f32 = (0..n)
        .filter(|&i| coeluting[i])
        .map(|i| library_intensities[i])
        .sum();
    if library_total > 0.0 {
        features.lib_share_over_50 = library_coeluting / library_total;
    }

    let library: Vec<f64> = library_intensities.iter().map(|&x| x as f64).collect();
    let observed: Vec<f64> = observation_intensities.iter().map(|&x| x as f64).collect();
    let correlation: Vec<f64> = correlations.iter().map(|&x| x as f64).collect();
    let library_share = shares(&library);
    let observed_share = shares(&observed);
    features.xic_libweighted = (0..n)
        .map(|i| library_share[i] * correlation[i])
        .sum::<f64>() as f32;
    features.xic_obsweighted = (0..n)
        .map(|i| observed_share[i] * correlation[i])
        .sum::<f64>() as f32;

    // library intensity ranks: fragments by descending library intensity, ties in their scored order
    let mut by_intensity: Vec<usize> = (0..n).collect();
    by_intensity.sort_by(|&a, &b| library[b].total_cmp(&library[a]));
    let n_rank_groups = top_k_fragments.div_ceil(INTENSITY_RANK_GROUP).max(1);
    let mut evidence: Vec<IonEvidence> = Vec::with_capacity(n);
    for (intensity_rank, &i) in by_intensity.iter().enumerate() {
        let hit = observed[i] > 0.0;
        evidence.push(IonEvidence {
            class: ion_class(
                fragment_types[i],
                fragment_numbers[i],
                fragment_charges[i],
                intensity_rank,
                n_rank_groups,
            ),
            bin: correlation_bin(correlation[i]),
            hit,
            positive: hit && correlation[i] > 0.0,
            library_share: library_share[i] as f32,
        });
    }
    (features, evidence)
}

/// Label-blind noise model of fragment hits per ion class: how often a fragment of the class has signal and how its
/// correlation is distributed when it has, among candidates that are not the best of their precursor (rank >= 1,
/// targets and decoys alike). A co-eluting y2 is then cheap evidence, a co-eluting y9 is not.
pub struct IonNoiseModel {
    hit_rate: Vec<f64>,
    /// share of a class's positive hits in each bin or above, CORRELATION_BINS per class
    survival: Vec<f64>,
}

struct NoiseCounts {
    present: Vec<f64>,
    hits: Vec<f64>,
    histogram: Vec<f64>,
}

impl NoiseCounts {
    fn new(n_classes: usize) -> Self {
        Self {
            present: vec![0.0; n_classes],
            hits: vec![0.0; n_classes],
            histogram: vec![0.0; n_classes * CORRELATION_BINS],
        }
    }

    fn add(mut self, evidence: &[IonEvidence]) -> Self {
        for e in evidence {
            let c = e.class as usize;
            self.present[c] += 1.0;
            if e.hit {
                self.hits[c] += 1.0;
            }
            if e.positive {
                self.histogram[c * CORRELATION_BINS + e.bin as usize] += 1.0;
            }
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.present.iter_mut().zip(other.present) {
            *a += b;
        }
        for (a, b) in self.hits.iter_mut().zip(other.hits) {
            *a += b;
        }
        for (a, b) in self.histogram.iter_mut().zip(other.histogram) {
            *a += b;
        }
        self
    }
}

impl IonNoiseModel {
    /// Fit on the fragment evidence of the noise candidates.
    pub fn fit<'a, I>(noise_evidence: I, n_classes: usize) -> Self
    where
        I: ParallelIterator<Item = &'a [IonEvidence]>,
    {
        let counts = noise_evidence
            .fold(
                || NoiseCounts::new(n_classes),
                |counts, evidence| counts.add(evidence),
            )
            .reduce(|| NoiseCounts::new(n_classes), NoiseCounts::merge);

        let hit_rate = (0..n_classes)
            .map(|c| counts.hits[c] / counts.present[c].max(1.0))
            .collect();
        let mut survival = vec![0.0; n_classes * CORRELATION_BINS];
        for c in 0..n_classes {
            let histogram = &counts.histogram[c * CORRELATION_BINS..(c + 1) * CORRELATION_BINS];
            let total = histogram.iter().sum::<f64>().max(1.0);
            let mut above = 0.0;
            for bin in (0..CORRELATION_BINS).rev() {
                above += histogram[bin];
                survival[c * CORRELATION_BINS + bin] = above / total;
            }
        }
        Self { hit_rate, survival }
    }

    /// ion_fisher_lib: the summed surprise -ln(chance) of a candidate's positive hits, weighted by library share,
    /// where chance is the class's hit rate times its share of hits correlating at least as well.
    pub fn fisher(&self, evidence: &[IonEvidence]) -> f32 {
        let mut library_weighted = 0.0f64;
        for e in evidence.iter().filter(|e| e.positive) {
            let c = e.class as usize;
            let chance = self.hit_rate[c] * self.survival[c * CORRELATION_BINS + e.bin as usize];
            let surprise = -chance.max(MIN_CHANCE).ln();
            library_weighted += e.library_share as f64 * surprise;
        }
        library_weighted as f32
    }
}
