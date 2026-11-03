//! Features from the individual scored fragments of a candidate.
//!
//! Summaries over all fragments (counts, mean correlations) lose which ions carry the signal and whether the
//! co-eluting ones have the intensities the library predicts. These features keep that information.
//!
//! A decoy's fragments are its target's (copied intensities, ion types and numbers) at the m/z of its own sequence,
//! so every feature measures the evidence of a target and of its decoy the same way.
//!
//! `ion_fisher` and `ion_fisher_lib` need a noise model fitted over all scored candidates (`IonNoiseModel`); the
//! other features depend on the candidate alone.

use rayon::prelude::*;

use crate::constants::FragmentType;
use crate::peak_group_scoring::utils::AGREEMENT_MIN_CORRELATION;

/// Ion-number bucket edges of the noise model's ion classes (bucket = number of edges <= ion number)
const NULL_NUMBER_EDGES: [u8; 3] = [3, 5, 8];
/// Library slots (fragments ordered by library intensity) per slot group of the noise model's ion classes
const SLOT_GROUP: usize = 3;
/// Fragment charges above this share the class of this charge
const MAX_FRAGMENT_CHARGE: u8 = 2;
/// Ion-number edges of the short / mid / long coherent ion counts
const COUNT_NUMBER_EDGES: [u8; 2] = [4, 8];
/// Correlation histogram bins of the noise model on [0, 1]
const CORRELATION_BINS: usize = 100;
/// The chance of a fragment hit is floored here: one fragment adds at most -ln(MIN_CHANCE) surprise
const MIN_CHANCE: f64 = 1e-4;
/// Number of ion kinds: b, y, other
const ION_KINDS: usize = 3;
const BASE_CLASSES: usize =
    ION_KINDS * (NULL_NUMBER_EDGES.len() + 1) * (MAX_FRAGMENT_CHARGE as usize + 1);

/// The features of one candidate that depend on the candidate alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FragmentFeatures {
    /// fragment correlations averaged with the library intensity shares as weights
    pub xic_libweighted: f32,
    /// fragment correlations averaged with the observed intensity shares as weights
    pub xic_obsweighted: f32,
    /// co-eluting doubly charged fragments
    pub ion_coherent_2plus: f32,
    /// co-eluting b and y fragments of short (< 4), mid (4-7) and long (>= 8) ion number
    pub ion_coherent_b_short: f32,
    pub ion_coherent_b_mid: f32,
    pub ion_coherent_b_long: f32,
    pub ion_coherent_y_short: f32,
    pub ion_coherent_y_mid: f32,
    pub ion_coherent_y_long: f32,
    /// spectral entropy similarity (Li et al. 2021) of observed and library intensities
    pub ent_sim: f32,
    /// the same with each observed intensity weighted by its (non-negative) correlation
    pub ent_sim_coherent: f32,
    /// over the doubly charged fragments: library-weighted correlation, library share of the co-eluting ones and
    /// their count
    pub z2_libw_corr: f32,
    pub z2_explained: f32,
    pub z2_n_over_50: f32,
}

/// What the noise model needs of one scored fragment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonEvidence {
    /// ion class: ion kind x ion-number bucket x fragment charge x library slot group
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
    BASE_CLASSES * top_k_fragments.div_ceil(SLOT_GROUP).max(1)
}

fn shares(values: &[f64]) -> Vec<f64> {
    let total: f64 = values.iter().sum();
    if total > 0.0 {
        values.iter().map(|v| v / total).collect()
    } else {
        vec![0.0; values.len()]
    }
}

fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.ln())
        .sum::<f64>()
}

/// Spectral entropy similarity of the library and observed intensities; 0 without observed signal.
/// A fragment without signal counts as a missing peak.
fn entropy_similarity(library: &[f64], observed: &[f64]) -> f64 {
    if observed.iter().sum::<f64>() <= 0.0 {
        return 0.0;
    }
    let p = shares(library);
    let q = shares(observed);
    let mixed: Vec<f64> = p.iter().zip(&q).map(|(a, b)| (a + b) / 2.0).collect();
    1.0 - (2.0 * entropy(&mixed) - entropy(&p) - entropy(&q)) / 4f64.ln()
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

fn ion_class(fragment_type: u8, number: u8, charge: u8, slot: usize, n_slot_groups: usize) -> u16 {
    let kind = match fragment_type {
        FragmentType::B => 0,
        FragmentType::Y => 1,
        _ => 2,
    };
    let bucket = number_bucket(number, &NULL_NUMBER_EDGES);
    let charge = charge.min(MAX_FRAGMENT_CHARGE) as usize;
    let base = (kind * (NULL_NUMBER_EDGES.len() + 1) + bucket) * (MAX_FRAGMENT_CHARGE as usize + 1)
        + charge;
    (base * n_slot_groups + slot / SLOT_GROUP) as u16
}

/// The fragment features of one candidate and the evidence of its fragments for the noise model.
///
/// The slices hold one value per scored fragment: library intensity, observed intensity (summed over the
/// candidate's cycles), correlation with the candidate's median profile, ion type, ion number and fragment charge.
/// `top_k_fragments` is the maximum number of scored fragments per candidate, at least the length of the slices.
pub fn calculate_fragment_features(
    library_intensities: &[f32],
    observation_intensities: &[f32],
    correlations: &[f32],
    fragment_types: &[u8],
    fragment_numbers: &[u8],
    fragment_charges: &[u8],
    top_k_fragments: usize,
) -> (FragmentFeatures, Vec<IonEvidence>) {
    let n = library_intensities.len();
    if n == 0
        || observation_intensities.len() != n
        || correlations.len() != n
        || fragment_types.len() != n
        || fragment_numbers.len() != n
        || fragment_charges.len() != n
    {
        return (FragmentFeatures::default(), Vec::new());
    }
    let library: Vec<f64> = library_intensities.iter().map(|&x| x as f64).collect();
    let observed: Vec<f64> = observation_intensities.iter().map(|&x| x as f64).collect();
    let correlation: Vec<f64> = correlations.iter().map(|&x| x as f64).collect();
    let library_share = shares(&library);
    let observed_share = shares(&observed);
    let coherent: Vec<bool> = (0..n)
        .map(|i| observed[i] > 0.0 && correlations[i] > AGREEMENT_MIN_CORRELATION)
        .collect();

    let mut features = FragmentFeatures {
        xic_libweighted: (0..n)
            .map(|i| library_share[i] * correlation[i])
            .sum::<f64>() as f32,
        xic_obsweighted: (0..n)
            .map(|i| observed_share[i] * correlation[i])
            .sum::<f64>() as f32,
        ent_sim: entropy_similarity(&library, &observed) as f32,
        ..Default::default()
    };
    let observed_coherent: Vec<f64> = (0..n)
        .map(|i| observed[i] * correlation[i].max(0.0))
        .collect();
    features.ent_sim_coherent = entropy_similarity(&library, &observed_coherent) as f32;

    let mut z2_library = 0.0f64;
    let mut z2_weighted_correlation = 0.0f64;
    let mut z2_coherent_library = 0.0f64;
    for i in 0..n {
        if fragment_charges[i] == MAX_FRAGMENT_CHARGE {
            z2_library += library[i];
            z2_weighted_correlation += library[i] * correlation[i];
            if coherent[i] {
                z2_coherent_library += library[i];
                features.z2_n_over_50 += 1.0;
                features.ion_coherent_2plus += 1.0;
            }
        }
        if !coherent[i] {
            continue;
        }
        let count = match (
            fragment_types[i],
            number_bucket(fragment_numbers[i], &COUNT_NUMBER_EDGES),
        ) {
            (FragmentType::B, 0) => &mut features.ion_coherent_b_short,
            (FragmentType::B, 1) => &mut features.ion_coherent_b_mid,
            (FragmentType::B, _) => &mut features.ion_coherent_b_long,
            (FragmentType::Y, 0) => &mut features.ion_coherent_y_short,
            (FragmentType::Y, 1) => &mut features.ion_coherent_y_mid,
            (FragmentType::Y, _) => &mut features.ion_coherent_y_long,
            _ => continue,
        };
        *count += 1.0;
    }
    if z2_library > 0.0 {
        features.z2_libw_corr = (z2_weighted_correlation / z2_library) as f32;
        features.z2_explained = (z2_coherent_library / z2_library) as f32;
    }

    // library slots: fragments by descending library intensity, ties in their scored order
    let mut by_intensity: Vec<usize> = (0..n).collect();
    by_intensity.sort_by(|&a, &b| library[b].total_cmp(&library[a]));
    let n_slot_groups = top_k_fragments.div_ceil(SLOT_GROUP).max(1);
    let mut evidence: Vec<IonEvidence> = Vec::with_capacity(n);
    for (slot, &i) in by_intensity.iter().enumerate() {
        let hit = observed[i] > 0.0;
        evidence.push(IonEvidence {
            class: ion_class(
                fragment_types[i],
                fragment_numbers[i],
                fragment_charges[i],
                slot,
                n_slot_groups,
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
/// targets and decoys alike). A coherent y2 is then cheap evidence, a coherent y9 is not.
pub struct IonNoiseModel {
    hit_rate: Vec<f64>,
    /// share of a class's positive hits in each bin or above, CORRELATION_BINS per class
    survival: Vec<f64>,
}

struct NullCounts {
    present: Vec<f64>,
    hits: Vec<f64>,
    histogram: Vec<f64>,
}

impl NullCounts {
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
    /// Fit on the fragment evidence of the null candidates.
    pub fn fit<'a, I>(null_evidence: I, n_classes: usize) -> Self
    where
        I: ParallelIterator<Item = &'a [IonEvidence]>,
    {
        let counts = null_evidence
            .fold(
                || NullCounts::new(n_classes),
                |counts, evidence| counts.add(evidence),
            )
            .reduce(|| NullCounts::new(n_classes), NullCounts::merge);

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

    /// (ion_fisher, ion_fisher_lib): the summed surprise -ln(chance) of a candidate's positive hits, plain and
    /// weighted by library share, where chance is the class's hit rate times its share of hits correlating at least
    /// as well.
    pub fn fisher(&self, evidence: &[IonEvidence]) -> (f32, f32) {
        let mut plain = 0.0f64;
        let mut library_weighted = 0.0f64;
        for e in evidence.iter().filter(|e| e.positive) {
            let c = e.class as usize;
            let chance = self.hit_rate[c] * self.survival[c * CORRELATION_BINS + e.bin as usize];
            let surprise = -chance.max(MIN_CHANCE).ln();
            plain += surprise;
            library_weighted += e.library_share as f64 * surprise;
        }
        (plain as f32, library_weighted as f32)
    }
}
