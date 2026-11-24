//! Learned features of the raw MS2 traces of a candidate, trained end-to-end by `symdia` and evaluated here from
//! the exported network in `symbolic_weights.rs`.
//!
//! The network reads, for the most intense library fragments and a window of cycles around the candidate, the
//! trace at the fragment m/z, its mass error, the traces at the neighbouring isotope positions and the signal in two
//! bands around the fragment. A voxel layer maps every (fragment, cycle) point to a few units, fixed reductions over
//! cycles turn them into per-fragment values, a fragment layer combines those with library properties, and fixed
//! reductions over fragments give the candidate-level features. Every layer is a small dense layer with one hidden
//! ReLU layer.
//!
//! Intensities are used relative to the candidate, the fragment and the run; mass error and retention time
//! deviation are expressed in units of their spread among the best-scoring candidates of the run (`RunScales`).
//! The network never reads a fragment or precursor m/z: exact masses would let it recognise mutated decoys.

use numpy::ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::candidate::{Candidate, CandidateCollection};
use crate::constants::FragmentType;
use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::symbolic_weights::{FEATURE_NAMES, MODEL, NUM_FEATURES};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;

/// Half-width in cycles of the window around `candidate.cycle_center` that the features read.
pub const SYMBOLIC_HALF_WINDOW: usize = 7;
pub const WINDOW_LEN: usize = 2 * SYMBOLIC_HALF_WINDOW + 1;
/// The most intense library fragments of a precursor that the features read.
pub const MAX_FRAGMENTS: usize = 12;
/// Traces next to a fragment in m/z: one, minus one and two isotope spacings, half a spacing (where only an ion of
/// twice the fragment charge has a peak), and the bands within `NEAR_BAND_TOLERANCES` and `FAR_BAND_TOLERANCES`
/// mass tolerances of the fragment without the band inside of them.
pub const NUM_CONTEXT_TRACES: usize = 6;
const ISOTOPE_STEPS: [f32; 4] = [1.0, -1.0, 2.0, 0.5];
/// Mass difference between 13C and 12C in Da.
const ISOTOPE_SPACING: f32 = 1.003_355;
const NEAR_BAND_TOLERANCES: f32 = 4.0;
const FAR_BAND_TOLERANCES: f32 = 20.0;

const NUM_ANCHOR_FRAGMENTS: usize = 6;
const NUM_SCALE_CANDIDATES: usize = 1000;
const NUM_PEAK_SAMPLE_CANDIDATES: usize = 8192;
const MAD_TO_STD: f32 = 1.4826;
const CORE_STD_ITERATIONS: usize = 50;
const MIN_SIGMA_MZ_FRACTION: f32 = 0.05;
const FALLBACK_TOLERANCE_TO_SIGMA: f32 = 3.0;
const MIN_SIGMA_RT_CYCLES: f32 = 0.5;

const U_MZ_CLAMP: f32 = 4.0;
const U_RT_CLAMP: f32 = 5.0;
const LOG_INTENSITY_DIVISOR: f32 = 5.0;
const ISOTOPE_RATIO_CLAMP: f32 = 4.0;
const ION_NUMBER_SCALE: f32 = 10.0;

pub const NUM_VOXEL_CHANNELS: usize = 9 + NUM_CONTEXT_TRACES;
pub const NUM_TIME_REDUCTIONS: usize = 9;
pub const NUM_FRAGMENT_SCALARS: usize = 8;
pub const NUM_CANDIDATE_SCALARS: usize = 3;
pub const HIDDEN: usize = 64;
pub const NUM_VOXEL_UNITS: usize = 12;
pub const NUM_FRAGMENT_UNITS: usize = 16;
pub const NUM_JOINT_UNITS: usize = 12;
pub const NUM_FRAGMENT_INPUTS: usize = NUM_VOXEL_UNITS * NUM_TIME_REDUCTIONS + NUM_FRAGMENT_SCALARS;

/// The learned features of one candidate, in the order of `symbolic_weights::FEATURE_NAMES`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolicFeatures {
    pub values: [f32; NUM_FEATURES],
}

impl Default for SymbolicFeatures {
    fn default() -> Self {
        Self {
            values: [0.0; NUM_FEATURES],
        }
    }
}

impl FeatureBlock for SymbolicFeatures {
    const NAMES: &'static [&'static str] = &FEATURE_NAMES;

    fn values(&self) -> Vec<f32> {
        self.values.to_vec()
    }
}

/// `w2 * relu(w1 * x + b1) + b2`. `w1` holds one row of `HIDDEN` weights per input, `w2` one row of output weights
/// per hidden unit.
pub struct DenseLayer {
    pub w1: &'static [f32],
    pub b1: &'static [f32; HIDDEN],
    pub w2: &'static [f32],
    pub b2: &'static [f32],
}

impl DenseLayer {
    fn eval(&self, x: &[f32], out: &mut [f32]) {
        let mut hidden = *self.b1;
        for (&value, weights) in x.iter().zip(self.w1.chunks_exact(HIDDEN)) {
            if value != 0.0 {
                for (h, w) in hidden.iter_mut().zip(weights) {
                    *h += w * value;
                }
            }
        }
        out.copy_from_slice(self.b2);
        for (&h, weights) in hidden.iter().zip(self.w2.chunks_exact(out.len())) {
            if h > 0.0 {
                for (o, w) in out.iter_mut().zip(weights) {
                    *o += w * h;
                }
            }
        }
    }
}

/// Where an input of the joint layer or an output feature comes from.
pub enum Source {
    /// Reduction `reduction` over fragments of fragment unit `unit`
    FragmentSet {
        unit: usize,
        reduction: u8,
    },
    /// `tanh` of a `FragmentSet`
    FragmentSetTanh {
        unit: usize,
        reduction: u8,
    },
    Joint(usize),
    CandidateScalar(usize),
}

pub struct Model {
    pub voxel: DenseLayer,
    pub fragment: DenseLayer,
    pub joint: DenseLayer,
    pub joint_inputs: &'static [Source],
    pub outputs: &'static [Source],
}

/// Spread of the mass error and of the retention time deviation among the best-scoring candidates of a run, and the
/// median intensity of the peaks around its candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunScales {
    pub median_peak: f32,
    pub sigma_mz: f32,
    pub sigma_rt: f32,
}

/// The traces next to the fragments of a precursor in m/z (`fragments x cycles`, fragments in library order),
/// starting at cycle `lo`. The two bands still include what lies inside of them.
pub struct ContextTraces {
    lo: usize,
    isotopes: [Array2<f32>; ISOTOPE_STEPS.len()],
    within_near: Array2<f32>,
    within_far: Array2<f32>,
}

impl ContextTraces {
    pub fn extract<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        lo: usize,
        hi: usize,
        mass_tolerance: f32,
    ) -> Self {
        let extract = |mz: &[f32], tolerance: f32| {
            DenseXICObservation::new(dia_data, precursor.mz, lo, hi, tolerance, mz).dense_xic
        };
        let isotopes = ISOTOPE_STEPS.map(|steps| {
            let mz: Vec<f32> = precursor
                .fragment_mz
                .iter()
                .zip(&precursor.fragment_charge)
                .map(|(mz, &charge)| mz + steps * ISOTOPE_SPACING / charge.max(1) as f32)
                .collect();
            extract(&mz, mass_tolerance)
        });
        Self {
            lo,
            isotopes,
            within_near: extract(
                &precursor.fragment_mz,
                NEAR_BAND_TOLERANCES * mass_tolerance,
            ),
            within_far: extract(&precursor.fragment_mz, FAR_BAND_TOLERANCES * mass_tolerance),
        }
    }
}

/// What the network reads for one candidate. Fragments are ordered by decreasing library intensity.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicInputs {
    pub num_fragments: usize,
    pub xic: [[f32; WINDOW_LEN]; MAX_FRAGMENTS],
    pub ppm_error: [[f32; WINDOW_LEN]; MAX_FRAGMENTS],
    pub context: [[[f32; WINDOW_LEN]; NUM_CONTEXT_TRACES]; MAX_FRAGMENTS],
    pub library_intensity: [f32; MAX_FRAGMENTS],
    pub is_y: [f32; MAX_FRAGMENTS],
    pub charge: [f32; MAX_FRAGMENTS],
    pub number: [f32; MAX_FRAGMENTS],
    pub delta_rt: f32,
    pub naa: f32,
    pub cycle_start_offset: f32,
    pub cycle_stop_offset: f32,
}

impl SymbolicInputs {
    /// Reads the inputs from XIC and m/z arrays (`fragments x cycles`) that start at cycle `dense_lo` and cover the
    /// window of the candidate; cycles outside the run stay zero. Without `context` the context traces stay zero.
    pub fn from_dense(
        dense_xic: &Array2<f32>,
        dense_mz: &Array2<f32>,
        dense_lo: usize,
        context: Option<&ContextTraces>,
        precursor: &Precursor,
        candidate: &Candidate,
        rt: &Array1<f32>,
    ) -> Self {
        let mut order: Vec<usize> = (0..precursor.fragment_mz.len()).collect();
        order.sort_by(|&a, &b| {
            precursor.fragment_intensity[b]
                .partial_cmp(&precursor.fragment_intensity[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(MAX_FRAGMENTS);

        let mut inputs = Self {
            num_fragments: order.len(),
            xic: [[0.0; WINDOW_LEN]; MAX_FRAGMENTS],
            ppm_error: [[0.0; WINDOW_LEN]; MAX_FRAGMENTS],
            context: [[[0.0; WINDOW_LEN]; NUM_CONTEXT_TRACES]; MAX_FRAGMENTS],
            library_intensity: [0.0; MAX_FRAGMENTS],
            is_y: [0.0; MAX_FRAGMENTS],
            charge: [0.0; MAX_FRAGMENTS],
            number: [0.0; MAX_FRAGMENTS],
            delta_rt: rt[candidate.cycle_center] - precursor.rt,
            naa: precursor.naa as f32,
            cycle_start_offset: candidate.cycle_start as f32 - candidate.cycle_center as f32,
            cycle_stop_offset: candidate.cycle_stop as f32 - candidate.cycle_center as f32,
        };

        let window_start = candidate.cycle_center as isize - SYMBOLIC_HALF_WINDOW as isize;
        let dense_hi = dense_lo + dense_xic.shape()[1];
        for (slot, &f) in order.iter().enumerate() {
            inputs.library_intensity[slot] = precursor.fragment_intensity[f].max(0.0);
            inputs.is_y[slot] = (precursor.fragment_type[f] == FragmentType::Y) as u8 as f32;
            inputs.charge[slot] = precursor.fragment_charge[f] as f32;
            inputs.number[slot] = precursor.fragment_number[f] as f32;
            let target_mz = precursor.fragment_mz[f];
            for t in 0..WINDOW_LEN {
                let cycle = window_start + t as isize;
                if cycle < dense_lo as isize || cycle >= dense_hi as isize {
                    continue;
                }
                let col = cycle as usize - dense_lo;
                let intensity = dense_xic[[f, col]];
                let observed_mz = dense_mz[[f, col]];
                inputs.xic[slot][t] = intensity;
                if intensity > 0.0 && observed_mz > 0.0 && target_mz > 0.0 {
                    inputs.ppm_error[slot][t] = (observed_mz - target_mz) / target_mz * 1e6;
                }
                let Some(context) = context else { continue };
                if cycle < context.lo as isize {
                    continue;
                }
                let col = cycle as usize - context.lo;
                if col >= context.within_near.shape()[1] {
                    continue;
                }
                for (trace, isotope) in inputs.context[slot].iter_mut().zip(&context.isotopes) {
                    trace[t] = isotope[[f, col]];
                }
                let within_near = context.within_near[[f, col]];
                inputs.context[slot][ISOTOPE_STEPS.len()][t] = within_near - intensity;
                inputs.context[slot][ISOTOPE_STEPS.len() + 1][t] =
                    context.within_far[[f, col]] - within_near;
            }
        }
        inputs
    }

    fn window<T: DIADataTrait>(dia_data: &T, candidate: &Candidate) -> (usize, usize) {
        let n_cycles = dia_data.rt_index().rt.len();
        let lo = candidate
            .cycle_center
            .saturating_sub(SYMBOLIC_HALF_WINDOW)
            .min(n_cycles);
        let hi = (candidate.cycle_center + SYMBOLIC_HALF_WINDOW + 1).min(n_cycles);
        (lo, hi)
    }

    fn extract_with<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        mass_tolerance: f32,
        with_context: bool,
    ) -> Self {
        let (lo, hi) = Self::window(dia_data, candidate);
        let dense = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            lo,
            hi,
            mass_tolerance,
            &precursor.fragment_mz,
        );
        let context = with_context
            .then(|| ContextTraces::extract(dia_data, precursor, lo, hi, mass_tolerance));
        Self::from_dense(
            &dense.dense_xic,
            &dense.dense_mz,
            lo,
            context.as_ref(),
            precursor,
            candidate,
            &dia_data.rt_index().rt,
        )
    }

    /// Extracts the window of the candidate from the raw data and reads the inputs from it.
    pub fn extract<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        mass_tolerance: f32,
    ) -> Self {
        Self::extract_with(dia_data, precursor, candidate, mass_tolerance, true)
    }

    /// As `extract`, but without the context traces, which the run scales do not need.
    fn extract_target<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        mass_tolerance: f32,
    ) -> Self {
        Self::extract_with(dia_data, precursor, candidate, mass_tolerance, false)
    }
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(values[(values.len() - 1) / 2])
}

/// Robust standard deviation: the median absolute deviation from the median, scaled to a normal distribution.
fn robust_std(values: &mut [f32]) -> Option<f32> {
    let center = median(values)?;
    let mut deviations: Vec<f32> = values.iter().map(|v| (v - center).abs()).collect();
    median(&mut deviations).map(|mad| MAD_TO_STD * mad)
}

/// Standard deviation of the narrow component of a sample that also holds a broad background: the normal part of
/// a mixture of a normal distribution and a uniform one on `-half_range..half_range`, fitted by expectation
/// maximisation. Unlike the median absolute deviation it stays with the narrow component when most of the sample
/// is background, as the mass errors of a run are in which few of the best-scoring candidates are real.
fn core_std(values: &[f32], half_range: f32) -> Option<f32> {
    let mut sorted = values.to_vec();
    let mut mean = median(&mut sorted)? as f64;
    let mut std = (robust_std(&mut sorted)? as f64 / 2.0).max(1e-3);
    let mut share = 0.5f64;
    let background = 1.0 / (2.0 * half_range as f64);
    for _ in 0..CORE_STD_ITERATIONS {
        let (mut weight, mut weighted, mut squares) = (0.0f64, 0.0f64, 0.0f64);
        for &value in values {
            let z = (value as f64 - mean) / std;
            let normal = share * (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt());
            let responsibility = normal / (normal + (1.0 - share) * background);
            weight += responsibility;
            weighted += responsibility * value as f64;
            squares += responsibility * (value as f64 - mean) * (value as f64 - mean);
        }
        if weight <= 0.0 {
            return None;
        }
        mean = weighted / weight;
        std = (squares / weight).sqrt().max(1e-3);
        share = (weight / values.len() as f64).clamp(1e-3, 1.0 - 1e-3);
    }
    Some(std as f32)
}

impl RunScales {
    /// Estimates the scales from all candidates of a run. The spreads come from the `NUM_SCALE_CANDIDATES` candidates
    /// with the highest score, the mass-error spread from the narrow component of their errors (`core_std`); the
    /// peak intensity comes from an even subsample of all candidates.
    pub fn estimate<T: DIADataTrait + Sync>(
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
        mass_tolerance: f32,
        top_k_fragments: usize,
    ) -> Self
    where
        T::QuadrupoleObservation: Sync,
    {
        let extract = |candidate: &Candidate| {
            lib.get_precursor_by_idx_filtered(
                candidate.precursor_idx,
                true,
                true,
                false,
                top_k_fragments,
            )
            .map(|precursor| {
                SymbolicInputs::extract_target(dia_data, &precursor, candidate, mass_tolerance)
            })
        };

        let mut by_score: Vec<&Candidate> = candidates.iter().collect();
        let num_top = NUM_SCALE_CANDIDATES.min(by_score.len());
        if num_top > 0 && num_top < by_score.len() {
            by_score.select_nth_unstable_by(num_top - 1, |a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let top_inputs: Vec<SymbolicInputs> = by_score[..num_top]
            .par_iter()
            .filter_map(|candidate| extract(candidate))
            .collect();

        let mut delta_rt: Vec<f32> = top_inputs.iter().map(|inputs| inputs.delta_rt).collect();
        let ppm_error: Vec<f32> = top_inputs
            .iter()
            .flat_map(|inputs| {
                (0..inputs.num_fragments)
                    .filter(|&f| inputs.xic[f][SYMBOLIC_HALF_WINDOW] > 0.0)
                    .map(|f| inputs.ppm_error[f][SYMBOLIC_HALF_WINDOW])
            })
            .collect();

        let rt = &dia_data.rt_index().rt;
        let mut cycle_durations: Vec<f32> = rt
            .windows(2)
            .into_iter()
            .map(|pair| pair[1] - pair[0])
            .collect();
        let min_sigma_rt = MIN_SIGMA_RT_CYCLES * median(&mut cycle_durations).unwrap_or(1.0);
        let sigma_rt = robust_std(&mut delta_rt).unwrap_or(0.0).max(min_sigma_rt);
        let sigma_mz = core_std(&ppm_error, mass_tolerance)
            .unwrap_or(mass_tolerance / FALLBACK_TOLERANCE_TO_SIGMA)
            .max(MIN_SIGMA_MZ_FRACTION * mass_tolerance);

        let step = (candidates.len() / NUM_PEAK_SAMPLE_CANDIDATES).max(1);
        let sample: Vec<&Candidate> = candidates.iter().step_by(step).collect();
        let mut peaks: Vec<f32> = sample
            .par_iter()
            .filter_map(|candidate| extract(candidate))
            .flat_map_iter(|inputs| {
                inputs
                    .xic
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|&intensity| intensity > 0.0)
                    .collect::<Vec<f32>>()
            })
            .collect();
        let median_peak = median(&mut peaks).unwrap_or(1.0);

        Self {
            median_peak,
            sigma_mz,
            sigma_rt,
        }
    }
}

fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Position of a cycle in the window and whether it lies within the candidate boundaries.
fn cycle_channels(inputs: &SymbolicInputs, t: usize) -> [f32; 3] {
    let half = SYMBOLIC_HALF_WINDOW as f32;
    let offset = t as f32 - half;
    let in_boundaries = offset >= inputs.cycle_start_offset && offset < inputs.cycle_stop_offset;
    [
        (offset / half).abs(),
        offset / half,
        in_boundaries as u8 as f32,
    ]
}

/// The voxel channels of a (fragment, cycle) point without signal in any trace; they depend on the cycle only.
fn empty_point_channels(inputs: &SymbolicInputs, t: usize) -> [f32; NUM_VOXEL_CHANNELS] {
    let mut channels = [0.0f32; NUM_VOXEL_CHANNELS];
    channels[4] = U_MZ_CLAMP;
    channels[6..9].copy_from_slice(&cycle_channels(inputs, t));
    channels
}

fn has_signal(inputs: &SymbolicInputs, f: usize, t: usize) -> bool {
    inputs.xic[f][t] > 0.0 || inputs.context[f].iter().any(|trace| trace[t] > 0.0)
}

/// The voxel channels of a (fragment, cycle) point: intensity relative to the candidate, to the fragment and to the
/// run, detection, mass error in units of its spread, position in the window and in the candidate boundaries, the
/// isotope and half-spacing traces relative to the fragment apex, and the two bands relative to the run.
fn point_channels(
    inputs: &SymbolicInputs,
    scales: &RunScales,
    candidate_max: f32,
    fragment_max: f32,
    f: usize,
    t: usize,
) -> [f32; NUM_VOXEL_CHANNELS] {
    let intensity = inputs.xic[f][t];
    let detected = intensity > 0.0;
    let u = (inputs.ppm_error[f][t] / scales.sigma_mz).clamp(-U_MZ_CLAMP, U_MZ_CLAMP);
    let median_peak = scales.median_peak.max(1e-12);
    let mut channels = [0.0f32; NUM_VOXEL_CHANNELS];
    channels[0] = intensity / candidate_max.max(1e-12);
    channels[1] = intensity / fragment_max.max(1e-12);
    channels[2] = (intensity / median_peak).ln_1p() / LOG_INTENSITY_DIVISOR;
    channels[3] = detected as u8 as f32;
    channels[4] = if detected { u.abs() } else { U_MZ_CLAMP };
    channels[5] = if detected { u } else { 0.0 };
    channels[6..9].copy_from_slice(&cycle_channels(inputs, t));
    for (j, trace) in inputs.context[f].iter().enumerate() {
        let value = trace[t].max(0.0);
        channels[9 + j] = if j >= ISOTOPE_STEPS.len() {
            (value / median_peak).ln_1p() / LOG_INTENSITY_DIVISOR
        } else if fragment_max > 0.0 {
            (value / fragment_max).min(ISOTOPE_RATIO_CLAMP)
        } else {
            0.0
        };
    }
    channels
}

/// Reduces the traces of one voxel unit (`fragments x cycles`) over the cycles, appending `NUM_TIME_REDUCTIONS`
/// values per fragment to `out`: mean, max, apex, mean within the candidate boundaries, intensity-weighted mean,
/// standard deviation, and the mean correlation to the other fragments weighted evenly, by library intensity and
/// towards the most intense library fragments.
fn time_reductions(
    traces: &[[f32; WINDOW_LEN]],
    inputs: &SymbolicInputs,
    library_share: &[f32],
    relative_xic: &[[f32; WINDOW_LEN]],
    out: &mut [Vec<f32>],
) {
    let n = traces.len();
    let inv_t = 1.0 / WINDOW_LEN as f32;
    let mut unit_traces = vec![[0.0f32; WINDOW_LEN]; n];
    let mut has_variation = vec![0.0f32; n];
    let mut basic = vec![[0.0f32; 6]; n];

    let num_in_boundaries = (0..WINDOW_LEN)
        .filter(|&t| {
            let offset = t as f32 - SYMBOLIC_HALF_WINDOW as f32;
            offset >= inputs.cycle_start_offset && offset < inputs.cycle_stop_offset
        })
        .count()
        .max(1) as f32;

    for f in 0..n {
        let trace = &traces[f];
        let mean = trace.iter().sum::<f32>() * inv_t;
        let max = trace.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut in_boundaries_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        let mut weighted_sum = 0.0f32;
        let mut squares = 0.0f32;
        for t in 0..WINDOW_LEN {
            let offset = t as f32 - SYMBOLIC_HALF_WINDOW as f32;
            if offset >= inputs.cycle_start_offset && offset < inputs.cycle_stop_offset {
                in_boundaries_sum += trace[t];
            }
            let weight = relative_xic[f][t] + 1e-9;
            weight_sum += weight;
            weighted_sum += trace[t] * weight;
            let centered = trace[t] - mean;
            unit_traces[f][t] = centered;
            squares += centered * centered;
        }
        let norm = squares.max(1e-10).sqrt();
        has_variation[f] = (norm > 1e-4) as u8 as f32;
        for value in unit_traces[f].iter_mut() {
            *value = *value / norm * has_variation[f];
        }
        basic[f] = [
            mean,
            max,
            trace[SYMBOLIC_HALF_WINDOW],
            in_boundaries_sum / num_in_boundaries,
            weighted_sum / weight_sum,
            (squares * inv_t).max(1e-7).sqrt(),
        ];
    }

    let correlation_to_others = |weight_of: &dyn Fn(usize) -> f32| -> Vec<f32> {
        let weights: Vec<f32> = (0..n).map(|f| weight_of(f) * has_variation[f]).collect();
        let weight_sum: f32 = weights.iter().sum();
        let mut template = [0.0f32; WINDOW_LEN];
        for f in 0..n {
            for t in 0..WINDOW_LEN {
                template[t] += unit_traces[f][t] * weights[f];
            }
        }
        (0..n)
            .map(|f| {
                let dot: f32 = (0..WINDOW_LEN)
                    .map(|t| unit_traces[f][t] * template[t])
                    .sum();
                let others = weight_sum - weights[f];
                if others > 1e-9 {
                    (dot - weights[f]) / others * has_variation[f]
                } else {
                    0.0
                }
            })
            .collect()
    };
    let corr_mean = correlation_to_others(&|_| 1.0);
    let corr_library = correlation_to_others(&|f| library_share[f]);
    let corr_anchor = correlation_to_others(&|f| (f < NUM_ANCHOR_FRAGMENTS) as u8 as f32);

    for f in 0..n {
        out[f].extend(basic[f].iter().map(|&v| sanitize(v)));
        out[f].push(sanitize(corr_mean[f]));
        out[f].push(sanitize(corr_library[f]));
        out[f].push(sanitize(corr_anchor[f]));
    }
}

fn entropy(probabilities: impl Iterator<Item = f32>) -> f32 {
    -probabilities.map(|p| p * p.max(1e-10).ln()).sum::<f32>()
}

/// Reduces the values of one fragment unit over the fragments: mean, lower median, standard deviation, library
/// weighted sum, library weighted mean over the detected fragments, soft count of positive values, log of the
/// positive sum, mean of the three largest, and cosine and Jensen-Shannon similarity of the positive part to the
/// library.
fn set_reduction(reduction: u8, values: &[f32], library_share: &[f32], detected: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / n as f32;
    let positive = || values.iter().map(|v| v.max(0.0));
    let result = match reduction {
        0 => mean,
        1 => {
            let mut sorted = values.to_vec();
            median(&mut sorted).unwrap_or(0.0)
        }
        2 => (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32)
            .max(1e-8)
            .sqrt(),
        3 => values.iter().zip(library_share).map(|(v, l)| v * l).sum(),
        4 => {
            let weight_sum: f32 = library_share.iter().zip(detected).map(|(l, d)| l * d).sum();
            values
                .iter()
                .zip(library_share.iter().zip(detected))
                .map(|(v, (l, d))| v * l * d)
                .sum::<f32>()
                / weight_sum.max(1e-9)
        }
        5 => values.iter().map(|v| 1.0 / (1.0 + (-3.0 * v).exp())).sum(),
        6 => positive().sum::<f32>().ln_1p(),
        7 => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k = n.min(3);
            sorted[..k].iter().sum::<f32>() / k as f32
        }
        8 => {
            let dot: f32 = positive().zip(library_share).map(|(p, l)| p * l).sum();
            let norm_positive = positive().map(|p| p * p).sum::<f32>().sqrt();
            let norm_library = library_share.iter().map(|l| l * l).sum::<f32>().sqrt();
            dot / (norm_positive * norm_library).max(1e-10)
        }
        _ => {
            let positive_sum: f32 = positive().sum();
            if positive_sum <= 1e-6 {
                0.0
            } else {
                let share = || positive().map(move |p| p / positive_sum.max(1e-10));
                let mixed = share().zip(library_share).map(|(p, l)| 0.5 * (p + l));
                let divergence = (2.0 * entropy(mixed)
                    - entropy(share())
                    - entropy(library_share.iter().copied()))
                    / 4.0f32.ln();
                (1.0 - divergence).clamp(0.0, 1.0)
            }
        }
    };
    sanitize(result)
}

fn max_of(values: impl Iterator<Item = f32>) -> f32 {
    values.fold(0.0f32, f32::max)
}

/// The learned features of a candidate from its inputs.
pub fn calculate_symbolic_features(
    inputs: &SymbolicInputs,
    scales: &RunScales,
) -> SymbolicFeatures {
    let mut features = SymbolicFeatures::default();
    let n = inputs.num_fragments;
    if n == 0 {
        return features;
    }
    let model = &MODEL;
    let candidate_max = max_of(inputs.xic[..n].iter().flatten().copied());
    let library_sum: f32 = inputs.library_intensity[..n].iter().sum::<f32>().max(1e-12);
    let library_share: Vec<f32> = inputs.library_intensity[..n]
        .iter()
        .map(|l| l / library_sum)
        .collect();
    let relative_xic: Vec<[f32; WINDOW_LEN]> = inputs.xic[..n]
        .iter()
        .map(|trace| trace.map(|intensity| intensity / candidate_max.max(1e-12)))
        .collect();
    let detected: Vec<f32> = (0..n)
        .map(|f| inputs.xic[f].iter().any(|&v| v > 0.0) as u8 as f32)
        .collect();

    // Most points of a candidate carry no signal; the voxel units of such a point depend on the cycle only.
    let mut empty_units = [[0.0f32; NUM_VOXEL_UNITS]; WINDOW_LEN];
    for (t, units) in empty_units.iter_mut().enumerate() {
        model.voxel.eval(&empty_point_channels(inputs, t), units);
    }
    let mut voxel_units = vec![[[0.0f32; WINDOW_LEN]; MAX_FRAGMENTS]; NUM_VOXEL_UNITS];
    let mut units = [0.0f32; NUM_VOXEL_UNITS];
    for f in 0..n {
        let fragment_max = max_of(inputs.xic[f].iter().copied());
        for t in 0..WINDOW_LEN {
            if has_signal(inputs, f, t) {
                let channels = point_channels(inputs, scales, candidate_max, fragment_max, f, t);
                model.voxel.eval(&channels, &mut units);
            } else {
                units = empty_units[t];
            }
            for (unit_traces, value) in voxel_units.iter_mut().zip(units) {
                unit_traces[f][t] = value;
            }
        }
    }

    let mut fragment_inputs: Vec<Vec<f32>> = vec![Vec::with_capacity(NUM_FRAGMENT_INPUTS); n];
    for unit_traces in &voxel_units {
        time_reductions(
            &unit_traces[..n],
            inputs,
            &library_share,
            &relative_xic,
            &mut fragment_inputs,
        );
    }
    let mut fragment_values = [[0.0f32; MAX_FRAGMENTS]; NUM_FRAGMENT_UNITS];
    let mut units = [0.0f32; NUM_FRAGMENT_UNITS];
    for f in 0..n {
        fragment_inputs[f].extend([
            library_share[f],
            (100.0 * library_share[f]).ln_1p(),
            detected[f],
            inputs.is_y[f],
            (inputs.charge[f] >= 2.0) as u8 as f32,
            (inputs.number[f] / inputs.naa.max(1.0)).clamp(0.0, 1.0),
            (f + 1) as f32 / n as f32,
            inputs.number[f] / ION_NUMBER_SCALE,
        ]);
        model.fragment.eval(&fragment_inputs[f], &mut units);
        for (values, value) in fragment_values.iter_mut().zip(units) {
            values[f] = value;
        }
    }

    let deviation = inputs.delta_rt.abs() / scales.sigma_rt;
    let candidate_scalars: [f32; NUM_CANDIDATE_SCALARS] = [
        deviation.min(U_RT_CLAMP),
        deviation.ln_1p(),
        n as f32 / MAX_FRAGMENTS as f32,
    ];

    let fragment_set = |unit: usize, reduction: u8| {
        set_reduction(
            reduction,
            &fragment_values[unit][..n],
            &library_share,
            &detected,
        )
    };
    let joint_inputs: Vec<f32> = model
        .joint_inputs
        .iter()
        .map(|source| match source {
            Source::FragmentSet { unit, reduction } => fragment_set(*unit, *reduction),
            Source::FragmentSetTanh { unit, reduction } => fragment_set(*unit, *reduction).tanh(),
            Source::CandidateScalar(i) => candidate_scalars[*i],
            Source::Joint(_) => 0.0,
        })
        .collect();
    let mut joint_values = [0.0f32; NUM_JOINT_UNITS];
    model.joint.eval(&joint_inputs, &mut joint_values);

    for (value, source) in features.values.iter_mut().zip(model.outputs) {
        *value = sanitize(match source {
            Source::FragmentSet { unit, reduction } => fragment_set(*unit, *reduction),
            Source::FragmentSetTanh { unit, reduction } => fragment_set(*unit, *reduction).tanh(),
            Source::Joint(i) => joint_values[*i],
            Source::CandidateScalar(i) => candidate_scalars[*i],
        });
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peak_group_scoring::symbolic_fixture::{
        FixtureCase, FIXTURE_CASES, FIXTURE_MEDIAN_PEAK, FIXTURE_SIGMA_MZ, FIXTURE_SIGMA_RT,
    };

    const FIXTURE_SCALES: RunScales = RunScales {
        median_peak: FIXTURE_MEDIAN_PEAK,
        sigma_mz: FIXTURE_SIGMA_MZ,
        sigma_rt: FIXTURE_SIGMA_RT,
    };

    fn inputs_of(case: &FixtureCase) -> SymbolicInputs {
        SymbolicInputs {
            num_fragments: case.num_fragments,
            xic: case.xic,
            ppm_error: case.ppm_error,
            context: case.context,
            library_intensity: case.library_intensity,
            is_y: case.is_y,
            charge: case.charge,
            number: case.number,
            delta_rt: case.delta_rt,
            naa: case.naa,
            cycle_start_offset: case.cycle_start_offset,
            cycle_stop_offset: case.cycle_stop_offset,
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance * (1.0 + e.abs()),
                "feature {} ({}): {} != {}",
                i,
                FEATURE_NAMES[i],
                a,
                e
            );
        }
    }

    #[test]
    fn test_features_match_the_pytorch_model_on_real_candidates() {
        for case in FIXTURE_CASES.iter() {
            // given
            let inputs = inputs_of(case);

            // when
            let features = calculate_symbolic_features(&inputs, &FIXTURE_SCALES);

            // then
            assert_close(&features.values, &case.expected, 2e-3);
        }
    }

    #[test]
    fn test_features_do_not_depend_on_the_intensity_scale() {
        for case in FIXTURE_CASES.iter() {
            // given
            let gain = 7.3;
            let inputs = inputs_of(case);
            let mut scaled = inputs.clone();
            for value in scaled.xic.iter_mut().flatten() {
                *value *= gain;
            }
            for value in scaled.context.iter_mut().flatten().flatten() {
                *value *= gain;
            }
            let scaled_scales = RunScales {
                median_peak: FIXTURE_MEDIAN_PEAK * gain,
                ..FIXTURE_SCALES
            };

            // when
            let features = calculate_symbolic_features(&inputs, &FIXTURE_SCALES);
            let scaled_features = calculate_symbolic_features(&scaled, &scaled_scales);

            // then
            assert_close(&scaled_features.values, &features.values, 1e-4);
        }
    }

    #[test]
    fn test_candidate_without_fragments_scores_zero() {
        // given
        let inputs = SymbolicInputs {
            num_fragments: 0,
            xic: [[0.0; WINDOW_LEN]; MAX_FRAGMENTS],
            ppm_error: [[0.0; WINDOW_LEN]; MAX_FRAGMENTS],
            context: [[[0.0; WINDOW_LEN]; NUM_CONTEXT_TRACES]; MAX_FRAGMENTS],
            library_intensity: [0.0; MAX_FRAGMENTS],
            is_y: [0.0; MAX_FRAGMENTS],
            charge: [0.0; MAX_FRAGMENTS],
            number: [0.0; MAX_FRAGMENTS],
            delta_rt: 1.0,
            naa: 10.0,
            cycle_start_offset: -3.0,
            cycle_stop_offset: 4.0,
        };

        // when
        let features = calculate_symbolic_features(&inputs, &FIXTURE_SCALES);

        // then
        assert_eq!(features, SymbolicFeatures::default());
    }

    #[test]
    fn test_points_without_signal_have_the_channels_of_an_empty_point() {
        for case in FIXTURE_CASES.iter() {
            // given
            let inputs = inputs_of(case);
            for f in 0..inputs.num_fragments {
                let fragment_max = max_of(inputs.xic[f].iter().copied());
                for t in 0..WINDOW_LEN {
                    if has_signal(&inputs, f, t) {
                        continue;
                    }

                    // when
                    let channels =
                        point_channels(&inputs, &FIXTURE_SCALES, 1.0, fragment_max, f, t);

                    // then
                    assert_eq!(channels, empty_point_channels(&inputs, t));
                }
            }
        }
    }

    #[test]
    fn test_core_std_ignores_a_broad_background() {
        // given: a quarter of the values within +-1 on a regular grid (standard deviation 0.58), the rest over +-15
        let narrow = (0..250).map(|i| -1.0 + 2.0 * i as f32 / 249.0);
        let broad = (0..750).map(|i| -15.0 + 30.0 * i as f32 / 749.0);
        let values: Vec<f32> = narrow.chain(broad).collect();

        // when
        let core = core_std(&values, 15.0).unwrap();
        let mad = robust_std(&mut values.clone()).unwrap();

        // then
        assert!((0.4..0.9).contains(&core), "core_std {}", core);
        assert!(mad > 5.0, "robust_std {}", mad);
    }

    #[test]
    fn test_core_std_of_a_sample_without_background_is_its_spread() {
        // given: the quantiles of a standard normal distribution at 1 %, 3 %, ... 99 %, from its symmetric tails
        let half: [f32; 25] = [
            0.0251, 0.0753, 0.1257, 0.1764, 0.2275, 0.2793, 0.3319, 0.3853, 0.4399, 0.4959, 0.5534,
            0.6128, 0.6745, 0.7388, 0.8064, 0.8779, 0.9542, 1.0364, 1.1264, 1.2265, 1.3408, 1.4758,
            1.6449, 1.8808, 2.3263,
        ];
        let values: Vec<f32> = half.iter().flat_map(|&v| [-v, v]).collect();

        // when
        let core = core_std(&values, 7.0).unwrap();

        // then
        assert!((0.85..1.05).contains(&core), "core_std {}", core);
    }

    #[test]
    fn test_robust_std_of_a_normal_like_sample_is_its_spread() {
        // given
        let mut values: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        // when
        let spread = robust_std(&mut values).unwrap();

        // then
        assert!((spread - MAD_TO_STD).abs() < 1e-6);
    }
}
