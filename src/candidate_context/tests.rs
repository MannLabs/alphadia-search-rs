//! These tests cover the log-m/z binning, the key packing, and the features on a small
//! synthetic run: two candidates that share three of six fragments at the same cycle, one
//! candidate three cycles away, one in another isolation window, and a second rank of the same
//! precursor.

use super::algorithm::{
    compute_context_features, log_mz_bin, log_mz_bin_width, pack_key, ContextFeatures,
    ContextParameters,
};
use crate::candidate::{Candidate, CandidateCollection};
use crate::constants::{FragmentType, Loss};
use crate::dia_data::DIAData;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;
use crate::speclib_flat::SpecLibFlat;
use approx::assert_abs_diff_eq;
use numpy::ndarray::{Array1, Array4};

const MASS_TOLERANCE_PPM: f32 = 10.0;
const WINDOW_LOW: (f32, f32) = (450.0, 550.0);
const WINDOW_HIGH: (f32, f32) = (650.0, 750.0);

/// Six fragments, intensities 6 down to 1.
const FRAGMENTS_A: [(f32, f32); 6] = [
    (300.0, 6.0),
    (400.0, 5.0),
    (500.0, 4.0),
    (600.0, 3.0),
    (700.0, 2.0),
    (800.0, 1.0),
];
/// Shares the first three m/z with `FRAGMENTS_A`. Those carry 9 of the 12 intensity units.
const FRAGMENTS_B: [(f32, f32); 6] = [
    (300.0, 3.0),
    (400.0, 3.0),
    (500.0, 3.0),
    (610.0, 1.0),
    (720.0, 1.0),
    (830.0, 1.0),
];

/// One isolation window per `(lower, upper)` pair. The observations carry no peaks, because the
/// features only read the window limits.
fn dia_data(windows: &[(f32, f32)]) -> DIAData {
    DIAData {
        rt_index: RTIndex::new(),
        quadrupole_observations: windows
            .iter()
            .map(|&(lower, upper)| {
                QuadrupoleObservation::new_with_capacity([lower, upper], 0, 0, 0)
            })
            .collect(),
        rt_values: Array1::zeros(0),
        cycle: Array4::zeros((0, 0, 0, 0)),
    }
}

/// A library from `(precursor_idx, precursor m/z, fragments as (m/z, intensity))`. Every other
/// field gets a placeholder, except the fragment type and number: those are B and 2, so that the
/// y1 filter of the scorer keeps every fragment.
fn library(precursors: &[(usize, f32, &[(f32, f32)])]) -> SpecLibFlat {
    let mut precursor_idx = Vec::new();
    let mut precursor_mz = Vec::new();
    let mut flat_frag_start_idx = Vec::new();
    let mut flat_frag_stop_idx = Vec::new();
    let mut fragment_mz = Vec::new();
    let mut fragment_intensity = Vec::new();
    for &(idx, mz, fragments) in precursors {
        precursor_idx.push(idx);
        precursor_mz.push(mz);
        flat_frag_start_idx.push(fragment_mz.len());
        for &(frag_mz, intensity) in fragments {
            fragment_mz.push(frag_mz);
            fragment_intensity.push(intensity);
        }
        flat_frag_stop_idx.push(fragment_mz.len());
    }
    let n_precursors = precursor_idx.len();
    let n_fragments = fragment_mz.len();
    SpecLibFlat::from_vecs(
        precursor_idx,
        precursor_mz.clone(),
        precursor_mz,
        vec![100.0; n_precursors],
        vec![100.0; n_precursors],
        vec![10; n_precursors],
        flat_frag_start_idx,
        flat_frag_stop_idx,
        fragment_mz.clone(),
        fragment_mz,
        fragment_intensity,
        vec![1; n_fragments],
        vec![1; n_fragments],
        vec![Loss::NONE; n_fragments],
        vec![2; n_fragments],
        vec![2; n_fragments],
        vec![FragmentType::B; n_fragments],
    )
}

/// The features only read the apex cycle, so start and stop can equal it.
fn candidate(precursor_idx: usize, rank: usize, score: f32, cycle: usize) -> Candidate {
    Candidate::new(precursor_idx, rank, score, cycle, cycle, cycle)
}

/// What the Python caller passes. A test that varies one parameter overrides that field.
fn params() -> ContextParameters {
    ContextParameters {
        mass_tolerance: MASS_TOLERANCE_PPM,
        top_k_fragments: 12,
        min_shared: 3,
        cycle_radius: 1,
    }
}

fn row(features: &[ContextFeatures], precursor_idx: usize, rank: usize) -> &ContextFeatures {
    features
        .iter()
        .find(|f| f.precursor_idx == precursor_idx && f.rank == rank)
        .unwrap()
}

/// No neighbour, no match, no competitor: only the claimant rank is non-zero, at its base of 1.
fn assert_no_competition(features: &ContextFeatures) {
    let expected = ContextFeatures {
        precursor_idx: features.precursor_idx,
        rank: features.rank,
        claimant_rank: 1.0,
        ..Default::default()
    };
    assert_eq!(*features, expected);
}

/// The scenario of `DOCS.md`: A (precursor 0) and B (precursor 1) share three fragments at
/// cycle 10, C (precursor 2) has the fragments of A three cycles later, D (precursor 3) has the
/// fragments of A at cycle 10 but lies in another isolation window.
fn scenario() -> (DIAData, SpecLibFlat, CandidateCollection) {
    let dia_data = dia_data(&[WINDOW_LOW, WINDOW_HIGH]);
    let lib = library(&[
        (0, 500.0, &FRAGMENTS_A),
        (1, 505.0, &FRAGMENTS_B),
        (2, 510.0, &FRAGMENTS_A),
        (3, 700.0, &FRAGMENTS_A),
    ]);
    let candidates = CandidateCollection::from_vec(vec![
        candidate(0, 0, 2.0, 10),
        candidate(1, 0, 1.0, 10),
        candidate(2, 0, 3.0, 13),
        candidate(3, 0, 5.0, 10),
    ]);
    (dia_data, lib, candidates)
}

/// The features of `scenario()` with the default parameters.
fn scenario_features() -> Vec<ContextFeatures> {
    let (dia_data, lib, candidates) = scenario();
    compute_context_features(&dia_data, &lib, &candidates, &params()).unwrap()
}

#[test]
fn test_log_mz_bin_is_same_or_adjacent_within_tolerance() {
    for tolerance_ppm in [1.0f32, 5.0, 10.0, 15.0, 30.0, 50.0] {
        let width = log_mz_bin_width(tolerance_ppm);
        let mut mz = 150.0f32;
        while mz < 2000.0 {
            let bin = log_mz_bin(mz, width);
            for fraction in [-1.0f32, -0.5, 0.5, 1.0] {
                let other = mz * (1.0 + fraction * tolerance_ppm * 1e-6);
                let other_bin = log_mz_bin(other, width);
                assert!(
                    bin.abs_diff(other_bin) <= 1,
                    "m/z {mz} and {other} at {tolerance_ppm} ppm fall into bins {bin} and {other_bin}"
                );
            }
            mz += 7.3;
        }
    }
}

#[test]
fn test_log_mz_bin_handles_non_positive_mz() {
    let width = log_mz_bin_width(10.0);
    assert_eq!(log_mz_bin(0.0, width), 0);
    assert_eq!(log_mz_bin(-5.0, width), 0);
}

#[test]
fn test_pack_key_follows_lexicographic_order() {
    let triples = [
        (0u32, 0u32, 0u32),
        (0, 0, 1),
        (0, 0, u32::MAX),
        (0, 1, 0),
        (0, 65_535, 5),
        (1, 0, 0),
        (7, 3, 9),
        (7, 4, 0),
        (65_535, 65_535, u32::MAX),
    ];
    let keys: Vec<u64> = triples
        .iter()
        .map(|&(window, cycle, bin)| pack_key(window, cycle, bin))
        .collect();

    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
    // Consecutive bins of one window and cycle are one contiguous key range.
    assert_eq!(pack_key(3, 20, 101), pack_key(3, 20, 100) + 1);
}

#[test]
fn test_two_candidates_sharing_three_fragments_compete() {
    let features = scenario_features();

    let a = row(&features, 0, 0);
    assert_eq!(a.candidate_density, 1.0);
    assert_eq!(a.n_competitors, 1.0);
    assert_eq!(a.n_competitors_higher, 0.0);
    assert_eq!(a.claimant_rank, 1.0);
    assert_eq!(a.shared_frac_any, 0.5);
    assert_eq!(a.shared_frac_higher, 0.0);
    assert_eq!(a.shared_lib_intensity_frac_higher, 0.0);
    assert_abs_diff_eq!(a.competitor_log_ratio, (1.0f32 / 2.0).ln(), epsilon = 1e-6);

    let b = row(&features, 1, 0);
    assert_eq!(b.candidate_density, 1.0);
    assert_eq!(b.n_competitors, 1.0);
    assert_eq!(b.n_competitors_higher, 1.0);
    assert_eq!(b.claimant_rank, 2.0);
    assert_eq!(b.shared_frac_any, 0.5);
    assert_eq!(b.shared_frac_higher, 0.5);
    assert_eq!(b.shared_lib_intensity_frac_higher, 0.75);
    assert_abs_diff_eq!(b.competitor_log_ratio, 2.0f32.ln(), epsilon = 1e-6);
}

#[test]
fn test_candidate_three_cycles_away_is_not_a_competitor() {
    assert_no_competition(row(&scenario_features(), 2, 0));
}

#[test]
fn test_candidate_in_another_window_is_not_a_competitor() {
    let features = scenario_features();

    assert_no_competition(row(&features, 3, 0));
    // D shares all six fragments of A, but the window keeps it out of A's features as well.
    assert_eq!(row(&features, 0, 0).n_competitors, 1.0);
    assert_eq!(row(&features, 0, 0).shared_frac_any, 0.5);
}

#[test]
fn test_result_keeps_candidate_order() {
    let features = scenario_features();

    let order: Vec<(usize, usize)> = features.iter().map(|f| (f.precursor_idx, f.rank)).collect();
    assert_eq!(order, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
}

#[test]
fn test_min_shared_threshold_separates_matches_from_competitors() {
    let (dia_data, lib, candidates) = scenario();
    let strict = ContextParameters {
        min_shared: 4,
        ..params()
    };

    let features = compute_context_features(&dia_data, &lib, &candidates, &strict).unwrap();

    let b = row(&features, 1, 0);
    assert_eq!(b.n_competitors, 0.0);
    assert_eq!(b.claimant_rank, 1.0);
    assert_eq!(b.shared_frac_higher, 0.0);
    assert_eq!(b.competitor_log_ratio, 0.0);
    // The three shared m/z are still matches, only not enough for a competitor.
    assert_eq!(b.shared_frac_any, 0.5);
    assert_eq!(b.candidate_density, 1.0);
}

#[test]
fn test_cycle_radius_limits_the_comparison() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A), (1, 505.0, &FRAGMENTS_B)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 0, 2.0, 10), candidate(1, 0, 1.0, 11)]);

    let within = compute_context_features(&dia_data, &lib, &candidates, &params()).unwrap();
    assert_eq!(row(&within, 1, 0).n_competitors, 1.0);
    assert_eq!(row(&within, 1, 0).candidate_density, 1.0);

    let same_cycle_only = ContextParameters {
        cycle_radius: 0,
        ..params()
    };
    let outside = compute_context_features(&dia_data, &lib, &candidates, &same_cycle_only).unwrap();
    assert_no_competition(row(&outside, 1, 0));
    assert_no_competition(row(&outside, 0, 0));
}

#[test]
fn test_other_rank_of_the_same_precursor_is_not_a_competitor() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 0, 2.0, 10), candidate(0, 1, 1.0, 11)]);

    let features = compute_context_features(&dia_data, &lib, &candidates, &params()).unwrap();

    for rank in 0..2 {
        let f = row(&features, 0, rank);
        assert_eq!(f.n_competitors, 0.0);
        assert_eq!(f.shared_frac_any, 0.0);
        // The other rank still crowds the window.
        assert_eq!(f.candidate_density, 1.0);
    }
}

#[test]
fn test_fragments_within_tolerance_match_across_bins() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let shifted: Vec<(f32, f32)> = FRAGMENTS_A
        .iter()
        .map(|&(mz, intensity)| (mz * (1.0 + 8e-6), intensity))
        .collect();
    let lib = library(&[(0, 500.0, &FRAGMENTS_A), (1, 505.0, &shifted)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 0, 2.0, 10), candidate(1, 0, 1.0, 10)]);

    // 8 ppm apart: inside a 10 ppm tolerance, outside a 5 ppm tolerance.
    let matched = compute_context_features(&dia_data, &lib, &candidates, &params()).unwrap();
    assert_eq!(row(&matched, 1, 0).shared_frac_any, 1.0);
    assert_eq!(row(&matched, 1, 0).n_competitors, 1.0);

    let narrow = ContextParameters {
        mass_tolerance: 5.0,
        ..params()
    };
    let unmatched = compute_context_features(&dia_data, &lib, &candidates, &narrow).unwrap();
    assert_eq!(row(&unmatched, 1, 0).shared_frac_any, 0.0);
    assert_eq!(row(&unmatched, 1, 0).n_competitors, 0.0);
}

#[test]
fn test_candidate_without_library_precursor_or_window_has_no_competition() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[
        (0, 500.0, &FRAGMENTS_A),
        (1, 505.0, &FRAGMENTS_B),
        (2, 900.0, &FRAGMENTS_A),
    ]);
    let candidates = CandidateCollection::from_vec(vec![
        candidate(0, 0, 2.0, 10),
        candidate(1, 0, 1.0, 10),
        // Precursor 2 lies in no isolation window, precursor 99 is not in the library.
        candidate(2, 0, 4.0, 10),
        candidate(99, 0, 4.0, 10),
    ]);

    let features = compute_context_features(&dia_data, &lib, &candidates, &params()).unwrap();

    assert_eq!(features.len(), 4);
    assert_no_competition(row(&features, 2, 0));
    assert_no_competition(row(&features, 99, 0));
    assert_eq!(row(&features, 1, 0).n_competitors, 1.0);
    assert_eq!(row(&features, 1, 0).candidate_density, 1.0);
}

#[test]
fn test_empty_candidates_give_empty_result() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A)]);

    let features =
        compute_context_features(&dia_data, &lib, &CandidateCollection::new(), &params()).unwrap();

    assert!(features.is_empty());
}

#[test]
fn test_invalid_parameters_are_rejected() {
    let (dia_data, lib, candidates) = scenario();

    for invalid in [
        ContextParameters {
            mass_tolerance: 0.0,
            ..params()
        },
        ContextParameters {
            top_k_fragments: 0,
            ..params()
        },
        ContextParameters {
            min_shared: 0,
            ..params()
        },
    ] {
        assert!(invalid.validate().is_err());
        // compute() runs the same validation before it reads the data.
        assert!(compute_context_features(&dia_data, &lib, &candidates, &invalid).is_err());
    }
}

#[test]
fn test_cycle_beyond_the_packed_key_is_rejected() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A)]);
    let candidates = CandidateCollection::from_vec(vec![candidate(0, 0, 2.0, 1 << 16)]);

    let result = compute_context_features(&dia_data, &lib, &candidates, &params());

    assert!(result.unwrap_err().contains("cycles"));
}

#[test]
fn test_huge_cycle_radius_still_counts_the_candidate_itself() {
    let dia_data = dia_data(&[WINDOW_LOW]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A)]);
    let candidates = CandidateCollection::from_vec(vec![candidate(0, 0, 2.0, 10)]);
    let everything = ContextParameters {
        cycle_radius: u32::MAX,
        ..params()
    };

    let features = compute_context_features(&dia_data, &lib, &candidates, &everything).unwrap();

    assert_no_competition(row(&features, 0, 0));
}
