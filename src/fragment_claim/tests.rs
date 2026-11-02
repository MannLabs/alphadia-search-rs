//! Given/When/Then tests of the claim features on a small synthetic run: two precursors in
//! one isolation window whose fragment traces overlap at three m/z, plus the cases that must
//! not claim (lower score, equal score, another window, another apex cycle, another rank of
//! the same precursor) and the edges of the index (overlapping windows, first and last cycle,
//! claim tolerance, empty input, thread count).

use super::algorithm::{compute_claim_features, ClaimFeatures, ClaimParameters};
use crate::candidate::{Candidate, CandidateCollection};
use crate::constants::{FragmentType, Loss};
use crate::dia_data::{AlphaRawView, DIAData};
use crate::dia_data_builder::DIADataBuilder;
use crate::speclib_flat::SpecLibFlat;
use approx::assert_abs_diff_eq;
use numpy::ndarray::{ArrayView1, ArrayView4};

const TOLERANCE_PPM: f32 = 10.0;
/// Elution profile over the five cycles of the run; every peak follows it.
const PROFILE: [f32; 5] = [1.0, 2.0, 4.0, 2.0, 1.0];
pub(crate) const WINDOW_LOW: (f32, f32) = (450.0, 550.0);
const WINDOW_HIGH: (f32, f32) = (650.0, 750.0);

/// Precursor A: five fragments, library intensities 5 down to 1.
pub(crate) const FRAGMENTS_A: [(f32, f32); 5] = [
    (300.0, 5.0),
    (400.0, 4.0),
    (500.0, 3.0),
    (600.0, 2.0),
    (700.0, 1.0),
];
/// Precursor B: shares the three most intense ones with A, plus one of its own.
pub(crate) const FRAGMENTS_B: [(f32, f32); 4] =
    [(300.0, 4.0), (400.0, 3.0), (500.0, 2.0), (830.0, 1.0)];

/// A run with one spectrum per window and cycle, every spectrum carrying peaks at the given
/// m/z values scaled by the profile of its cycle.
pub(crate) fn dia_data(windows: &[((f32, f32), &[f32])]) -> DIAData {
    let mut peak_mz = Vec::new();
    let mut peak_intensity = Vec::new();
    let mut spectrum_peak_start_idx = Vec::new();
    let mut spectrum_peak_stop_idx = Vec::new();
    let mut spectrum_delta_scan_idx = Vec::new();
    let mut spectrum_cycle_idx = Vec::new();
    let mut spectrum_rt = Vec::new();
    let mut isolation_lower_mz = Vec::new();
    let mut isolation_upper_mz = Vec::new();
    for (cycle, &scale) in PROFILE.iter().enumerate() {
        for (w, &((lower, upper), mzs)) in windows.iter().enumerate() {
            spectrum_peak_start_idx.push(peak_mz.len() as i64);
            for &mz in mzs {
                peak_mz.push(mz);
                peak_intensity.push(1000.0 * scale);
            }
            spectrum_peak_stop_idx.push(peak_mz.len() as i64);
            spectrum_delta_scan_idx.push(w as i64);
            spectrum_cycle_idx.push(cycle as i64);
            spectrum_rt.push(cycle as f32);
            isolation_lower_mz.push(lower);
            isolation_upper_mz.push(upper);
        }
    }
    let cycle = [0.0f32];
    let view = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&spectrum_delta_scan_idx),
        isolation_lower_mz: ArrayView1::from(&isolation_lower_mz),
        isolation_upper_mz: ArrayView1::from(&isolation_upper_mz),
        spectrum_peak_start_idx: ArrayView1::from(&spectrum_peak_start_idx),
        spectrum_peak_stop_idx: ArrayView1::from(&spectrum_peak_stop_idx),
        spectrum_cycle_idx: ArrayView1::from(&spectrum_cycle_idx),
        spectrum_rt: ArrayView1::from(&spectrum_rt),
        peak_mz: ArrayView1::from(&peak_mz),
        peak_intensity: ArrayView1::from(&peak_intensity),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &cycle).unwrap(),
    };
    DIADataBuilder::from_alpha_raw(&view)
}

/// A library from `(precursor_idx, precursor m/z, fragments as (m/z, intensity))`; fragment
/// type B and number 2 keep every fragment through the y1 filter.
pub(crate) fn library(precursors: &[(usize, f32, &[(f32, f32)])]) -> SpecLibFlat {
    let mut precursor_idx = Vec::new();
    let mut precursor_mz = Vec::new();
    let mut starts = Vec::new();
    let mut stops = Vec::new();
    let mut fragment_mz = Vec::new();
    let mut fragment_intensity = Vec::new();
    for &(idx, mz, fragments) in precursors {
        precursor_idx.push(idx);
        precursor_mz.push(mz);
        starts.push(fragment_mz.len());
        for &(frag_mz, intensity) in fragments {
            fragment_mz.push(frag_mz);
            fragment_intensity.push(intensity);
        }
        stops.push(fragment_mz.len());
    }
    let (np, nf) = (precursor_idx.len(), fragment_mz.len());
    SpecLibFlat::from_vecs(
        precursor_idx,
        precursor_mz.clone(),
        precursor_mz,
        vec![2.0; np],
        vec![2.0; np],
        vec![10; np],
        starts,
        stops,
        fragment_mz.clone(),
        fragment_mz,
        fragment_intensity,
        vec![1; nf],
        vec![1; nf],
        vec![Loss::NONE; nf],
        vec![2; nf],
        vec![2; nf],
        vec![FragmentType::B; nf],
    )
}

/// A candidate spanning the whole run with its apex at `cycle`.
pub(crate) fn candidate(precursor_idx: usize, rank: usize, score: f32, cycle: usize) -> Candidate {
    Candidate::new(precursor_idx, rank, score, 0, cycle, PROFILE.len())
}

fn params() -> ClaimParameters {
    ClaimParameters {
        mass_tolerance: TOLERANCE_PPM,
        top_k_fragments: 12,
        claim_tolerance: TOLERANCE_PPM,
        cycle_radius: 1,
        soft_claims: false,
    }
}

fn row(features: &[ClaimFeatures], precursor_idx: usize, rank: usize) -> &ClaimFeatures {
    features
        .iter()
        .find(|f| f.precursor_idx == precursor_idx && f.rank == rank)
        .unwrap()
}

pub(crate) const ALL_MZ: [f32; 7] = [300.0, 400.0, 500.0, 600.0, 700.0, 830.0, 900.0];

pub(crate) fn two_precursors_one_window() -> (DIAData, SpecLibFlat) {
    (
        dia_data(&[(WINDOW_LOW, &ALL_MZ)]),
        library(&[(0, 500.0, &FRAGMENTS_A), (1, 510.0, &FRAGMENTS_B)]),
    )
}

#[test]
fn the_lower_scoring_candidate_loses_its_shared_fragments() {
    // given two co-eluting precursors in one window sharing three observed fragment m/z
    let (dia, lib) = two_precursors_one_window();
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 20.0, 2), candidate(1, 1, 9.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then B's three shared fragments are claimed by A and only its own 830 stays free
    let b = row(&features, 1, 1);
    assert_eq!(b.n_matched, 4.0);
    assert_eq!(b.n_claimed, 3.0);
    assert_abs_diff_eq!(b.frac_claimed, 0.75);
    assert_abs_diff_eq!(b.frac_shared_any, 0.75);
    assert_abs_diff_eq!(b.lib_frac_claimed, 9.0 / 10.0);
    assert_abs_diff_eq!(b.intensity_frac_claimed, 0.75, epsilon = 1e-5);
    assert_eq!(b.top3_claimed, 3.0);
    assert_eq!(b.n_claimants, 1.0);
    assert_abs_diff_eq!(b.claimant_log_ratio, (20.0f32 / 9.0).ln(), epsilon = 1e-6);
    assert_eq!(b.num_over_0_free, 1.0);
    assert_eq!(b.num_over_50_free, 1.0);
    assert_abs_diff_eq!(b.mean_corr_free, 1.0, epsilon = 1e-5);
    // (a single free b ion has hyperscore ln(1!) + ln(1.0) = 0)
    assert_eq!(b.hyperscore_free, 0.0);
    // and A keeps everything: the shared fragments are only shared, not claimed
    let a = row(&features, 0, 1);
    assert_eq!(a.n_matched, 5.0);
    assert_eq!(a.n_claimed, 0.0);
    assert_abs_diff_eq!(a.frac_shared_any, 0.6);
    assert_eq!(a.n_claimants, 0.0);
    assert_eq!(a.claimant_log_ratio, 0.0);
    assert_eq!(a.num_over_0_free, 5.0);
    assert!(a.hyperscore_free > 0.0);
    assert!(a.log10_intensity_free > b.log10_intensity_free);
}

#[test]
fn an_equal_score_does_not_claim() {
    // given the same two precursors with equal ownership scores
    let (dia, lib) = two_precursors_one_window();
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 9.0, 2), candidate(1, 1, 9.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then nothing is claimed but the sharing is still visible
    for idx in [0, 1] {
        let f = row(&features, idx, 1);
        assert_eq!(f.n_claimed, 0.0);
        assert!(f.frac_shared_any > 0.0);
    }
}

#[test]
fn a_candidate_outside_the_cycle_radius_does_not_claim() {
    // given B's apex three cycles away from A's
    let (dia, lib) = two_precursors_one_window();
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 20.0, 1), candidate(1, 1, 9.0, 4)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then B is untouched
    let b = row(&features, 1, 1);
    assert_eq!(b.n_claimed, 0.0);
    assert_eq!(b.frac_shared_any, 0.0);
    // and with a radius of three it is claimed
    let wide = ClaimParameters {
        cycle_radius: 3,
        ..params()
    };
    let features = compute_claim_features(&dia, &lib, &candidates, &wide).unwrap();
    assert_eq!(row(&features, 1, 1).n_claimed, 3.0);
}

#[test]
fn another_window_never_claims() {
    // given precursor C in the high window with A's fragments, scoring higher than A
    let dia = dia_data(&[(WINDOW_LOW, &ALL_MZ), (WINDOW_HIGH, &ALL_MZ)]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A), (2, 700.0, &FRAGMENTS_A)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 5.0, 2), candidate(2, 1, 50.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then the windows are independent
    assert_eq!(row(&features, 0, 1).n_claimed, 0.0);
    assert_eq!(row(&features, 0, 1).frac_shared_any, 0.0);
    assert_eq!(row(&features, 2, 1).n_matched, 5.0);
}

#[test]
fn another_rank_of_the_same_precursor_never_claims() {
    // given two ranks of precursor A at the same apex, the second scoring higher
    let (dia, lib) = two_precursors_one_window();
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 5.0, 2), candidate(0, 2, 50.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then neither rank claims or shares with the other
    for rank in [1, 2] {
        let f = row(&features, 0, rank);
        assert_eq!(f.n_claimed, 0.0);
        assert_eq!(f.frac_shared_any, 0.0);
        assert_eq!(f.n_matched, 5.0);
    }
}

#[test]
fn a_precursor_without_signal_or_without_library_entry_is_all_zero() {
    // given a precursor whose fragments are not in the data and one that is not in the library
    let dia = dia_data(&[(WINDOW_LOW, &[300.0, 400.0])]);
    let lib = library(&[(0, 500.0, &[(123.0, 1.0), (234.0, 1.0)])]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 5.0, 2), candidate(7, 1, 5.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then both rows are the zero row (with their identity)
    for idx in [0, 7] {
        let f = row(&features, idx, 1);
        assert_eq!(
            *f,
            ClaimFeatures {
                precursor_idx: idx,
                rank: 1,
                ..Default::default()
            }
        );
    }
}

#[test]
fn parameters_are_validated() {
    assert!(ClaimParameters {
        mass_tolerance: 0.0,
        ..params()
    }
    .validate()
    .is_err());
    assert!(ClaimParameters {
        claim_tolerance: f32::NAN,
        ..params()
    }
    .validate()
    .is_err());
    assert!(ClaimParameters {
        top_k_fragments: 0,
        ..params()
    }
    .validate()
    .is_err());
    assert!(params().validate().is_ok());
}

#[test]
fn soft_claims_give_the_shared_peaks_to_the_precursor_that_explains_them() {
    // given A (five fragments) and B (three of its four fragments shared with A) at one apex
    let (dia, lib) = two_precursors_one_window();
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 9.0, 2), candidate(1, 1, 20.0, 2)]);
    let soft = ClaimParameters {
        soft_claims: true,
        ..params()
    };
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &soft).unwrap();
    // then B, whose own fragment 830 cannot explain its shared peaks, loses more of its
    // intensity than A, regardless of B's higher selection score
    let (a, b) = (row(&features, 0, 1), row(&features, 1, 1));
    assert!(
        b.soft_intensity_frac_claimed > a.soft_intensity_frac_claimed,
        "{} vs {}",
        b.soft_intensity_frac_claimed,
        a.soft_intensity_frac_claimed
    );
    assert!(b.soft_top3_frac_claimed > 0.0);
    assert!(b.soft_log10_intensity_own < a.soft_log10_intensity_own);
    // and without soft claims the columns are zero
    let hard = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    let b = row(&hard, 1, 1);
    assert_eq!(b.soft_intensity_frac_claimed, 0.0);
    assert_eq!(b.soft_log10_intensity_own, 0.0);
}

#[test]
fn only_the_first_valid_window_of_a_candidate_is_indexed() {
    // given overlapping windows: A (m/z 500) lies in both, B (m/z 560) only in the second,
    // and both windows carry the shared fragment peaks
    let window_overlap = (480.0, 580.0);
    let dia = dia_data(&[(WINDOW_LOW, &ALL_MZ), (window_overlap, &ALL_MZ)]);
    let lib = library(&[(0, 500.0, &FRAGMENTS_A), (1, 560.0, &FRAGMENTS_B)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 20.0, 2), candidate(1, 1, 9.0, 2)]);
    // when
    let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then A is indexed in its first window only, so B in the second window is not claimed
    let b = row(&features, 1, 1);
    assert_eq!(b.n_matched, 4.0);
    assert_eq!(b.n_claimed, 0.0);
    assert_eq!(b.frac_shared_any, 0.0);
    // and A's traces are summed over both windows, as in the scorer: five fragments of
    // 2 windows x 10,000 intensity each
    let a = row(&features, 0, 1);
    assert_eq!(a.n_matched, 5.0);
    assert_abs_diff_eq!(a.log10_intensity_free, 5.0, epsilon = 1e-5);
}

#[test]
fn apices_at_the_first_and_the_last_cycle_claim() {
    let (dia, lib) = two_precursors_one_window();
    let last = PROFILE.len() - 1;
    for (apex_a, apex_b) in [(0, 0), (last, last), (0, 1), (last, last - 1)] {
        // given A and B with their apices at the edges of the run
        let candidates = CandidateCollection::from_vec(vec![
            candidate(0, 1, 20.0, apex_a),
            candidate(1, 1, 9.0, apex_b),
        ]);
        // when
        let features = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
        // then the cycle range of the query is clamped and B loses its shared fragments
        assert_eq!(
            row(&features, 1, 1).n_claimed,
            3.0,
            "apices {apex_a} / {apex_b}"
        );
    }
}

#[test]
fn the_claim_tolerance_is_independent_of_the_extraction_tolerance() {
    // given B's most intense fragment observed 20 ppm away from A's, two separate peaks that
    // the 10 ppm extraction keeps apart
    let shifted_mz = 300.006;
    let all_mz = [300.0, shifted_mz, 400.0, 500.0, 600.0, 700.0, 830.0];
    let dia = dia_data(&[(WINDOW_LOW, &all_mz)]);
    let fragments_b = [(shifted_mz, 4.0), (400.0, 3.0), (500.0, 2.0), (830.0, 1.0)];
    let lib = library(&[(0, 500.0, &FRAGMENTS_A), (1, 510.0, &fragments_b)]);
    let candidates =
        CandidateCollection::from_vec(vec![candidate(0, 1, 20.0, 2), candidate(1, 1, 9.0, 2)]);
    // when the claim tolerance equals the extraction tolerance
    let narrow = compute_claim_features(&dia, &lib, &candidates, &params()).unwrap();
    // then the shifted fragment is B's own
    assert_eq!(row(&narrow, 1, 1).n_matched, 4.0);
    assert_eq!(row(&narrow, 1, 1).n_claimed, 2.0);
    // and with a 30 ppm claim tolerance it is the same ion as A's and claimed
    let wide = ClaimParameters {
        claim_tolerance: 30.0,
        ..params()
    };
    let wide = compute_claim_features(&dia, &lib, &candidates, &wide).unwrap();
    assert_eq!(row(&wide, 1, 1).n_matched, 4.0);
    assert_eq!(row(&wide, 1, 1).n_claimed, 3.0);
}

#[test]
fn an_empty_candidate_set_gives_no_rows() {
    // given no candidates
    let (dia, lib) = two_precursors_one_window();
    let candidates = CandidateCollection::from_vec(Vec::new());
    for soft_claims in [false, true] {
        let params = ClaimParameters {
            soft_claims,
            ..params()
        };
        // when / then
        let features = compute_claim_features(&dia, &lib, &candidates, &params).unwrap();
        assert!(features.is_empty());
    }
}

#[test]
fn the_output_does_not_depend_on_the_thread_count() {
    // given candidates in two windows, with two ranks of A and soft claims on
    let dia = dia_data(&[(WINDOW_LOW, &ALL_MZ), (WINDOW_HIGH, &ALL_MZ)]);
    let lib = library(&[
        (0, 500.0, &FRAGMENTS_A),
        (1, 510.0, &FRAGMENTS_B),
        (2, 700.0, &FRAGMENTS_A),
        (3, 710.0, &FRAGMENTS_B),
    ]);
    let candidates = CandidateCollection::from_vec(vec![
        candidate(0, 1, 20.0, 2),
        candidate(0, 2, 7.0, 1),
        candidate(1, 1, 9.0, 2),
        candidate(2, 1, 3.0, 3),
        candidate(3, 1, 12.0, 2),
    ]);
    let soft = ClaimParameters {
        soft_claims: true,
        ..params()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_claim_features(&dia, &lib, &candidates, &soft).unwrap())
    };
    // when
    let single = run(1);
    let parallel = run(8);
    // then the rows are identical, in the caller's order
    assert_eq!(single, parallel);
    let order: Vec<(usize, usize)> = single.iter().map(|f| (f.precursor_idx, f.rank)).collect();
    assert_eq!(order, vec![(0, 1), (0, 2), (1, 1), (2, 1), (3, 1)]);
}

#[test]
fn a_cycle_beyond_the_index_is_an_error() {
    // given a candidate whose apex cycle does not fit into the 16 bits of the index key
    let (dia, lib) = two_precursors_one_window();
    let cycle_beyond_index = 1 << 16;
    let candidates = CandidateCollection::from_vec(vec![Candidate::new(
        0,
        1,
        1.0,
        0,
        cycle_beyond_index,
        cycle_beyond_index + 1,
    )]);
    // when
    let result = compute_claim_features(&dia, &lib, &candidates, &params());
    // then
    assert!(result.unwrap_err().contains("cycles"));
}
