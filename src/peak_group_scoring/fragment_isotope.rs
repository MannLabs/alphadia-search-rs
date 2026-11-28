//! Whether the isotope peaks next to a candidate's fragments confirm that the fragment peaks are real ions.
//!
//! A real fragment ion has its M+1 isotope peak at +1.00336 Da / z. This peak co-elutes with the fragment, at the
//! intensity ratio that the fragment mass predicts. A chance peak at the fragment m/z has no such partner.
//!
//! A real monoisotopic fragment has no peak at M-1. A co-eluting peak at M-1 shows that the matched peak can be the
//! M+1 isotope of a different ion. In HeLa Astral runs, the M-1 peak co-eluted with 2.2 % of the co-eluting
//! fragments of confident targets and with 6.4-6.9 % of those of decoys.
//!
//! The features use only the co-eluting fragments of the candidate. The isotope XICs have the tolerance and the
//! cycles of the fragment XICs.
//!
//! Decoys must not get isotope evidence that false targets cannot get. A decoy elutes in the window of its target, so
//! a decoy isotope XIC that overlaps a peak of its target picks up real signal. `isotope_shifted_ions` and
//! `isotope_eligible_fragments` exclude the fragments at which this can occur. The rules use only the library and the
//! elution groups and are the same for targets and decoys.

use numpy::ndarray::{ArrayView2, Axis};

use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::utils::{calculate_correlation_safe, COELUTION_MIN_CORRELATION};
use crate::precursor::Precursor;
use crate::SpecLibFlat;

/// Mass difference between the monoisotopic peak and the first 13C isotope peak (Da)
pub const ISOTOPE_SPACING: f32 = 1.003_354_8;
/// Proton mass (Da)
const PROTON_MASS: f32 = 1.007_276;
/// Expected M+1 / M intensity ratio per Da of fragment mass (averagine, Poisson approximation)
const MPLUS1_RATIO_PER_DA: f32 = 0.000_54;
/// The expected M+1 / M ratio is at least this value, so that the log ratio of very light fragments stays finite
const MIN_EXPECTED_MPLUS1_RATIO: f32 = 1e-3;
/// Maximum log ratio error of one fragment, so that a missing M+1 peak counts as a bounded error
pub const MAX_LOG_RATIO_ERROR: f32 = 3.0;
/// Isotope peaks of a fragment, from M0, that can overlap the M+1 or M-1 XIC of a different fragment
const COLLIDING_ISOTOPE_PEAKS: u8 = 3;
/// Largest isotope shift between the same ion of two precursors at which the M-1..M+1 XICs of one ion overlap the
/// M0..M+2 peaks of the other
const MAX_PAIR_ISOTOPE_SHIFT: u8 = 3;

feature_block! {
    /// The fragment-isotope features of one candidate.
    pub struct FragmentIsotopeFeatures {
        /// observed-intensity weighted mean correlation of the co-eluting fragments' XICs with their M+1 XICs
        pub mplus1_correlation: f32,
        /// library-intensity share of the co-eluting fragments whose M+1 co-elutes
        pub mplus1_lib_share: f32,
        /// number of co-eluting fragments whose M+1 co-elutes
        pub mplus1_count: f32,
        /// observed-intensity weighted mean |ln(observed M+1 / M ratio / expected ratio)| over the co-eluting
        /// fragments, at most MAX_LOG_RATIO_ERROR per fragment
        pub mplus1_ratio_error: f32,
        /// observed-intensity weighted mean correlation of the co-eluting fragments' XICs with their M-1 XICs
        pub mminus1_correlation: f32,
        /// library-intensity share of the co-eluting fragments whose M-1 co-elutes
        pub mminus1_lib_share: f32,
        /// number of co-eluting fragments whose M-1 co-elutes
        pub mminus1_count: f32,
        /// M-1 intensity as a share of the M-1 and M intensity, summed over the co-eluting fragments
        pub mminus1_intensity_share: f32,
    }
}

/// Key of one library ion of one precursor: precursor index and ion annotation.
pub fn ion_key(
    precursor_idx: usize,
    fragment_type: u8,
    number: u8,
    charge: u8,
    loss_type: u8,
) -> u64 {
    ((precursor_idx as u64) << 32)
        | ((fragment_type as u64) << 24)
        | ((number as u64) << 16)
        | ((charge as u64) << 8)
        | loss_type as u64
}

/// Whether two m/z windows of `mass_tolerance` ppm overlap.
fn windows_overlap(mz_a: f32, mz_b: f32, mass_tolerance: f32) -> bool {
    (mz_a - mz_b).abs() <= (mz_a + mz_b) * mass_tolerance * 1e-6
}

/// Whether the same ion of two precursors is shifted by 1 to MAX_PAIR_ISOTOPE_SHIFT isotope spacings. The result does
/// not depend on the order of the two ions, so that both precursors exclude the ion.
fn is_isotope_shift(mz_a: f32, mz_b: f32, charge: u8, mass_tolerance: f32) -> bool {
    let tolerance = (mz_a + mz_b) * mass_tolerance * 1e-6;
    let difference = (mz_a - mz_b).abs();
    if difference <= tolerance {
        return false;
    }
    let spacing = ISOTOPE_SPACING / charge.max(1) as f32;
    (1..=MAX_PAIR_ISOTOPE_SHIFT).any(|k| (difference - k as f32 * spacing).abs() <= tolerance)
}

/// The keys (see `ion_key`) of the ions of the precursor at array index `index` whose isotope XICs can overlap the
/// isotope peaks of the same ion of a different precursor of its elution group, sorted.
///
/// The precursors of an elution group elute together. A decoy is in the elution group of its target, so a decoy
/// isotope XIC that overlaps a peak of its target picks up real signal. Such an ion is excluded from the isotope
/// evidence of both precursors, because the relation is symmetric. The rule is the same for every precursor and does
/// not use decoy labels.
///
/// The XICs of an ion cover its M-1, M0 and M+1 m/z. The other precursor's ion has real peaks at M0, M+1
/// and M+2. So the XICs overlap if the two ions are 1 to MAX_PAIR_ISOTOPE_SHIFT isotope spacings apart. The same ion
/// at the same m/z belongs to a different charge state of the same peptide and does not count.
pub fn isotope_shifted_ions(lib: &SpecLibFlat, index: usize, mass_tolerance: f32) -> Vec<u64> {
    let own = lib.precursor_ions(index);
    let mut shifted = vec![false; own.mz_library.len()];
    for &other in lib.elution_group_members(index) {
        if other as usize == index {
            continue;
        }
        let other = lib.precursor_ions(other as usize);
        for i in 0..own.mz_library.len() {
            if shifted[i] || own.intensity[i] <= 0.0 {
                continue;
            }
            shifted[i] = (0..other.mz_library.len()).any(|j| {
                other.intensity[j] > 0.0
                    && other.number[j] == own.number[i]
                    && other.fragment_type[j] == own.fragment_type[i]
                    && other.charge[j] == own.charge[i]
                    && other.loss_type[j] == own.loss_type[i]
                    && is_isotope_shift(
                        own.mz_library[i],
                        other.mz_library[j],
                        own.charge[i],
                        mass_tolerance,
                    )
            });
        }
    }
    let mut keys: Vec<u64> = (0..own.mz_library.len())
        .filter(|&i| shifted[i])
        .map(|i| {
            ion_key(
                own.precursor_idx,
                own.fragment_type[i],
                own.number[i],
                own.charge[i],
                own.loss_type[i],
            )
        })
        .collect();
    keys.sort_unstable();
    keys
}

/// Whether each fragment of a precursor can give isotope evidence for its candidates.
///
/// Two kinds of fragments are not eligible:
/// - Ions in `isotope_shifted_ions` of the precursor, which overlap the isotope peaks of a different precursor of
///   the elution group.
/// - Fragments whose M+1 or M-1 XIC overlaps the M0, M+1 or M+2 peak of a different fragment of the same precursor.
///   For a real peptide, such an XIC shows the peptide's own signal and not the isotope evidence of the fragment.
pub fn isotope_eligible_fragments(
    precursor: &Precursor,
    mass_tolerance: f32,
    isotope_shifted_ions: &[u64],
) -> Vec<bool> {
    let spacing = |charge: u8| ISOTOPE_SPACING / charge.max(1) as f32;
    let fragment_mz = &precursor.fragment_mz;
    let fragment_charge = &precursor.fragment_charge;

    (0..fragment_mz.len())
        .map(|i| {
            let key = ion_key(
                precursor.precursor_idx,
                precursor.fragment_type[i],
                precursor.fragment_number[i],
                fragment_charge[i],
                precursor.fragment_loss_type[i],
            );
            if isotope_shifted_ions.binary_search(&key).is_ok() {
                return false;
            }
            let own_spacing = spacing(fragment_charge[i]);
            let neighbours = [fragment_mz[i] + own_spacing, fragment_mz[i] - own_spacing];
            let collides = (0..fragment_mz.len()).filter(|&j| j != i).any(|j| {
                (0..COLLIDING_ISOTOPE_PEAKS).any(|k| {
                    let peak = fragment_mz[j] + k as f32 * spacing(fragment_charge[j]);
                    neighbours
                        .iter()
                        .any(|&neighbour| windows_overlap(neighbour, peak, mass_tolerance))
                })
            });
            !collides
        })
        .collect()
}

/// The expected M+1 / M intensity ratio of a fragment.
fn expected_mplus1_ratio(fragment_mz: f32, fragment_charge: u8) -> f32 {
    let charge = fragment_charge.max(1) as f32;
    let fragment_mass = (fragment_mz - PROTON_MASS) * charge;
    (MPLUS1_RATIO_PER_DA * fragment_mass).max(MIN_EXPECTED_MPLUS1_RATIO)
}

/// The fragment-isotope features of one candidate.
///
/// `dense_xic`, `mplus1_xic` and `mminus1_xic` hold the fragment XICs and their isotope XICs (fragments x cycles).
/// The slices hold one value per scored fragment: library m/z, charge and intensity, observed intensity (summed over
/// the candidate's cycles), correlation with the median profile and eligibility (see `isotope_eligible_fragments`).
/// Fragments that are not eligible do not count, also not in the library-intensity total.
#[allow(clippy::too_many_arguments)]
pub fn calculate_fragment_isotope_features(
    dense_xic: ArrayView2<f32>,
    mplus1_xic: ArrayView2<f32>,
    mminus1_xic: ArrayView2<f32>,
    fragment_mz: &[f32],
    fragment_charge: &[u8],
    library_intensities: &[f32],
    observation_intensities: &[f32],
    correlations: &[f32],
    eligible: &[bool],
) -> FragmentIsotopeFeatures {
    let n = fragment_mz.len();
    let mut features = FragmentIsotopeFeatures {
        mplus1_ratio_error: MAX_LOG_RATIO_ERROR,
        ..Default::default()
    };
    if n == 0
        || fragment_charge.len() != n
        || library_intensities.len() != n
        || observation_intensities.len() != n
        || correlations.len() != n
        || eligible.len() != n
        || dense_xic.nrows() != n
        || mplus1_xic.dim() != dense_xic.dim()
        || mminus1_xic.dim() != dense_xic.dim()
    {
        return features;
    }

    let mplus1_intensities = mplus1_xic.sum_axis(Axis(1));
    let mminus1_intensities = mminus1_xic.sum_axis(Axis(1));
    let library_total: f32 = (0..n)
        .filter(|&i| eligible[i])
        .map(|i| library_intensities[i])
        .sum();

    let mut observed_total = 0.0f32;
    let mut mplus1_correlation_sum = 0.0f32;
    let mut mplus1_ratio_error_sum = 0.0f32;
    let mut mplus1_library = 0.0f32;
    let mut mminus1_correlation_sum = 0.0f32;
    let mut mminus1_library = 0.0f32;
    let mut mminus1_observed_total = 0.0f32;

    for i in 0..n {
        let observed = observation_intensities[i];
        if !eligible[i] || correlations[i] <= COELUTION_MIN_CORRELATION || observed <= 0.0 {
            continue;
        }
        let trace = dense_xic.row(i).to_vec();
        let mplus1_correlation = calculate_correlation_safe(&trace, &mplus1_xic.row(i).to_vec());
        let mminus1_correlation = calculate_correlation_safe(&trace, &mminus1_xic.row(i).to_vec());

        let mplus1_observed = mplus1_intensities[i];
        let mplus1_ratio_error = if mplus1_observed > 0.0 {
            ((mplus1_observed / observed)
                / expected_mplus1_ratio(fragment_mz[i], fragment_charge[i]))
            .ln()
            .abs()
            .min(MAX_LOG_RATIO_ERROR)
        } else {
            MAX_LOG_RATIO_ERROR
        };

        observed_total += observed;
        mplus1_correlation_sum += observed * mplus1_correlation;
        mplus1_ratio_error_sum += observed * mplus1_ratio_error;
        mminus1_correlation_sum += observed * mminus1_correlation;
        mminus1_observed_total += mminus1_intensities[i];
        if mplus1_correlation > COELUTION_MIN_CORRELATION {
            features.mplus1_count += 1.0;
            mplus1_library += library_intensities[i];
        }
        if mminus1_correlation > COELUTION_MIN_CORRELATION {
            features.mminus1_count += 1.0;
            mminus1_library += library_intensities[i];
        }
    }

    if observed_total > 0.0 {
        features.mplus1_correlation = mplus1_correlation_sum / observed_total;
        features.mplus1_ratio_error = mplus1_ratio_error_sum / observed_total;
        features.mminus1_correlation = mminus1_correlation_sum / observed_total;
        features.mminus1_intensity_share =
            mminus1_observed_total / (mminus1_observed_total + observed_total);
    }
    if library_total > 0.0 {
        features.mplus1_lib_share = mplus1_library / library_total;
        features.mminus1_lib_share = mminus1_library / library_total;
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Array2};

    const FRAGMENT_MZ: [f32; 2] = [1001.0, 1001.0];
    const FRAGMENT_CHARGE: [u8; 2] = [1, 1];
    const LIBRARY: [f32; 2] = [1.0, 1.0];
    const OBSERVED: [f32; 2] = [12.0, 24.0];

    fn fragment_xic() -> Array2<f32> {
        array![[1.0, 10.0, 1.0], [2.0, 20.0, 2.0]]
    }

    fn calculate(
        mplus1_xic: &Array2<f32>,
        mminus1_xic: &Array2<f32>,
        correlations: &[f32],
    ) -> FragmentIsotopeFeatures {
        calculate_fragment_isotope_features(
            fragment_xic().view(),
            mplus1_xic.view(),
            mminus1_xic.view(),
            &FRAGMENT_MZ,
            &FRAGMENT_CHARGE,
            &LIBRARY,
            &OBSERVED,
            correlations,
            &[true, true],
        )
    }

    #[test]
    fn test_mplus1_at_the_expected_ratio_confirms_every_fragment() {
        // Given: co-eluting M+1 traces at the averagine ratio of a 1000 Da fragment (about 0.54) and no M-1 traces
        let mplus1_xic = array![[0.54, 5.4, 0.54], [1.08, 10.8, 1.08]];
        let mminus1_xic = Array2::zeros((2, 3));

        // When: the features are calculated
        let features = calculate(&mplus1_xic, &mminus1_xic, &[0.9, 0.9]);

        // Then: the M+1 features confirm both fragments and the M-1 features are 0
        assert!(features.mplus1_correlation > 0.99);
        assert_eq!(features.mplus1_count, 2.0);
        assert!((features.mplus1_lib_share - 1.0).abs() < 1e-6);
        assert!(features.mplus1_ratio_error < 0.05);
        assert_eq!(features.mminus1_correlation, 0.0);
        assert_eq!(features.mminus1_count, 0.0);
        assert_eq!(features.mminus1_lib_share, 0.0);
        assert_eq!(features.mminus1_intensity_share, 0.0);
    }

    #[test]
    fn test_missing_mplus1_gives_the_maximum_ratio_error() {
        // Given: no isotope traces
        let empty_xic = Array2::zeros((2, 3));

        // When: the features are calculated
        let features = calculate(&empty_xic, &empty_xic, &[0.9, 0.9]);

        // Then: no fragment is confirmed and the ratio error is at its maximum
        assert_eq!(features.mplus1_count, 0.0);
        assert_eq!(features.mplus1_lib_share, 0.0);
        assert_eq!(features.mplus1_ratio_error, MAX_LOG_RATIO_ERROR);
    }

    #[test]
    fn test_coeluting_mminus1_flags_the_fragment() {
        // Given: a co-eluting M-1 trace at the first fragment only
        let mplus1_xic = Array2::zeros((2, 3));
        let mminus1_xic = array![[3.0, 30.0, 3.0], [0.0, 0.0, 0.0]];

        // When: the features are calculated
        let features = calculate(&mplus1_xic, &mminus1_xic, &[0.9, 0.9]);

        // Then: the M-1 features count the first fragment and its share of the intensity
        assert_eq!(features.mminus1_count, 1.0);
        assert!((features.mminus1_lib_share - 0.5).abs() < 1e-6);
        let first_fragment_weight = OBSERVED[0] / (OBSERVED[0] + OBSERVED[1]);
        assert!((features.mminus1_correlation - first_fragment_weight).abs() < 1e-3);
        let mminus1_observed = 36.0;
        let expected_share = mminus1_observed / (mminus1_observed + OBSERVED[0] + OBSERVED[1]);
        assert!((features.mminus1_intensity_share - expected_share).abs() < 1e-6);
    }

    #[test]
    fn test_fragments_that_do_not_coelute_are_left_out() {
        // Given: isotope traces at both fragments, but no fragment co-elutes with the median profile
        let isotope_xic = fragment_xic();

        // When: the features are calculated
        let features = calculate(&isotope_xic, &isotope_xic, &[0.1, 0.1]);

        // Then: the features are those of a candidate without isotope evidence
        assert_eq!(features.mplus1_count, 0.0);
        assert_eq!(features.mminus1_count, 0.0);
        assert_eq!(features.mplus1_correlation, 0.0);
        assert_eq!(features.mplus1_ratio_error, MAX_LOG_RATIO_ERROR);
    }

    #[test]
    fn test_mismatched_inputs_give_the_default_features() {
        // Given: isotope traces with fewer cycles than the fragment XICs
        let short_xic = Array2::zeros((2, 2));

        // When: the features are calculated
        let features = calculate(&short_xic, &short_xic, &[0.9, 0.9]);

        // Then: the features are those of a candidate without isotope evidence
        assert_eq!(features.mplus1_count, 0.0);
        assert_eq!(features.mplus1_ratio_error, MAX_LOG_RATIO_ERROR);
    }

    #[test]
    fn test_fragment_isotopes_of_an_ineligible_fragment_do_not_count() {
        // Given: co-eluting M+1 and M-1 traces at both fragments, but only the first fragment is eligible
        let isotope_xic = fragment_xic();

        // When: the features are calculated
        let features = calculate_fragment_isotope_features(
            fragment_xic().view(),
            isotope_xic.view(),
            isotope_xic.view(),
            &FRAGMENT_MZ,
            &FRAGMENT_CHARGE,
            &LIBRARY,
            &OBSERVED,
            &[0.9, 0.9],
            &[true, false],
        );

        // Then: only the first fragment counts, also in the library-intensity total
        assert_eq!(features.mplus1_count, 1.0);
        assert_eq!(features.mminus1_count, 1.0);
        assert!((features.mplus1_lib_share - 1.0).abs() < 1e-6);
        assert!((features.mminus1_intensity_share - 0.5).abs() < 1e-6);
    }

    use crate::constants::{FragmentType, Loss};

    const PRECURSOR_IDX: usize = 3;

    /// A precursor with singly charged y ions at `fragment_mz` and the given charges, numbered from 2
    fn precursor(fragment_mz: &[f32], fragment_charge: &[u8]) -> Precursor {
        let n = fragment_mz.len();
        Precursor {
            precursor_idx: PRECURSOR_IDX,
            mz: 500.0,
            mz_library: 500.0,
            rt: 10.0,
            rt_library: 10.0,
            naa: 20,
            fragment_mz: fragment_mz.to_vec(),
            fragment_mz_library: fragment_mz.to_vec(),
            fragment_intensity: vec![1.0; n],
            fragment_cardinality: vec![1; n],
            fragment_charge: fragment_charge.to_vec(),
            fragment_loss_type: vec![Loss::NONE; n],
            fragment_number: (2..2 + n as u8).collect(),
            fragment_position: (2..2 + n as u8).collect(),
            fragment_type: vec![FragmentType::Y; n],
        }
    }

    /// A library with one precursor per entry of `fragment_mz`, each with the same y ions 2, 3, .. at its m/z values,
    /// in the given elution groups
    fn library(fragment_mz: &[Vec<f32>], elution_groups: &[u64]) -> SpecLibFlat {
        let num_precursors = fragment_mz.len();
        let n = fragment_mz[0].len();
        let mz: Vec<f32> = fragment_mz.concat();
        SpecLibFlat::from_vecs(
            (0..num_precursors).collect(),
            vec![500.0; num_precursors],
            vec![500.0; num_precursors],
            vec![10.0; num_precursors],
            vec![10.0; num_precursors],
            vec![20; num_precursors],
            elution_groups.to_vec(),
            (0..num_precursors).map(|p| p * n).collect(),
            (0..num_precursors).map(|p| (p + 1) * n).collect(),
            mz.clone(),
            mz,
            vec![1.0; num_precursors * n],
            vec![1; num_precursors * n],
            vec![1; num_precursors * n],
            vec![Loss::NONE; num_precursors * n],
            (0..num_precursors).flat_map(|_| 2..2 + n as u8).collect(),
            (0..num_precursors).flat_map(|_| 2..2 + n as u8).collect(),
            vec![FragmentType::Y; num_precursors * n],
        )
    }

    fn y_ion_key(precursor_idx: usize, number: u8) -> u64 {
        ion_key(precursor_idx, FragmentType::Y, number, 1, Loss::NONE)
    }

    #[test]
    fn test_ions_two_isotope_spacings_from_the_same_ion_in_the_elution_group_are_shifted() {
        // Given: an elution group with a precursor, a second charge state with the same ions and a precursor whose y3
        // is two isotope spacings heavier
        let lib = library(
            &[
                vec![400.0, 1500.0],
                vec![400.0, 1500.0],
                vec![414.0, 1500.0 + 2.0 * ISOTOPE_SPACING],
            ],
            &[7, 7, 7],
        );

        // When: the isotope-shifted ions of each precursor are found
        let shifted: Vec<Vec<u64>> = (0..3).map(|i| isotope_shifted_ions(&lib, i, 7.0)).collect();

        // Then: the y3 ion of every precursor is shifted, and the y2 ions and the identical ions are not
        assert_eq!(
            shifted,
            vec![
                vec![y_ion_key(0, 3)],
                vec![y_ion_key(1, 3)],
                vec![y_ion_key(2, 3)]
            ]
        );
    }

    #[test]
    fn test_ions_of_different_elution_groups_are_not_compared() {
        // Given: two precursors whose y3 ions are one isotope spacing apart, in different elution groups
        let lib = library(
            &[vec![400.0, 1500.0], vec![414.0, 1500.0 + ISOTOPE_SPACING]],
            &[7, 8],
        );

        // When: the isotope-shifted ions of each precursor are found
        let shifted: Vec<Vec<u64>> = (0..2).map(|i| isotope_shifted_ions(&lib, i, 7.0)).collect();

        // Then: no ion is shifted
        assert_eq!(shifted, vec![Vec::<u64>::new(), Vec::new()]);
    }

    #[test]
    fn test_elution_groups_need_not_be_sorted() {
        // Given: precursors 0 and 2 in elution group 9, and precursor 1 in elution group 4, with the y3 ions of 0
        // and 2 one isotope spacing apart
        let lib = library(
            &[
                vec![400.0, 1500.0],
                vec![414.0, 1500.0 + ISOTOPE_SPACING],
                vec![428.0, 1500.0 + ISOTOPE_SPACING],
            ],
            &[9, 4, 9],
        );

        // When: the isotope-shifted ions of each precursor are found
        let shifted: Vec<Vec<u64>> = (0..3).map(|i| isotope_shifted_ions(&lib, i, 7.0)).collect();

        // Then: the y3 ions of precursors 0 and 2 are shifted, precursor 1 is in a different group
        assert_eq!(
            shifted,
            vec![vec![y_ion_key(0, 3)], Vec::new(), vec![y_ion_key(2, 3)]]
        );
        assert_eq!(lib.elution_group_members(1), &[1]);
    }

    #[test]
    fn test_shifted_ions_are_not_eligible() {
        // Given: a precursor whose y3 is an isotope-shifted ion
        let precursor = precursor(&[400.0, 900.0], &[1, 1]);
        let shifted = vec![y_ion_key(PRECURSOR_IDX, 3)];

        // When: the eligibility is calculated
        let eligible = isotope_eligible_fragments(&precursor, 7.0, &shifted);

        // Then: y3 is not eligible
        assert_eq!(eligible, vec![true, false]);
    }

    #[test]
    fn test_fragments_whose_isotope_xic_overlaps_a_peak_of_another_fragment_are_not_eligible() {
        // Given: a fragment at 500, a fragment at its M+1 m/z, a doubly charged fragment two isotope spacings below
        // 800 and a distant fragment
        let precursor = precursor(
            &[
                500.0,
                500.0 + ISOTOPE_SPACING,
                800.0 - ISOTOPE_SPACING,
                800.0,
                1100.0,
            ],
            &[1, 1, 2, 1, 1],
        );

        // When: the eligibility is calculated
        let eligible = isotope_eligible_fragments(&precursor, 7.0, &[]);

        // Then: the first two fragments overlap each other, the 800 fragment's M-1 overlaps the M+2 peak of the
        // doubly charged fragment, and the distant fragment is eligible
        assert_eq!(eligible, vec![false, false, true, false, true]);
    }

    #[test]
    fn test_isotope_shift_does_not_depend_on_the_order_of_the_ions() {
        // Given: pairs of ions near the edge of the tolerance window of one and two isotope spacings
        let pairs: Vec<(f32, f32)> = (0..2000)
            .map(|i| {
                let mz = 300.0 + i as f32 * 0.7;
                let offset = (i % 7) as f32 * 0.002 - 0.006;
                (mz, mz + (1 + i % 2) as f32 * ISOTOPE_SPACING + offset)
            })
            .collect();

        // When / Then: both orders give the same result
        for (a, b) in pairs {
            assert_eq!(
                is_isotope_shift(a, b, 1, 7.0),
                is_isotope_shift(b, a, 1, 7.0),
                "{a} {b}"
            );
        }
    }
}
