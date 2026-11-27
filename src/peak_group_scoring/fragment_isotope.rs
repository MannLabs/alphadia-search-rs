//! The fragments of a candidate whose isotope XICs can give isotope evidence.
//!
//! The M+1 XIC of a fragment is at +1.00336 Da / z from the fragment m/z, and its M-1 XIC is at -1.00336 Da / z.
//!
//! Decoys must not get isotope evidence that false targets cannot get. A decoy elutes in the window of its target, so
//! a decoy isotope XIC that overlaps a peak of its target picks up real signal. `isotope_shifted_ions` and
//! `isotope_eligible_fragments` exclude the fragments at which this can occur. The rules use only the library and the
//! elution groups and are the same for targets and decoys.

use crate::precursor::Precursor;
use crate::SpecLibFlat;

/// Mass difference between the monoisotopic peak and the first 13C isotope peak (Da)
pub const ISOTOPE_SPACING: f32 = 1.003_354_8;
/// Isotope peaks of a fragment, from M0, that can overlap the M+1 or M-1 XIC of a different fragment
const COLLIDING_ISOTOPE_PEAKS: u8 = 3;
/// Largest isotope shift between the same ion of two precursors at which the M-1..M+1 XICs of one ion overlap the
/// M0..M+2 peaks of the other
const MAX_PAIR_ISOTOPE_SHIFT: u8 = 3;

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

#[cfg(test)]
mod tests {
    use super::*;

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
