use super::*;
use crate::constants::{FragmentType, Loss};
use approx::assert_relative_eq;
use numpy::PyArrayMethods;

#[test]
fn test_filter_sort_fragments_no_filtering() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![100.1, 200.1, 300.1, 400.1, 500.1];
    let fragment_intensity = vec![0.0, 15.0, 10.0, 20.0, 5.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        true, // Filter Y1 ions to broaden test scope
        false,
        usize::MAX,
    );

    assert_eq!(mz, fragment_mz);
    assert_eq!(intensity, fragment_intensity);
}

#[test]
fn test_filter_sort_fragments_non_zero_only() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![0.0, 15.0, 10.0, 0.0, 5.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        true,
        true, // Filter Y1 ions to broaden test scope
        false,
        usize::MAX,
    );

    assert_eq!(mz, vec![200.0, 300.0, 500.0]);
    assert_eq!(intensity, vec![15.0, 10.0, 5.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_filter_sort_fragments_top_k_selection() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![5.0, 15.0, 10.0, 20.0, 8.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        false,
        3,
    );

    // Top 3: 20.0, 15.0, 10.0 sorted by fragment_mz ascending
    assert_eq!(mz, vec![200.0, 300.0, 400.0]);
    assert_eq!(intensity, vec![15.0, 10.0, 20.0]);
}

#[test]
fn test_filter_sort_fragments_combined_non_zero_and_top_k() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![0.0, 15.0, 10.0, 0.0, 20.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        true,
        false,
        false,
        2,
    );

    // Non-zero: [15.0, 10.0, 20.0], top 2: [20.0, 15.0] sorted by fragment_mz ascending
    assert_eq!(mz, vec![200.0, 500.0]);
    assert_eq!(intensity, vec![15.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_filter_sort_fragments_mz_ascending_order() {
    let fragment_mz = vec![600.0, 200.0, 800.0, 100.0, 400.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![15.0, 25.0, 5.0, 30.0, 20.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        false,
        3,
    );

    // Top 3: 30.0, 25.0, 20.0 sorted by fragment_mz ascending: 100.0, 200.0, 400.0
    assert_eq!(mz, vec![100.0, 200.0, 400.0]);
    assert_eq!(intensity, vec![30.0, 25.0, 20.0]);

    // Verify fragments are sorted by fragment_mz in ascending order
    assert!(mz.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn test_filter_sort_fragments_identical_intensities() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![10.0, 15.0, 10.0, 10.0, 10.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        false,
        3,
    );

    // Should get highest intensity (15.0) and 2 of the 10.0s sorted by fragment_mz ascending
    assert_eq!(mz.len(), 3);
    assert_eq!(intensity.len(), 3);
    assert!(intensity.contains(&15.0));

    // Count the 10.0s (should be 2)
    let count_tens = intensity.iter().filter(|&&x| x == 10.0).count();
    assert_eq!(count_tens, 2);

    // Verify fragments are sorted by fragment_mz in ascending order
    assert!(mz.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn test_filter_sort_fragments_empty_input() {
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        false,
        false,
        false,
        5,
    );

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_sort_fragments_single_fragment() {
    // Non-zero single fragment
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &[500.0],
        &[500.1],
        &[10.0],
        &[1],
        &[1],
        &[Loss::NONE],
        &[1],
        &[1],
        &[FragmentType::B],
        true,
        false,
        false,
        1,
    );
    assert_eq!(mz, vec![500.0]);
    assert_eq!(intensity, vec![10.0]);

    // Zero single fragment with non-zero filter
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &[500.0],
        &[500.1],
        &[0.0],
        &[1],
        &[1],
        &[Loss::NONE],
        &[1],
        &[1],
        &[FragmentType::B],
        true,
        false,
        false,
        1,
    );
    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_sort_fragments_all_zero_intensities() {
    let fragment_mz = vec![100.0, 200.0, 300.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![0.0, 0.0, 0.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        true,
        true, // Filter Y1 ions to broaden test scope
        false,
        usize::MAX,
    );

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_sort_fragments_top_k_zero() {
    let fragment_mz = vec![100.0, 200.0, 300.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![10.0, 20.0, 15.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        false,
        0,
    );

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_sort_fragments_top_k_larger_than_available() {
    let fragment_mz = vec![100.0, 200.0];
    let fragment_mz_library = vec![100.1; fragment_mz.len()]; // dummy data
    let fragment_intensity = vec![10.0, 20.0];
    let fragment_cardinality = vec![1u8; fragment_mz.len()];
    let fragment_charge = vec![1u8; fragment_mz.len()];
    let fragment_loss_type = vec![Loss::NONE; fragment_mz.len()];
    let fragment_number = vec![1u8; fragment_mz.len()];
    let fragment_position = vec![1u8; fragment_mz.len()];
    let fragment_type = vec![FragmentType::B; fragment_mz.len()];

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        false,
        10,
    );

    assert_eq!(mz, fragment_mz);
    assert_eq!(intensity, fragment_intensity);
}

#[test]
fn test_filter_sort_fragments_invariants() {
    // Test that all fundamental invariants hold across various parameter combinations
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![5.0, 0.0, 15.0, 10.0, 20.0];

    for non_zero in [false, true] {
        for k in [0, 1, 2, 3, 10] {
            let fragment_mz_library = vec![100.1; fragment_mz.len()];
            let fragment_cardinality = vec![1u8; fragment_mz.len()];
            let fragment_charge = vec![1u8; fragment_mz.len()];
            let fragment_loss_type = vec![0u8; fragment_mz.len()];
            let fragment_number = vec![1u8; fragment_mz.len()];
            let fragment_position = vec![1u8; fragment_mz.len()];
            let fragment_type = vec![1u8; fragment_mz.len()];

            let (filtered_mz, _filtered_mz_library, filtered_intensity, _, _, _, _, _, _) =
                filter_sort_fragments(
                    &fragment_mz,
                    &fragment_mz_library,
                    &fragment_intensity,
                    &fragment_cardinality,
                    &fragment_charge,
                    &fragment_loss_type,
                    &fragment_number,
                    &fragment_position,
                    &fragment_type,
                    non_zero,
                    false,
                    false,
                    k,
                );

            // Invariant: Output vectors have same length
            assert_eq!(filtered_mz.len(), filtered_intensity.len());

            // Invariant: All fragments come from original set
            for (&mz, &intensity) in filtered_mz.iter().zip(filtered_intensity.iter()) {
                let original_idx = fragment_mz.iter().position(|&x| x == mz).unwrap();
                assert_eq!(fragment_intensity[original_idx], intensity);
            }

            // Invariant: Non-zero filtering works correctly
            if non_zero {
                assert!(filtered_intensity.iter().all(|&x| x > 0.0));
            }

            // Invariant: Top-k constraint is respected
            assert!(filtered_mz.len() <= k);

            // Invariant: Original ordering is preserved
            if filtered_mz.len() > 1 {
                for i in 1..filtered_mz.len() {
                    let idx1 = fragment_mz
                        .iter()
                        .position(|&x| x == filtered_mz[i - 1])
                        .unwrap();
                    let idx2 = fragment_mz
                        .iter()
                        .position(|&x| x == filtered_mz[i])
                        .unwrap();
                    assert!(idx1 < idx2);
                }
            }
        }
    }
}

#[test]
fn test_speclib_flat_creation_sorting() {
    use numpy::PyArray1;
    use pyo3::Python;

    Python::initialize();
    Python::attach(|py| {
        // Create unsorted test data - precursor_idx intentionally out of order
        let precursor_idx = PyArray1::from_slice(py, &[3usize, 1, 4, 2]);
        let precursor_mz = PyArray1::from_slice(py, &[300.0f32, 100.0, 400.0, 200.0]);
        let precursor_rt = PyArray1::from_slice(py, &[30.0f32, 10.0, 40.0, 20.0]);
        let precursor_naa = PyArray1::from_slice(py, &[15u8, 10, 20, 12]);
        let flat_frag_start_idx = PyArray1::from_slice(py, &[6usize, 0, 9, 3]);
        let flat_frag_stop_idx = PyArray1::from_slice(py, &[9usize, 3, 12, 6]);
        let fragment_mz = PyArray1::from_slice(
            py,
            &[
                // Fragments for precursor 1 (idx 0-3)
                101.0f32, 102.0, 103.0, // Fragments for precursor 3 (idx 3-6)
                301.0, 302.0, 303.0, // Fragments for precursor 3 (idx 6-9)
                311.0, 312.0, 313.0, // Fragments for precursor 4 (idx 9-12)
                401.0, 402.0, 403.0,
            ],
        );
        let fragment_intensity = PyArray1::from_slice(
            py,
            &[
                10.0f32, 11.0, 12.0, // precursor 1
                30.0, 31.0, 32.0, // precursor 3
                33.0, 34.0, 35.0, // precursor 3
                40.0, 41.0, 42.0, // precursor 4
            ],
        );
        let fragment_cardinality = PyArray1::from_slice(py, &[1u8; 12]);
        let fragment_charge = PyArray1::from_slice(py, &[1u8; 12]);
        let fragment_loss_type = PyArray1::from_slice(py, &[Loss::NONE; 12]);
        let fragment_number = PyArray1::from_slice(py, &[1u8; 12]);
        let fragment_position = PyArray1::from_slice(py, &[1u8; 12]);
        let fragment_type = PyArray1::from_slice(py, &[FragmentType::B; 12]);

        let speclib = SpecLibFlat::from_arrays(
            precursor_idx.readonly(),
            precursor_mz.readonly(), // library
            precursor_mz.readonly(), // observed - reusing library values for test
            precursor_rt.readonly(), // library
            precursor_rt.readonly(), // observed - reusing library values for test
            precursor_naa.readonly(),
            flat_frag_start_idx.readonly(),
            flat_frag_stop_idx.readonly(),
            fragment_mz.readonly(), // library
            fragment_mz.readonly(), // observed - reusing library values for test
            fragment_intensity.readonly(),
            fragment_cardinality.readonly(),
            fragment_charge.readonly(),
            fragment_loss_type.readonly(),
            fragment_number.readonly(),
            fragment_position.readonly(),
            fragment_type.readonly(),
        );

        // Verify precursor_idx is now sorted
        let precursor_1 = speclib.get_precursor(0);
        let precursor_2 = speclib.get_precursor(1);
        let precursor_3 = speclib.get_precursor(2);
        let precursor_4 = speclib.get_precursor(3);

        assert_eq!(precursor_1.precursor_idx, 1);
        assert_eq!(precursor_2.precursor_idx, 2);
        assert_eq!(precursor_3.precursor_idx, 3);
        assert_eq!(precursor_4.precursor_idx, 4);

        // Verify corresponding data was reordered correctly
        assert_eq!(precursor_1.mz, 100.0);
        assert_eq!(precursor_2.mz, 200.0);
        assert_eq!(precursor_3.mz, 300.0);
        assert_eq!(precursor_4.mz, 400.0);

        assert_eq!(precursor_1.rt, 10.0);
        assert_eq!(precursor_2.rt, 20.0);
        assert_eq!(precursor_3.rt, 30.0);
        assert_eq!(precursor_4.rt, 40.0);
    });
}

#[test]
fn test_speclib_flat_fragment_mz_sorting() {
    // Test that SpecLibFlat sorts fragments by fragment_mz in ascending order
    // when using filter_sort_fragments during construction

    // Create test data with unsorted fragment_mz values
    let precursor_idx = vec![0];
    let precursor_mz = vec![500.0];
    let precursor_mz_library = vec![500.1];
    let precursor_rt = vec![100.0];
    let precursor_rt_library = vec![100.1];
    let precursor_naa = vec![10];
    let precursor_start_idx = vec![0];
    let precursor_stop_idx = vec![5];

    // Unsorted fragment_mz values with corresponding intensities
    let fragment_mz = vec![400.0, 100.0, 300.0, 200.0, 500.0];
    let fragment_mz_library = vec![400.1, 100.1, 300.1, 200.1, 500.1];
    let fragment_intensity = vec![20.0, 30.0, 15.0, 25.0, 10.0];
    let fragment_cardinality = vec![1u8; 5];
    let fragment_charge = vec![1u8; 5];
    let fragment_loss_type = vec![Loss::NONE; 5];
    let fragment_number = vec![1u8; 5];
    let fragment_position = vec![1u8; 5];
    let fragment_type = vec![FragmentType::B; 5];

    pyo3::Python::initialize();

    // Create numpy arrays for the test data
    use numpy::PyArray1;

    pyo3::Python::attach(|py| {
        let precursor_idx_arr = PyArray1::from_slice(py, &precursor_idx);
        let precursor_mz_arr = PyArray1::from_slice(py, &precursor_mz);
        let precursor_mz_library_arr = PyArray1::from_slice(py, &precursor_mz_library);
        let precursor_rt_arr = PyArray1::from_slice(py, &precursor_rt);
        let precursor_rt_library_arr = PyArray1::from_slice(py, &precursor_rt_library);
        let precursor_naa_arr = PyArray1::from_slice(py, &precursor_naa);
        let precursor_start_idx_arr = PyArray1::from_slice(py, &precursor_start_idx);
        let precursor_stop_idx_arr = PyArray1::from_slice(py, &precursor_stop_idx);
        let fragment_mz_arr = PyArray1::from_slice(py, &fragment_mz);
        let fragment_mz_library_arr = PyArray1::from_slice(py, &fragment_mz_library);
        let fragment_intensity_arr = PyArray1::from_slice(py, &fragment_intensity);
        let fragment_cardinality_arr = PyArray1::from_slice(py, &fragment_cardinality);
        let fragment_charge_arr = PyArray1::from_slice(py, &fragment_charge);
        let fragment_loss_type_arr = PyArray1::from_slice(py, &fragment_loss_type);
        let fragment_number_arr = PyArray1::from_slice(py, &fragment_number);
        let fragment_position_arr = PyArray1::from_slice(py, &fragment_position);
        let fragment_type_arr = PyArray1::from_slice(py, &fragment_type);

        let speclib = SpecLibFlat::from_arrays(
            precursor_idx_arr.readonly(),
            precursor_mz_arr.readonly(),
            precursor_mz_library_arr.readonly(),
            precursor_rt_arr.readonly(),
            precursor_rt_library_arr.readonly(),
            precursor_naa_arr.readonly(),
            precursor_start_idx_arr.readonly(),
            precursor_stop_idx_arr.readonly(),
            fragment_mz_library_arr.readonly(),
            fragment_mz_arr.readonly(),
            fragment_intensity_arr.readonly(),
            fragment_cardinality_arr.readonly(),
            fragment_charge_arr.readonly(),
            fragment_loss_type_arr.readonly(),
            fragment_number_arr.readonly(),
            fragment_position_arr.readonly(),
            fragment_type_arr.readonly(),
        );

        let precursor = speclib.get_precursor_filtered(0, false, false, false, usize::MAX);

        // Verify that fragments are sorted by fragment_mz in ascending order
        assert_eq!(
            precursor.fragment_mz,
            vec![100.0, 200.0, 300.0, 400.0, 500.0]
        );

        // Verify corresponding intensities are correctly matched
        assert_eq!(
            precursor.fragment_intensity,
            vec![30.0, 25.0, 15.0, 20.0, 10.0]
        );

        // Verify ascending order
        assert!(precursor.fragment_mz.windows(2).all(|w| w[0] <= w[1]));

        // Test filtering with non_zero = true
        let precursor_filtered = speclib.get_precursor_filtered(0, true, false, false, usize::MAX);
        assert!(precursor_filtered
            .fragment_intensity
            .iter()
            .all(|&x| x > 0.0));
        assert_eq!(precursor_filtered.fragment_mz.len(), 5); // All intensities are > 0

        // Test filtering with top_k = 3
        let precursor_top3 = speclib.get_precursor_filtered(0, false, false, false, 3);
        assert_eq!(precursor_top3.fragment_mz.len(), 3);
        // Should contain top 3 intensities: 30.0, 25.0, 20.0 sorted by mz: 100.0, 200.0, 400.0
        assert_eq!(precursor_top3.fragment_mz, vec![100.0, 200.0, 400.0]);
        assert_eq!(precursor_top3.fragment_intensity, vec![30.0, 25.0, 20.0]);

        // Test combined filtering: non_zero = true and top_k = 2
        let precursor_combined = speclib.get_precursor_filtered(0, true, false, false, 2);
        assert_eq!(precursor_combined.fragment_mz.len(), 2);
        assert!(precursor_combined
            .fragment_intensity
            .iter()
            .all(|&x| x > 0.0));
        // Should contain top 2 intensities: 30.0, 25.0 sorted by mz: 100.0, 200.0
        assert_eq!(precursor_combined.fragment_mz, vec![100.0, 200.0]);
        assert_eq!(precursor_combined.fragment_intensity, vec![30.0, 25.0]);
    });
}

#[test]
fn test_filter_sort_fragments_y1_ion_filtering() {
    // Test that y1 ions (fragment_type = Y and fragment_number = 1) are filtered out
    let fragment_mz = vec![150.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_mz_library = vec![150.1, 200.1, 300.1, 400.1, 500.1];
    let fragment_intensity = vec![10.0, 15.0, 20.0, 25.0, 30.0]; // All non-zero
    let fragment_cardinality = vec![1u8; 5];
    let fragment_charge = vec![1u8; 5];
    let fragment_loss_type = vec![Loss::NONE; 5];

    // Mix of fragment numbers with y1 ion included
    let fragment_number = vec![1, 2, 3, 1, 2]; // Two y1 ions at indices 0 and 3
    let fragment_position = vec![1u8; 5];

    // Mix of fragment types: y1, y2, b3, y1, b2
    let fragment_type = vec![
        FragmentType::Y, // y1 - should be filtered
        FragmentType::Y, // y2 - should be kept
        FragmentType::B, // b3 - should be kept
        FragmentType::Y, // y1 - should be filtered
        FragmentType::B, // b2 - should be kept
    ];

    let (mz, _mz_library, intensity, _, _, _, number, _, frag_type) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false, // Don't filter zero intensities
        true,  // Filter Y1 ions - this is what we're testing!
        false,
        usize::MAX, // No top-k limit
    );

    // Should have 3 fragments remaining (original 5 minus 2 y1 ions)
    assert_eq!(mz.len(), 3);
    assert_eq!(intensity.len(), 3);

    // Verify no y1 ions remain (no fragments with fragment_type=Y AND fragment_number=1)
    for i in 0..frag_type.len() {
        if frag_type[i] == FragmentType::Y {
            assert_ne!(
                number[i], 1,
                "Found y1 ion that should have been filtered out"
            );
        }
    }

    // Expected remaining fragments (sorted by mz): y2(200.0), b3(300.0), b2(500.0)
    assert_eq!(mz, vec![200.0, 300.0, 500.0]);
    assert_eq!(intensity, vec![15.0, 20.0, 30.0]);

    // Verify the remaining fragments are the expected ones
    assert_eq!(
        frag_type,
        vec![FragmentType::Y, FragmentType::B, FragmentType::B]
    );
    assert_eq!(number, vec![2, 3, 2]); // y2, b3, b2
}

#[test]
fn test_filter_sort_fragments_short_ion_filtering_before_top_k() {
    // Given: b1, b2, y1 and y2 as the most intense fragments, then b3, y3 and a2
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0];
    let fragment_mz_library = fragment_mz.clone();
    let fragment_intensity = vec![70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0];
    let fragment_cardinality = vec![1u8; 7];
    let fragment_charge = vec![1u8; 7];
    let fragment_loss_type = vec![Loss::NONE; 7];
    let fragment_number = vec![1, 2, 1, 2, 3, 3, 2];
    let fragment_position = vec![1u8; 7];
    let fragment_type = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::A,
    ];

    // When: the short ions are filtered out and the top 3 fragments are kept
    let (mz, _, _, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz_library,
        &fragment_intensity,
        &fragment_cardinality,
        &fragment_charge,
        &fragment_loss_type,
        &fragment_number,
        &fragment_position,
        &fragment_type,
        false,
        false,
        true,
        3,
    );

    // Then: the top 3 come from b3, y3 and a2, as the short b and y ions are dropped before the top-k choice
    assert_eq!(mz, vec![500.0, 600.0, 700.0]);
}

#[test]
fn test_filter_sort_fragments_keeps_short_ions_without_the_filter() {
    // Given: b1, y2 and b3
    let fragment_mz = vec![100.0, 200.0, 300.0];
    let fragment_type = vec![FragmentType::B, FragmentType::Y, FragmentType::B];

    // When: the fragments are filtered without the short ion filter
    let (mz, _, _, _, _, _, _, _, _) = filter_sort_fragments(
        &fragment_mz,
        &fragment_mz,
        &[30.0, 20.0, 10.0],
        &[1u8; 3],
        &[1u8; 3],
        &[Loss::NONE; 3],
        &[1, 2, 3],
        &[1u8; 3],
        &fragment_type,
        false,
        false,
        false,
        usize::MAX,
    );

    // Then: all fragments are kept
    assert_eq!(mz, vec![100.0, 200.0, 300.0]);
}

/// Library with 3 precursors. They are given in the order 30, 10, 20 to test the sorting.
/// Precursor 10 has 2 fragments, precursor 20 has 1 fragment and precursor 30 has 3 fragments.
fn three_precursor_library() -> SpecLibFlat {
    SpecLibFlat::from_vecs(
        vec![30, 10, 20],
        vec![700.0, 300.0, 500.0],
        vec![700.0, 300.0, 500.0],
        vec![1.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.5],
        vec![7, 8, 9],
        vec![3, 0, 2],
        vec![6, 2, 3],
        vec![100.0, 110.0, 200.0, 300.0, 310.0, 320.0],
        vec![100.0, 110.0, 200.0, 300.0, 310.0, 320.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1; 6],
        vec![1, 1, 2, 1, 1, 2],
        vec![0; 6],
        vec![1, 2, 3, 4, 5, 6],
        vec![0, 1, 2, 3, 4, 5],
        vec![
            FragmentType::B,
            FragmentType::Y,
            FragmentType::B,
            FragmentType::Y,
            FragmentType::B,
            FragmentType::Y,
        ],
    )
}

#[test]
fn test_normalize_rt_maps_library_range_to_run_range() {
    // Given a library with normalized RTs 0.0, 0.5 and 1.0
    let mut library = three_precursor_library();

    // When the RTs are scaled to a run from 100 s to 300 s
    library.normalize_rt(100.0, 300.0);

    // Then the minimum and the maximum become the run limits and the search RT is the library RT
    assert_eq!(library.precursor_rt_library, vec![100.0, 200.0, 300.0]);
    assert_eq!(library.precursor_rt, library.precursor_rt_library);
}

#[test]
fn test_normalize_rt_with_one_rt_value_gives_run_stop() {
    // Given a library in which all precursors have the same RT
    let mut library = three_precursor_library();
    library.precursor_rt_library = vec![0.3; 3];

    // When the RTs are scaled
    library.normalize_rt(100.0, 300.0);

    // Then all RTs are the run stop, as np.interp gives for an empty range
    assert_eq!(library.precursor_rt_library, vec![300.0; 3]);
}

#[test]
fn test_filter_precursor_mz_removes_precursors_and_their_fragments() {
    // Given a library with precursor m/z 300, 500 and 700
    let mut library = three_precursor_library();

    // When the m/z range is from 300 to 500, limits included
    library.filter_precursor_mz(300.0, 500.0);

    // Then precursor 30 and its 3 fragments are removed and the fragments have no gaps
    assert_eq!(library.precursor_idx, vec![10, 20]);
    assert_eq!(library.flat_frag_start_idx, vec![0, 2]);
    assert_eq!(library.flat_frag_stop_idx, vec![2, 3]);
    assert_eq!(library.fragment_mz_library, vec![100.0, 110.0, 200.0]);
    assert_eq!(library.fragment_number, vec![1, 2, 3]);
    assert_eq!(library.fragment_mz, library.fragment_mz_library);
}

#[test]
fn test_subset_by_idx_keeps_the_given_precursors() {
    // Given a library with the precursors 10, 20 and 30
    let library = three_precursor_library();

    // When the subset of precursors 30 and 10 is made
    let subset = library.subset_by_idx(&[30, 10]).unwrap();

    // Then the subset has the precursors in ascending order with their fragments
    assert_eq!(subset.precursor_idx, vec![10, 30]);
    assert_eq!(subset.flat_frag_start_idx, vec![0, 2]);
    assert_eq!(subset.flat_frag_stop_idx, vec![2, 5]);
    assert_eq!(
        subset.fragment_mz_library,
        vec![100.0, 110.0, 300.0, 310.0, 320.0]
    );
    let precursor = subset.get_precursor(1);
    assert_eq!(precursor.precursor_idx, 30);
    assert_eq!(precursor.fragment_intensity, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_subset_by_idx_with_unknown_precursor_is_an_error() {
    // Given a library with the precursors 10, 20 and 30
    let library = three_precursor_library();

    // When a subset with precursor 40 is made
    let result = library.subset_by_idx(&[10, 40]);

    // Then the result is an error
    assert_eq!(
        result.err().unwrap(),
        "1 of 2 precursors are not in the spectral library"
    );
}

/// Path in the temporary directory that is unique for this test process and `name`.
fn temp_hdf_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("speclib_flat_{}_{name}.hdf", std::process::id()))
}

/// Writes `values` like alphabase does: chunked, with shuffle and LZF compression.
fn write_column<T: hdf5::H5Type>(group: &hdf5::Group, name: &str, values: &[T]) {
    group
        .new_dataset_builder()
        .chunk(values.len().max(1))
        .shuffle()
        .lzf()
        .with_data(values)
        .create(name)
        .unwrap();
}

/// Writes the library of `three_precursor_library` with the column types that alphabase uses.
fn write_three_precursor_hdf(path: &std::path::Path) {
    let file = hdf5::File::create(path).unwrap();
    let library = file.create_group("library").unwrap();
    let precursors = library.create_group("precursor_df").unwrap();
    write_column(&precursors, "precursor_idx", &[30i64, 10, 20]);
    write_column(&precursors, "mz_library", &[700.0f64, 300.0, 500.0]);
    write_column(&precursors, "rt_library", &[1.0f64, 0.0, 0.5]);
    write_column(&precursors, "nAA", &[7i64, 8, 9]);
    write_column(&precursors, "flat_frag_start_idx", &[3i64, 0, 2]);
    write_column(&precursors, "flat_frag_stop_idx", &[6i64, 2, 3]);
    // A column that the reader does not use
    write_column(&precursors, "decoy", &[0i8, 0, 1]);

    let fragments = library.create_group("fragment_df").unwrap();
    write_column(
        &fragments,
        "mz_library",
        &[100.0f64, 110.0, 200.0, 300.0, 310.0, 320.0],
    );
    write_column(&fragments, "intensity", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    write_column(&fragments, "cardinality", &[1u8; 6]);
    write_column(&fragments, "charge", &[1u8, 1, 2, 1, 1, 2]);
    write_column(&fragments, "loss_type", &[0i16; 6]);
    write_column(&fragments, "number", &[1u32, 2, 3, 4, 5, 6]);
    write_column(&fragments, "position", &[0u32, 1, 2, 3, 4, 5]);
    write_column(
        &fragments,
        "type",
        &[
            FragmentType::B as i8,
            FragmentType::Y as i8,
            FragmentType::B as i8,
            FragmentType::Y as i8,
            FragmentType::B as i8,
            FragmentType::Y as i8,
        ],
    );
}

#[test]
fn test_read_flat_hdf_gives_the_same_library_as_from_vecs() {
    // Given an alphabase-like flat HDF file of the three-precursor library
    let path = temp_hdf_path("same_library");
    write_three_precursor_hdf(&path);

    // When the file is read
    let library = read_flat_hdf(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    // Then the library is the same as the library from vectors
    let expected = three_precursor_library();
    assert_eq!(library.precursor_idx, expected.precursor_idx);
    assert_eq!(library.precursor_mz_library, expected.precursor_mz_library);
    assert_eq!(library.precursor_mz, expected.precursor_mz);
    assert_eq!(library.precursor_rt_library, expected.precursor_rt_library);
    assert_eq!(library.precursor_rt, expected.precursor_rt);
    assert_eq!(library.precursor_naa, expected.precursor_naa);
    assert_eq!(library.flat_frag_start_idx, expected.flat_frag_start_idx);
    assert_eq!(library.flat_frag_stop_idx, expected.flat_frag_stop_idx);
    assert_eq!(library.fragment_mz_library, expected.fragment_mz_library);
    assert_eq!(library.fragment_mz, expected.fragment_mz);
    assert_eq!(library.fragment_intensity, expected.fragment_intensity);
    assert_eq!(library.fragment_cardinality, expected.fragment_cardinality);
    assert_eq!(library.fragment_charge, expected.fragment_charge);
    assert_eq!(library.fragment_loss_type, expected.fragment_loss_type);
    assert_eq!(library.fragment_number, expected.fragment_number);
    assert_eq!(library.fragment_position, expected.fragment_position);
    assert_eq!(library.fragment_type, expected.fragment_type);
}

#[test]
fn test_read_flat_hdf_with_missing_column_is_an_error() {
    // Given a flat HDF file without fragment intensities
    let path = temp_hdf_path("missing_column");
    write_three_precursor_hdf(&path);
    {
        let file = hdf5::File::append(&path).unwrap();
        file.unlink("library/fragment_df/intensity").unwrap();
    }

    // When the file is read
    let result = read_flat_hdf(&path);
    std::fs::remove_file(&path).unwrap();

    // Then the error names the missing column
    let error = result.err().unwrap();
    assert!(error.contains("Column 'intensity' not found"), "{error}");
}

#[test]
fn test_read_flat_hdf_reads_a_library_written_by_alphabase() {
    // Given a flat library of 50 precursors with 12 fragments each, written by alphabase `SpecLibFlat.save_hdf`
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/speclib_flat_alphabase.hdf");

    // When the file is read
    let library = read_flat_hdf(&path).unwrap();

    // Then all precursors and fragments are read with the values that alphabase loads
    assert_eq!(library.num_precursors(), 50);
    assert_eq!(library.num_fragments(), 600);
    assert_eq!(library.precursor_idx, (0..50).collect::<Vec<usize>>());

    let precursor = library.get_precursor(0);
    assert_relative_eq!(precursor.mz_library, 365.870_83, epsilon = 1e-3);
    assert_relative_eq!(precursor.rt_library, 0.147_39, epsilon = 1e-5);
    assert_eq!(precursor.naa, 9);
    assert_eq!(precursor.fragment_mz_library.len(), 12);

    assert_relative_eq!(
        library
            .precursor_mz_library
            .iter()
            .map(|&mz| mz as f64)
            .sum::<f64>(),
        44_003.758,
        epsilon = 1e-2
    );
    assert_relative_eq!(
        library
            .fragment_intensity
            .iter()
            .map(|&i| i as f64)
            .sum::<f64>(),
        536.216_8,
        epsilon = 1e-3
    );
    assert_eq!(library.fragment_mz_library[0], 58.028_74);
    assert_eq!(library.fragment_charge[..2], [1, 2]);
    assert_eq!(library.fragment_type[..3], [98, 98, 121]);
}
