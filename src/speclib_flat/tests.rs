use super::*;
use crate::constants::{FragmentType, Loss};
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
    let (mz, _mz_library, intensity, _, _, _, _, _, _) =
        filter_sort_fragments(&[], &[], &[], &[], &[], &[], &[], &[], &[], false, false, 5);

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
    use pyo3::{prepare_freethreaded_python, Python};

    prepare_freethreaded_python();
    Python::with_gil(|py| {
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

    pyo3::prepare_freethreaded_python();

    // Create numpy arrays for the test data
    use numpy::PyArray1;

    pyo3::Python::with_gil(|py| {
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

        let precursor = speclib.get_precursor_filtered(0, false, false, usize::MAX);

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
        let precursor_filtered = speclib.get_precursor_filtered(0, true, false, usize::MAX);
        assert!(precursor_filtered
            .fragment_intensity
            .iter()
            .all(|&x| x > 0.0));
        assert_eq!(precursor_filtered.fragment_mz.len(), 5); // All intensities are > 0

        // Test filtering with top_k = 3
        let precursor_top3 = speclib.get_precursor_filtered(0, false, false, 3);
        assert_eq!(precursor_top3.fragment_mz.len(), 3);
        // Should contain top 3 intensities: 30.0, 25.0, 20.0 sorted by mz: 100.0, 200.0, 400.0
        assert_eq!(precursor_top3.fragment_mz, vec![100.0, 200.0, 400.0]);
        assert_eq!(precursor_top3.fragment_intensity, vec![30.0, 25.0, 20.0]);

        // Test combined filtering: non_zero = true and top_k = 2
        let precursor_combined = speclib.get_precursor_filtered(0, true, false, 2);
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
        false,      // Don't filter zero intensities
        true,       // Filter Y1 ions - this is what we're testing!
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
fn test_speclib_flat_idf_creation() {
    use numpy::PyArray1;
    use pyo3::{prepare_freethreaded_python, Python};

    prepare_freethreaded_python();
    Python::with_gil(|py| {
        let precursor_idx = PyArray1::from_slice(py, &[1usize]);
        let precursor_mz = PyArray1::from_slice(py, &[300.0f32]);
        let precursor_rt = PyArray1::from_slice(py, &[30.0f32]);
        let precursor_naa = PyArray1::from_slice(py, &[10u8]);
        let precursor_start_idx = PyArray1::from_slice(py, &[0usize]);
        let precursor_stop_idx = PyArray1::from_slice(py, &[2usize]);

        let fragment_mz_library = PyArray1::from_slice(py, &[100.0f32, 100.0]);
        let fragment_mz = PyArray1::from_slice(py, &[100.0f32, 100.0]);
        let fragment_intensity = PyArray1::from_slice(py, &[10.0f32, 20.0]);
        let fragment_cardinality = PyArray1::from_slice(py, &[1u8; 2]);
        let fragment_charge = PyArray1::from_slice(py, &[1u8; 2]);
        let fragment_loss_type = PyArray1::from_slice(py, &[Loss::NONE; 2]);
        let fragment_number = PyArray1::from_slice(py, &[1u8; 2]);
        let fragment_position = PyArray1::from_slice(py, &[1u8; 2]);
        let fragment_type = PyArray1::from_slice(py, &[FragmentType::B; 2]);

        let speclib = SpecLibFlat::from_arrays(
            precursor_idx.readonly(),
            precursor_mz.readonly(),
            precursor_mz.readonly(),
            precursor_rt.readonly(),
            precursor_rt.readonly(),
            precursor_naa.readonly(),
            precursor_start_idx.readonly(),
            precursor_stop_idx.readonly(),
            fragment_mz_library.readonly(),
            fragment_mz.readonly(),
            fragment_intensity.readonly(),
            fragment_cardinality.readonly(),
            fragment_charge.readonly(),
            fragment_loss_type.readonly(),
            fragment_number.readonly(),
            fragment_position.readonly(),
            fragment_type.readonly(),
        );

        let idf_values = speclib.idf.get_idf(&[100.0, 300.0]);

        assert_eq!(idf_values.len(), 2);

        // 100.0 appears twice in library (df=2), so IDF = ln(2/2) = ln(1) = 0
        assert!((idf_values[0] - 1.0_f32.ln()).abs() < 1e-6);

        // 300.0 is not in library (df=1 due to max(1)), so IDF = ln(2/1) = ln(2)
        assert!((idf_values[1] - 2.0_f32.ln()).abs() < 1e-6);

        assert_eq!(speclib.idf.total_fragments, 2.0);
    });
}

#[test]
fn test_speclib_flat_empty_idf() {
    // Test that empty SpecLibFlat has empty IDF that returns 1.0 values
    let empty_speclib = SpecLibFlat::new();

    let query_mz = vec![100.0, 200.0, 300.0];
    let idf_values = empty_speclib.idf.get_idf(&query_mz);

    // Empty library should return all 1.0 values
    assert_eq!(idf_values, vec![1.0, 1.0, 1.0]);
    assert_eq!(empty_speclib.idf.total_fragments, 0.0);
}
