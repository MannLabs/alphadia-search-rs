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
        filter_sort_fragments(&[], &[], &[], &[], &[], &[], &[], &[], &[], false, 5);

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
        let precursor_start_idx = PyArray1::from_slice(py, &[6usize, 0, 9, 3]);
        let precursor_stop_idx = PyArray1::from_slice(py, &[9usize, 3, 12, 6]);
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
            precursor_mz.readonly(), // observed
            precursor_rt.readonly(), // library
            precursor_rt.readonly(), // observed
            precursor_naa.readonly(),
            precursor_start_idx.readonly(),
            precursor_stop_idx.readonly(),
            fragment_mz.readonly(), // library
            fragment_mz.readonly(), // observed
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

        assert_eq!(precursor_1.idx, 1);
        assert_eq!(precursor_2.idx, 2);
        assert_eq!(precursor_3.idx, 3);
        assert_eq!(precursor_4.idx, 4);

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
fn test_speclib_flat_binary_search() {
    use numpy::PyArray1;
    use pyo3::{prepare_freethreaded_python, Python};

    prepare_freethreaded_python();
    Python::with_gil(|py| {
        // Create sorted test data
        let precursor_idx = PyArray1::from_slice(py, &[10usize, 20, 30]);
        let precursor_mz = PyArray1::from_slice(py, &[100.0f32, 200.0, 300.0]);
        let precursor_rt = PyArray1::from_slice(py, &[10.0f32, 20.0, 30.0]);
        let precursor_naa = PyArray1::from_slice(py, &[8u8, 12, 16]);
        let precursor_start_idx = PyArray1::from_slice(py, &[0usize, 2, 4]);
        let precursor_stop_idx = PyArray1::from_slice(py, &[2usize, 4, 6]);
        let fragment_mz = PyArray1::from_slice(py, &[101.0f32, 102.0, 201.0, 202.0, 301.0, 302.0]);
        let fragment_intensity = PyArray1::from_slice(py, &[10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let fragment_cardinality = PyArray1::from_slice(py, &[1u8; 6]);
        let fragment_charge = PyArray1::from_slice(py, &[1u8; 6]);
        let fragment_loss_type = PyArray1::from_slice(py, &[Loss::NONE; 6]);
        let fragment_number = PyArray1::from_slice(py, &[1u8; 6]);
        let fragment_position = PyArray1::from_slice(py, &[1u8; 6]);
        let fragment_type = PyArray1::from_slice(py, &[FragmentType::B; 6]);

        let speclib = SpecLibFlat::from_arrays(
            precursor_idx.readonly(),
            precursor_mz.readonly(), // library
            precursor_mz.readonly(), // observed
            precursor_rt.readonly(), // library
            precursor_rt.readonly(), // observed
            precursor_naa.readonly(),
            precursor_start_idx.readonly(),
            precursor_stop_idx.readonly(),
            fragment_mz.readonly(), // library
            fragment_mz.readonly(), // observed
            fragment_intensity.readonly(),
            fragment_cardinality.readonly(),
            fragment_charge.readonly(),
            fragment_loss_type.readonly(),
            fragment_number.readonly(),
            fragment_position.readonly(),
            fragment_type.readonly(),
        );

        // Test binary search functionality
        assert!(speclib.get_precursor_by_idx(10).is_some());
        assert!(speclib.get_precursor_by_idx(20).is_some());
        assert!(speclib.get_precursor_by_idx(30).is_some());
        assert!(speclib.get_precursor_by_idx(15).is_none());
        assert!(speclib.get_precursor_by_idx(5).is_none());
        assert!(speclib.get_precursor_by_idx(35).is_none());

        // Verify correct precursor is returned
        let precursor_20 = speclib.get_precursor_by_idx(20).unwrap();
        assert_eq!(precursor_20.idx, 20);
        assert_eq!(precursor_20.mz, 200.0);
        assert_eq!(precursor_20.fragment_mz, vec![201.0, 202.0]);
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

        let precursor = speclib.get_precursor_by_idx(0).unwrap();

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
    });
}
