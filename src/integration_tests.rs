//! Integration tests comparing old and new implementations for equivalence.

#[cfg(test)]
mod tests {
    use crate::dia_data::AlphaRawView;
    use crate::dia_data_builder::DIADataBuilder;
    use crate::dia_data_builder_next_gen::OptimizedDIADataBuilder;
    use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
    use numpy::ndarray::{Array1, ArrayView1};

    fn create_mock_alpha_raw_view<'a>(
        spectrum_delta_scan_idx: &'a [i64],
        isolation_lower_mz: &'a [f32],
        isolation_upper_mz: &'a [f32],
        spectrum_peak_start_idx: &'a [i64],
        spectrum_peak_stop_idx: &'a [i64],
        spectrum_cycle_idx: &'a [i64],
        spectrum_rt: &'a [f32],
        peak_mz: &'a [f32],
        peak_intensity: &'a [f32],
    ) -> AlphaRawView<'a> {
        AlphaRawView {
            spectrum_delta_scan_idx: ArrayView1::from(spectrum_delta_scan_idx),
            isolation_lower_mz: ArrayView1::from(isolation_lower_mz),
            isolation_upper_mz: ArrayView1::from(isolation_upper_mz),
            spectrum_peak_start_idx: ArrayView1::from(spectrum_peak_start_idx),
            spectrum_peak_stop_idx: ArrayView1::from(spectrum_peak_stop_idx),
            spectrum_cycle_idx: ArrayView1::from(spectrum_cycle_idx),
            spectrum_rt: ArrayView1::from(spectrum_rt),
            peak_mz: ArrayView1::from(peak_mz),
            peak_intensity: ArrayView1::from(peak_intensity),
        }
    }

    /// Generic helper function to test XIC slice filling for any DIADataTrait implementation
    fn test_xic_slice_generic<T: DIADataTrait>(
        dia_data: &T,
        obs_idx: usize,
        cycle_start: usize,
        cycle_stop: usize,
        tolerance: f32,
        test_mz: f32,
    ) -> Array1<f32> {
        let xic_size = cycle_stop - cycle_start;
        let mut xic = Array1::<f32>::zeros(xic_size);

        dia_data.quadrupole_observations()[obs_idx].fill_xic_slice(
            dia_data.mz_index(),
            &mut xic.view_mut(),
            cycle_start,
            cycle_stop,
            tolerance,
            test_mz,
        );

        xic
    }

    /// Compare XIC results between two DIADataTrait implementations
    fn assert_xic_equivalence<T1: DIADataTrait, T2: DIADataTrait>(
        dia_data1: &T1,
        dia_data2: &T2,
        obs_idx: usize,
        cycle_start: usize,
        cycle_stop: usize,
        tolerance: f32,
        test_mz: f32,
        context: &str,
    ) {
        let xic1 = test_xic_slice_generic(
            dia_data1,
            obs_idx,
            cycle_start,
            cycle_stop,
            tolerance,
            test_mz,
        );
        let xic2 = test_xic_slice_generic(
            dia_data2,
            obs_idx,
            cycle_start,
            cycle_stop,
            tolerance,
            test_mz,
        );

        let xic_size = cycle_stop - cycle_start;
        for i in 0..xic_size {
            assert!(
                (xic1[i] - xic2[i]).abs() < 1e-6,
                "XIC mismatch at {}, obs={}, mz={}, tolerance={}, cycles=[{},{}), idx={}: impl1={}, impl2={}",
                context, obs_idx, test_mz, tolerance, cycle_start, cycle_stop, i, xic1[i], xic2[i]
            );
        }
    }

    /// Generic test runner that compares two implementations across multiple test scenarios
    fn run_equivalence_test<F>(alpha_raw_view: &AlphaRawView, test_scenarios: F)
    where
        F: Fn(&crate::dia_data::DIAData, &crate::dia_data_next_gen::DIADataNextGen),
    {
        let old_dia_data = DIADataBuilder::from_alpha_raw(alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(alpha_raw_view);

        // Verify basic structure equivalence
        assert_eq!(
            old_dia_data.num_observations(),
            new_dia_data.num_observations()
        );

        test_scenarios(&old_dia_data, &new_dia_data);
    }

    /// Test multiple XIC scenarios for a given observation
    fn test_observation_scenarios<T1: DIADataTrait, T2: DIADataTrait>(
        dia_data1: &T1,
        dia_data2: &T2,
        obs_idx: usize,
        test_cases: &[(f32, f32)],       // (test_mz, tolerance) pairs
        cycle_ranges: &[(usize, usize)], // (cycle_start, cycle_stop) pairs
        context: &str,
    ) {
        for &(test_mz, tolerance) in test_cases {
            for &(cycle_start, cycle_stop) in cycle_ranges {
                assert_xic_equivalence(
                    dia_data1,
                    dia_data2,
                    obs_idx,
                    cycle_start,
                    cycle_stop,
                    tolerance,
                    test_mz,
                    context,
                );
            }
        }
    }

    #[test]
    fn test_fill_xic_slice_equivalence_basic() {
        // Create test data that will exercise both implementations thoroughly
        let spectrum_delta_scan_idx = [0i64, 0, 0, 1, 1, 1, 2, 2, 2];
        let isolation_lower_mz = [
            100.0f32, 100.0, 100.0, 200.0, 200.0, 200.0, 300.0, 300.0, 300.0,
        ];
        let isolation_upper_mz = [
            150.0f32, 150.0, 150.0, 250.0, 250.0, 250.0, 350.0, 350.0, 350.0,
        ];
        let spectrum_peak_start_idx = [0i64, 3, 6, 9, 12, 15, 18, 21, 24];
        let spectrum_peak_stop_idx = [3i64, 6, 9, 12, 15, 18, 21, 24, 27];
        let spectrum_cycle_idx = [0i64, 1, 2, 0, 1, 2, 0, 1, 2];
        let spectrum_rt = [1.0f32, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2];

        // Create peaks that will map to overlapping m/z regions for testing
        let peak_mz = [
            // Observation 0 (cycles 0, 1, 2)
            120.0f32, 125.0, 130.0, // cycle 0
            121.0, 126.0, 131.0, // cycle 1
            122.0, 127.0, 132.0, // cycle 2
            // Observation 1 (cycles 0, 1, 2)
            220.0, 225.0, 230.0, // cycle 0
            221.0, 226.0, 231.0, // cycle 1
            222.0, 227.0, 232.0, // cycle 2
            // Observation 2 (cycles 0, 1, 2)
            320.0, 325.0, 330.0, // cycle 0
            321.0, 326.0, 331.0, // cycle 1
            322.0, 327.0, 332.0, // cycle 2
        ];

        let peak_intensity = [
            // Observation 0
            1000.0f32, 1100.0, 1200.0, // cycle 0
            2000.0, 2100.0, 2200.0, // cycle 1
            3000.0, 3100.0, 3200.0, // cycle 2
            // Observation 1
            4000.0, 4100.0, 4200.0, // cycle 0
            5000.0, 5100.0, 5200.0, // cycle 1
            6000.0, 6100.0, 6200.0, // cycle 2
            // Observation 2
            7000.0, 7100.0, 7200.0, // cycle 0
            8000.0, 8100.0, 8200.0, // cycle 1
            9000.0, 9100.0, 9200.0, // cycle 2
        ];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        // Define test scenarios
        let test_cases = [
            (125.0f32, 1000.0f32), // Wide tolerance, should hit multiple peaks
            (225.0f32, 100.0f32),  // Medium tolerance
            (325.0f32, 10.0f32),   // Narrow tolerance
            (400.0f32, 1000.0f32), // m/z outside isolation windows
        ];

        let cycle_ranges = [
            (0, 3), // Full range
            (0, 1), // Partial range from start
            (1, 3), // Partial range to end
            (1, 2), // Single cycle
        ];

        // Use generic test runner
        run_equivalence_test(&alpha_raw_view, |old_dia_data, new_dia_data| {
            // Test fill_xic_slice equivalence for each observation
            for obs_idx in 0..old_dia_data.num_observations() {
                test_observation_scenarios(
                    old_dia_data,
                    new_dia_data,
                    obs_idx,
                    &test_cases,
                    &cycle_ranges,
                    "basic_test",
                );
            }
        });
    }

    #[test]
    fn test_fill_xic_slice_equivalence_edge_cases() {
        // Test with non-sequential cycles and complex peak distributions
        let spectrum_delta_scan_idx = [0i64, 0, 0, 0, 0];
        let isolation_lower_mz = [500.0f32, 500.0, 500.0, 500.0, 500.0];
        let isolation_upper_mz = [600.0f32, 600.0, 600.0, 600.0, 600.0];
        let spectrum_peak_start_idx = [0i64, 2, 4, 6, 8];
        let spectrum_peak_stop_idx = [2i64, 4, 6, 8, 10];
        let spectrum_cycle_idx = [5i64, 1, 3, 0, 2]; // Non-sequential cycles
        let spectrum_rt = [1.0f32, 1.1, 1.2, 1.3, 1.4];

        // Peaks that will all map to similar m/z indices (testing cycle ordering)
        let peak_mz = [
            550.0f32, 550.1, // cycle 5
            550.2, 550.3, // cycle 1
            550.4, 550.5, // cycle 3
            550.6, 550.7, // cycle 0
            550.8, 550.9, // cycle 2
        ];

        let peak_intensity = [
            1000.0f32, 1001.0, // cycle 5
            1002.0, 1003.0, // cycle 1
            1004.0, 1005.0, // cycle 3
            1006.0, 1007.0, // cycle 0
            1008.0, 1009.0, // cycle 2
        ];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        // Build both implementations
        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test with wide tolerance to capture all peaks
        let mut old_xic = Array1::<f32>::zeros(6); // cycles 0-5
        let mut new_xic = Array1::<f32>::zeros(6);

        old_dia_data.quadrupole_observations[0].fill_xic_slice(
            &old_dia_data.mz_index,
            &mut old_xic.view_mut(),
            0,
            6,
            1000.0, // Very wide tolerance
            550.5,  // Center m/z
        );

        new_dia_data.quadrupole_observations[0].fill_xic_slice(
            &new_dia_data.mz_index,
            &mut new_xic.view_mut(),
            0,
            6,
            1000.0,
            550.5,
        );

        // Verify cycle ordering is preserved correctly
        for i in 0..6 {
            assert!(
                (old_xic[i] - new_xic[i]).abs() < 1e-6,
                "XIC mismatch for non-sequential cycles at cycle {}: old={}, new={}",
                i,
                old_xic[i],
                new_xic[i]
            );
        }

        // Verify that cycles have expected intensities based on ordering
        // Cycle 0 should have intensities 1006.0 + 1007.0 = 2013.0
        let expected_cycle_0 = 1006.0 + 1007.0;
        assert!((old_xic[0] - expected_cycle_0).abs() < 1e-6);
        assert!((new_xic[0] - expected_cycle_0).abs() < 1e-6);
    }

    #[test]
    fn test_fill_xic_slice_equivalence_empty_observations() {
        // Test with observations that have no peaks
        let spectrum_delta_scan_idx = [0i64, 2]; // Skip delta_scan_idx = 1
        let isolation_lower_mz = [100.0f32, 300.0];
        let isolation_upper_mz = [150.0f32, 350.0];
        let spectrum_peak_start_idx = [0i64, 2];
        let spectrum_peak_stop_idx = [2i64, 4];
        let spectrum_cycle_idx = [0i64, 0];
        let spectrum_rt = [1.0f32, 3.0];

        let peak_mz = [120.0f32, 125.0, 320.0, 325.0];
        let peak_intensity = [1000.0f32, 1100.0, 3000.0, 3100.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        // Build both implementations
        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Should have 3 observations (0, 1, 2) but observation 1 will be empty
        assert_eq!(
            old_dia_data.num_observations(),
            new_dia_data.num_observations()
        );

        // Test fill_xic_slice on empty observation (index 1)
        let mut old_xic = Array1::<f32>::zeros(1);
        let mut new_xic = Array1::<f32>::zeros(1);

        old_dia_data.quadrupole_observations[1].fill_xic_slice(
            &old_dia_data.mz_index,
            &mut old_xic.view_mut(),
            0,
            1,
            1000.0,
            200.0, // Any m/z value
        );

        new_dia_data.quadrupole_observations[1].fill_xic_slice(
            &new_dia_data.mz_index,
            &mut new_xic.view_mut(),
            0,
            1,
            1000.0,
            200.0,
        );

        // Both should return zeros for empty observation
        assert_eq!(old_xic[0], 0.0);
        assert_eq!(new_xic[0], 0.0);
        assert_eq!(old_xic[0], new_xic[0]);
    }

    #[test]
    fn test_fill_xic_slice_equivalence_precision() {
        // Test with high precision requirements and overlapping peaks
        let spectrum_delta_scan_idx = [0i64, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0, 200.0];
        let spectrum_peak_start_idx = [0i64, 3, 6];
        let spectrum_peak_stop_idx = [3i64, 6, 9];
        let spectrum_cycle_idx = [0i64, 1, 2];
        let spectrum_rt = [1.0f32, 1.1, 1.2];

        // Create peaks with very close m/z values to test precision
        let peak_mz = [
            150.000000f32,
            150.000001,
            150.000002, // cycle 0
            150.000003,
            150.000004,
            150.000005, // cycle 1
            150.000006,
            150.000007,
            150.000008, // cycle 2
        ];

        let peak_intensity = [
            1.0f32, 2.0, 3.0, // cycle 0
            4.0, 5.0, 6.0, // cycle 1
            7.0, 8.0, 9.0, // cycle 2
        ];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        // Build both implementations
        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test with various precision tolerances
        let tolerances = [0.1f32, 1.0, 10.0, 100.0, 1000.0];

        for tolerance in tolerances {
            let mut old_xic = Array1::<f32>::zeros(3);
            let mut new_xic = Array1::<f32>::zeros(3);

            old_dia_data.quadrupole_observations[0].fill_xic_slice(
                &old_dia_data.mz_index,
                &mut old_xic.view_mut(),
                0,
                3,
                tolerance,
                150.000004, // Center m/z
            );

            new_dia_data.quadrupole_observations[0].fill_xic_slice(
                &new_dia_data.mz_index,
                &mut new_xic.view_mut(),
                0,
                3,
                tolerance,
                150.000004,
            );

            // Verify precise equivalence
            for i in 0..3 {
                assert!(
                    (old_xic[i] - new_xic[i]).abs() < 1e-10,
                    "High precision XIC mismatch at tolerance={}, cycle={}: old={}, new={}",
                    tolerance,
                    i,
                    old_xic[i],
                    new_xic[i]
                );
            }
        }
    }

    #[test]
    fn test_equivalence_comprehensive_validation() {
        // Large comprehensive test with complex data patterns
        let num_obs = 5;
        let cycles_per_obs = 4;
        let peaks_per_spectrum = 10;

        let mut spectrum_delta_scan_idx = Vec::new();
        let mut isolation_lower_mz = Vec::new();
        let mut isolation_upper_mz = Vec::new();
        let mut spectrum_peak_start_idx = Vec::new();
        let mut spectrum_peak_stop_idx = Vec::new();
        let mut spectrum_cycle_idx = Vec::new();
        let mut spectrum_rt = Vec::new();
        let mut peak_mz = Vec::new();
        let mut peak_intensity = Vec::new();

        let mut peak_idx = 0;

        for obs in 0..num_obs {
            for cycle in 0..cycles_per_obs {
                spectrum_delta_scan_idx.push(obs as i64);
                isolation_lower_mz.push(100.0 + obs as f32 * 100.0);
                isolation_upper_mz.push(150.0 + obs as f32 * 100.0);
                spectrum_peak_start_idx.push(peak_idx);
                spectrum_peak_stop_idx.push(peak_idx + peaks_per_spectrum);
                spectrum_cycle_idx.push(cycle as i64);
                spectrum_rt.push(1.0 + cycle as f32 * 0.1);

                // Add peaks for this spectrum
                for peak in 0..peaks_per_spectrum {
                    let base_mz = 125.0 + obs as f32 * 100.0;
                    peak_mz.push(base_mz + peak as f32 * 0.1);
                    peak_intensity
                        .push(1000.0 + (obs * cycles_per_obs + cycle) as f32 * 100.0 + peak as f32);
                }

                peak_idx += peaks_per_spectrum;
            }
        }

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        // Build both implementations
        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Comprehensive validation
        assert_eq!(
            old_dia_data.num_observations(),
            new_dia_data.num_observations()
        );
        assert_eq!(old_dia_data.num_observations(), num_obs);

        // Test multiple scenarios per observation
        for obs_idx in 0..num_obs {
            let base_mz = 125.0 + obs_idx as f32 * 100.0;

            // Test various m/z offsets and tolerances
            for mz_offset in [-10.0f32, -1.0, 0.0, 1.0, 10.0] {
                for tolerance in [1.0f32, 10.0, 100.0, 1000.0] {
                    let mut old_xic = Array1::<f32>::zeros(cycles_per_obs);
                    let mut new_xic = Array1::<f32>::zeros(cycles_per_obs);

                    let test_mz = base_mz + mz_offset;

                    old_dia_data.quadrupole_observations[obs_idx].fill_xic_slice(
                        &old_dia_data.mz_index,
                        &mut old_xic.view_mut(),
                        0,
                        cycles_per_obs,
                        tolerance,
                        test_mz,
                    );

                    new_dia_data.quadrupole_observations[obs_idx].fill_xic_slice(
                        &new_dia_data.mz_index,
                        &mut new_xic.view_mut(),
                        0,
                        cycles_per_obs,
                        tolerance,
                        test_mz,
                    );

                    for cycle in 0..cycles_per_obs {
                        assert!((old_xic[cycle] - new_xic[cycle]).abs() < 1e-6,
                            "Comprehensive test mismatch: obs={}, mz_offset={}, tolerance={}, cycle={}: old={}, new={}", 
                            obs_idx, mz_offset, tolerance, cycle, old_xic[cycle], new_xic[cycle]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_binary_search_edge_cases() {
        // Test exact boundary matches and duplicate cycles
        let spectrum_delta_scan_idx = [0i64, 0, 0, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0, 100.0, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0, 200.0, 200.0, 200.0];
        let spectrum_peak_start_idx = [0i64, 2, 4, 6, 8];
        let spectrum_peak_stop_idx = [2i64, 4, 6, 8, 10];
        // Test duplicate cycle indices and exact boundary values
        let spectrum_cycle_idx = [5i64, 5, 10, 10, 15]; // Duplicates
        let spectrum_rt = [1.0f32, 1.1, 1.2, 1.3, 1.4];

        // All peaks map to same mz_idx to test binary search within slice
        // Need 10 peaks total: 0-1, 2-3, 4-5, 6-7, 8-9
        let peak_mz = [
            150.0f32, 150.0, // spectrum 0: peaks 0-1, cycle 5
            150.0, 150.0, // spectrum 1: peaks 2-3, cycle 5
            150.0, 150.0, // spectrum 2: peaks 4-5, cycle 10
            150.0, 150.0, // spectrum 3: peaks 6-7, cycle 10
            150.0, 150.0, // spectrum 4: peaks 8-9, cycle 15
        ];

        let peak_intensity = [
            100.0f32, 200.0, // spectrum 0: cycle 5
            300.0, 400.0, // spectrum 1: cycle 5
            500.0, 600.0, // spectrum 2: cycle 10
            700.0, 800.0, // spectrum 3: cycle 10
            900.0, 1000.0, // spectrum 4: cycle 15
        ];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test exact boundary matches
        let test_cases = [
            (5, 16, 150.0),  // Start exactly at cycle 5
            (10, 16, 150.0), // Start exactly at cycle 10
            (0, 6, 150.0),   // Stop after cycle 5
            (0, 11, 150.0),  // Stop after cycle 10
            (5, 11, 150.0),  // Exact range 5-10
            (10, 16, 150.0), // Range 10-15
        ];

        for (cycle_start, cycle_stop, test_mz) in test_cases {
            let xic_size = cycle_stop - cycle_start;
            let mut old_xic = Array1::<f32>::zeros(xic_size);
            let mut new_xic = Array1::<f32>::zeros(xic_size);

            old_dia_data.quadrupole_observations[0].fill_xic_slice(
                &old_dia_data.mz_index,
                &mut old_xic.view_mut(),
                cycle_start,
                cycle_stop,
                1000.0, // Wide tolerance
                test_mz,
            );

            new_dia_data.quadrupole_observations[0].fill_xic_slice(
                &new_dia_data.mz_index,
                &mut new_xic.view_mut(),
                cycle_start,
                cycle_stop,
                1000.0,
                test_mz,
            );

            for i in 0..xic_size {
                assert!(
                    (old_xic[i] - new_xic[i]).abs() < 1e-6,
                    "Binary search edge case mismatch: cycles=[{},{}), idx={}: old={}, new={}",
                    cycle_start,
                    cycle_stop,
                    i,
                    old_xic[i],
                    new_xic[i]
                );
            }
        }
    }

    #[test]
    fn test_extreme_data_type_boundaries() {
        // Test with very large cycle indices near u16::MAX
        let spectrum_delta_scan_idx = [0i64, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0, 200.0];
        let spectrum_peak_start_idx = [0i64, 2, 4];
        let spectrum_peak_stop_idx = [2i64, 4, 6];
        // Test extreme cycle values
        let spectrum_cycle_idx = [0i64, 32767, 65535]; // 0, u16::MAX/2, u16::MAX
        let spectrum_rt = [1.0f32, 1.1, 1.2];

        let peak_mz = [150.0f32, 150.0, 150.0, 150.0, 150.0, 150.0];
        let peak_intensity = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test with large cycle ranges
        let mut old_xic = Array1::<f32>::zeros(65536);
        let mut new_xic = Array1::<f32>::zeros(65536);

        old_dia_data.quadrupole_observations[0].fill_xic_slice(
            &old_dia_data.mz_index,
            &mut old_xic.view_mut(),
            0,
            65536,
            1000.0,
            150.0,
        );

        new_dia_data.quadrupole_observations[0].fill_xic_slice(
            &new_dia_data.mz_index,
            &mut new_xic.view_mut(),
            0,
            65536,
            1000.0,
            150.0,
        );

        // Check specific indices that should have data
        assert!((old_xic[0] - new_xic[0]).abs() < 1e-6);
        assert!((old_xic[32767] - new_xic[32767]).abs() < 1e-6);
        assert!((old_xic[65535] - new_xic[65535]).abs() < 1e-6);

        // Verify non-zero intensities at expected positions
        assert!(old_xic[0] > 0.0);
        assert!(old_xic[32767] > 0.0);
        assert!(old_xic[65535] > 0.0);
        assert_eq!(old_xic[0], new_xic[0]);
        assert_eq!(old_xic[32767], new_xic[32767]);
        assert_eq!(old_xic[65535], new_xic[65535]);
    }

    #[test]
    fn test_sparse_slice_distribution() {
        // Test with most slices empty, but peaks at extreme mz_idx positions
        let spectrum_delta_scan_idx = [0i64, 0, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0, 100.0];
        let isolation_upper_mz = [2000.0f32, 2000.0, 2000.0, 2000.0];
        let spectrum_peak_start_idx = [0i64, 1, 2, 3];
        let spectrum_peak_stop_idx = [1i64, 2, 3, 4];
        let spectrum_cycle_idx = [0i64, 1, 2, 3];
        let spectrum_rt = [1.0f32, 1.1, 1.2, 1.3];

        // Peaks at very different m/z values to hit different slice boundaries
        let peak_mz = [
            151.0f32, // Very beginning of mz_index range
            1000.0,   // Middle of range
            1999.0,   // Very end of range
            500.0,    // Another middle position
        ];

        let peak_intensity = [1000.0f32, 2000.0, 3000.0, 4000.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test each extreme m/z position with tight tolerance
        for &test_mz in &peak_mz {
            let mut old_xic = Array1::<f32>::zeros(4);
            let mut new_xic = Array1::<f32>::zeros(4);

            old_dia_data.quadrupole_observations[0].fill_xic_slice(
                &old_dia_data.mz_index,
                &mut old_xic.view_mut(),
                0,
                4,
                1.0, // 1 ppm tolerance - should be precise
                test_mz,
            );

            new_dia_data.quadrupole_observations[0].fill_xic_slice(
                &new_dia_data.mz_index,
                &mut new_xic.view_mut(),
                0,
                4,
                1.0,
                test_mz,
            );

            for i in 0..4 {
                assert!(
                    (old_xic[i] - new_xic[i]).abs() < 1e-6,
                    "Sparse slice test mismatch at mz={}, cycle={}: old={}, new={}",
                    test_mz,
                    i,
                    old_xic[i],
                    new_xic[i]
                );
            }
        }
    }

    #[test]
    fn test_extreme_tolerance_values() {
        // Test with very small and very large tolerance values
        let spectrum_delta_scan_idx = [0i64, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0, 200.0];
        let spectrum_peak_start_idx = [0i64, 2, 4];
        let spectrum_peak_stop_idx = [2i64, 4, 6];
        let spectrum_cycle_idx = [0i64, 1, 2];
        let spectrum_rt = [1.0f32, 1.1, 1.2];

        // Peaks with specific m/z values
        let peak_mz = [
            150.0f32, 150.00001, // Very close peaks
            150.001, 150.002, // Slightly further
            151.0, 152.0, // Further apart
        ];

        let peak_intensity = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test extreme tolerance values
        let extreme_tolerances = [
            0.0001f32, // Nearly zero tolerance
            0.001,     // Very small tolerance
            1000000.0, // Massive tolerance (should include everything)
        ];

        for tolerance in extreme_tolerances {
            let mut old_xic = Array1::<f32>::zeros(3);
            let mut new_xic = Array1::<f32>::zeros(3);

            old_dia_data.quadrupole_observations[0].fill_xic_slice(
                &old_dia_data.mz_index,
                &mut old_xic.view_mut(),
                0,
                3,
                tolerance,
                150.0,
            );

            new_dia_data.quadrupole_observations[0].fill_xic_slice(
                &new_dia_data.mz_index,
                &mut new_xic.view_mut(),
                0,
                3,
                tolerance,
                150.0,
            );

            for i in 0..3 {
                assert!(
                    (old_xic[i] - new_xic[i]).abs() < 1e-6,
                    "Extreme tolerance test mismatch at tolerance={}, cycle={}: old={}, new={}",
                    tolerance,
                    i,
                    old_xic[i],
                    new_xic[i]
                );
            }
        }
    }

    #[test]
    fn test_empty_cycle_ranges() {
        // Test with empty and single-point cycle ranges
        let spectrum_delta_scan_idx = [0i64, 0, 0];
        let isolation_lower_mz = [100.0f32, 100.0, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0, 200.0];
        let spectrum_peak_start_idx = [0i64, 2, 4];
        let spectrum_peak_stop_idx = [2i64, 4, 6];
        let spectrum_cycle_idx = [5i64, 10, 15];
        let spectrum_rt = [1.0f32, 1.1, 1.2];

        let peak_mz = [150.0f32, 150.0, 150.0, 150.0, 150.0, 150.0];
        let peak_intensity = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Test edge cases for cycle ranges
        let edge_cases = [
            (5, 5),   // Empty range (start == stop)
            (7, 8),   // Single point range with no data
            (5, 6),   // Single point range with data
            (10, 11), // Single point range with data
            (0, 1),   // Range before any data
            (20, 21), // Range after all data
        ];

        for (cycle_start, cycle_stop) in edge_cases {
            if cycle_start >= cycle_stop {
                continue; // Skip invalid ranges
            }

            let xic_size = cycle_stop - cycle_start;
            let mut old_xic = Array1::<f32>::zeros(xic_size);
            let mut new_xic = Array1::<f32>::zeros(xic_size);

            old_dia_data.quadrupole_observations[0].fill_xic_slice(
                &old_dia_data.mz_index,
                &mut old_xic.view_mut(),
                cycle_start,
                cycle_stop,
                1000.0,
                150.0,
            );

            new_dia_data.quadrupole_observations[0].fill_xic_slice(
                &new_dia_data.mz_index,
                &mut new_xic.view_mut(),
                cycle_start,
                cycle_stop,
                1000.0,
                150.0,
            );

            for i in 0..xic_size {
                assert!(
                    (old_xic[i] - new_xic[i]).abs() < 1e-6,
                    "Empty range test mismatch: cycles=[{},{}), idx={}: old={}, new={}",
                    cycle_start,
                    cycle_stop,
                    i,
                    old_xic[i],
                    new_xic[i]
                );
            }
        }
    }

    #[test]
    fn test_single_observation_edge_case() {
        // Test with exactly one observation to stress rayon parallelization edge case
        let spectrum_delta_scan_idx = [0i64, 0];
        let isolation_lower_mz = [100.0f32, 100.0];
        let isolation_upper_mz = [200.0f32, 200.0];
        let spectrum_peak_start_idx = [0i64, 1];
        let spectrum_peak_stop_idx = [1i64, 2];
        let spectrum_cycle_idx = [0i64, 1];
        let spectrum_rt = [1.0f32, 1.1];

        let peak_mz = [150.0f32, 150.0];
        let peak_intensity = [1000.0f32, 2000.0];

        let alpha_raw_view = create_mock_alpha_raw_view(
            &spectrum_delta_scan_idx,
            &isolation_lower_mz,
            &isolation_upper_mz,
            &spectrum_peak_start_idx,
            &spectrum_peak_stop_idx,
            &spectrum_cycle_idx,
            &spectrum_rt,
            &peak_mz,
            &peak_intensity,
        );

        let old_dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        let new_dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);

        // Should have exactly 1 observation
        assert_eq!(old_dia_data.num_observations(), 1);
        assert_eq!(new_dia_data.num_observations(), 1);

        let mut old_xic = Array1::<f32>::zeros(2);
        let mut new_xic = Array1::<f32>::zeros(2);

        old_dia_data.quadrupole_observations[0].fill_xic_slice(
            &old_dia_data.mz_index,
            &mut old_xic.view_mut(),
            0,
            2,
            1000.0,
            150.0,
        );

        new_dia_data.quadrupole_observations[0].fill_xic_slice(
            &new_dia_data.mz_index,
            &mut new_xic.view_mut(),
            0,
            2,
            1000.0,
            150.0,
        );

        assert_eq!(old_xic[0], new_xic[0]);
        assert_eq!(old_xic[1], new_xic[1]);
        assert!(old_xic[0] > 0.0);
        assert!(old_xic[1] > 0.0);
    }
}
