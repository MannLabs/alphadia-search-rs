use super::*;
use crate::mz_index::MZIndex;
use numpy::ndarray::Array1;

#[test]
fn test_optimized_observation_creation() {
    let obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 10, 1000, 5000);

    assert_eq!(obs.isolation_window, [100.0, 150.0]);
    assert_eq!(obs.num_cycles, 10);
    assert_eq!(obs.slice_starts.len(), 1); // Just the initial 0
    assert_eq!(obs.cycle_indices.capacity(), 5000);
    assert_eq!(obs.intensities.capacity(), 5000);
}

#[test]
fn test_slice_data_access() {
    let mut obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 5, 3, 6);

    // Add data for slice 0
    obs.add_peak_data(1, 100.0);
    obs.add_peak_data(2, 200.0);
    obs.finalize_slice();

    // Add data for slice 1
    obs.add_peak_data(3, 300.0);
    obs.finalize_slice();

    // Add data for slice 2 (empty)
    obs.finalize_slice();

    // Test slice 0
    let (cycles, intensities) = obs.get_slice_data(0);
    assert_eq!(cycles, &[1, 2]);
    assert_eq!(intensities, &[100.0, 200.0]);

    // Test slice 1
    let (cycles, intensities) = obs.get_slice_data(1);
    assert_eq!(cycles, &[3]);
    assert_eq!(intensities, &[300.0]);

    // Test slice 2 (empty)
    let (cycles, intensities) = obs.get_slice_data(2);
    assert_eq!(cycles, &[] as &[u16]);
    assert_eq!(intensities, &[] as &[f32]);
}

#[test]
fn test_peak_data_building() {
    let mut obs = QuadrupoleObservation::new_with_capacity([200.0, 250.0], 8, 2, 10);

    // Build first slice with multiple peaks
    obs.add_peak_data(0, 1000.0);
    obs.add_peak_data(1, 1100.0);
    obs.add_peak_data(4, 1400.0);
    obs.finalize_slice();

    // Build second slice with different peaks
    obs.add_peak_data(2, 2200.0);
    obs.add_peak_data(3, 2300.0);
    obs.add_peak_data(5, 2500.0);
    obs.add_peak_data(7, 2700.0);
    obs.finalize_slice();

    // Verify slice_starts structure
    assert_eq!(obs.slice_starts, vec![0, 3, 7]);

    // Verify data integrity
    assert_eq!(obs.cycle_indices, vec![0, 1, 4, 2, 3, 5, 7]);
    assert_eq!(
        obs.intensities,
        vec![1000.0, 1100.0, 1400.0, 2200.0, 2300.0, 2500.0, 2700.0]
    );

    // Test slice access
    let (cycles0, intensities0) = obs.get_slice_data(0);
    assert_eq!(cycles0, &[0, 1, 4]);
    assert_eq!(intensities0, &[1000.0, 1100.0, 1400.0]);

    let (cycles1, intensities1) = obs.get_slice_data(1);
    assert_eq!(cycles1, &[2, 3, 5, 7]);
    assert_eq!(intensities1, &[2200.0, 2300.0, 2500.0, 2700.0]);
}

#[test]
fn test_fill_xic_slice_basic() {
    // Create a mock MZIndex to avoid index out of bounds issues
    let mz_index = MZIndex::new();

    // Create observation with enough slices to match a small subset of mz_index
    let num_slices = 10;
    let mut obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 5, num_slices, 20);

    // Add data for the first few slices only
    for slice_idx in 0..5 {
        obs.add_peak_data(slice_idx as u16, (slice_idx as f32 + 1.0) * 100.0);
        obs.finalize_slice();
    }

    // Fill remaining empty slices
    for _ in 5..num_slices {
        obs.finalize_slice();
    }

    // Create XIC array for 5 cycles
    let mut xic_array = Array1::<f32>::zeros(5);
    let mut xic_view = xic_array.view_mut();

    // Test fill_xic_slice with very narrow tolerance to only hit a few indices
    obs.fill_xic_slice(
        &mz_index,
        &mut xic_view,
        0,              // cycle_start_idx
        5,              // cycle_stop_idx
        0.1,            // very narrow tolerance to minimize range
        mz_index.mz[0], // Use first m/z value
    );

    // Should have some intensity in the first cycle (index 0)
    assert_eq!(xic_array[0], 100.0);
}

#[test]
fn test_fill_xic_slice_cycle_range() {
    let mz_index = MZIndex::new();

    // Create observation with enough slices
    let num_slices = 10;
    let mut obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 10, num_slices, 50);

    // Add data to first slice only, across all cycles
    for cycle in 0..10 {
        obs.add_peak_data(cycle, (cycle as f32) * 100.0);
    }
    obs.finalize_slice();

    // Fill remaining empty slices
    for _ in 1..num_slices {
        obs.finalize_slice();
    }

    // Test partial cycle range [2, 6)
    let mut xic_array = Array1::<f32>::zeros(4); // 4 cycles: 2, 3, 4, 5
    let mut xic_view = xic_array.view_mut();

    obs.fill_xic_slice(
        &mz_index,
        &mut xic_view,
        2,              // cycle_start_idx
        6,              // cycle_stop_idx
        0.1,            // narrow tolerance to hit only first few mz indices
        mz_index.mz[0], // Use first m/z value
    );

    // Should get cycles 2, 3, 4, 5 with intensities 200, 300, 400, 500
    let expected = vec![200.0, 300.0, 400.0, 500.0];
    assert_eq!(xic_array.to_vec(), expected);
}

#[test]
fn test_fill_xic_slice_mass_tolerance() {
    let mut obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 3, 5, 6);

    // Add data to different slices
    for i in 0..5 {
        obs.add_peak_data(1, 100.0 * (i as f32 + 1.0));
        obs.finalize_slice();
    }

    // Create MZIndex using default constructor
    let mz_index = MZIndex::new();

    let mut xic_array = Array1::<f32>::zeros(3);
    let mut xic_view = xic_array.view_mut();

    // Test with narrow tolerance - should only hit specific m/z indices
    obs.fill_xic_slice(
        &mz_index,
        &mut xic_view,
        0,
        3,
        0.1,            // very narrow tolerance (0.1 ppm)
        mz_index.mz[2], // Use actual m/z value from index 2
    );

    // Should only get intensity from slice 2 (300.0)
    assert_eq!(xic_array[1], 300.0);
    assert_eq!(xic_array[0], 0.0);
    assert_eq!(xic_array[2], 0.0);
}

#[test]
fn test_edge_cases() {
    // Test empty observation
    let obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 5, 0, 0);

    assert_eq!(obs.slice_starts.len(), 1);
    assert_eq!(obs.cycle_indices.len(), 0);
    assert_eq!(obs.intensities.len(), 0);

    // Test single slice with no data
    let mut obs2 = QuadrupoleObservation::new_with_capacity([200.0, 250.0], 3, 1, 0);
    obs2.finalize_slice();

    let (cycles, intensities) = obs2.get_slice_data(0);
    assert_eq!(cycles.len(), 0);
    assert_eq!(intensities.len(), 0);
}

#[test]
fn test_isolation_window_validation() {
    let obs1 = QuadrupoleObservation::new_with_capacity([500.0, 600.0], 15, 100, 1000);

    assert_eq!(obs1.isolation_window[0], 500.0);
    assert_eq!(obs1.isolation_window[1], 600.0);
    assert_eq!(obs1.num_cycles, 15);

    // Test different isolation window
    let obs2 = QuadrupoleObservation::new_with_capacity([300.5, 325.8], 8, 50, 200);

    assert_eq!(obs2.isolation_window[0], 300.5);
    assert_eq!(obs2.isolation_window[1], 325.8);
    assert_eq!(obs2.num_cycles, 8);
}

#[test]
fn test_large_data_handling() {
    let mut obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 1000, 100, 10000);

    // Add data to multiple slices with many peaks
    for slice_idx in 0..10 {
        for peak_idx in 0..100 {
            let cycle = (slice_idx * 10 + peak_idx) % 1000;
            let intensity = (slice_idx as f32) * 100.0 + (peak_idx as f32);
            obs.add_peak_data(cycle as u16, intensity);
        }
        obs.finalize_slice();
    }

    // Verify structure
    assert_eq!(obs.slice_starts.len(), 11); // 10 slices + initial 0
    assert_eq!(obs.cycle_indices.len(), 1000);
    assert_eq!(obs.intensities.len(), 1000);

    // Test random slice access
    let (cycles, intensities) = obs.get_slice_data(5);
    assert_eq!(cycles.len(), 100);
    assert_eq!(intensities.len(), 100);

    // Verify first few values for slice 5
    assert_eq!(intensities[0], 500.0); // slice_idx=5 * 100 + peak_idx=0
    assert_eq!(intensities[1], 501.0); // slice_idx=5 * 100 + peak_idx=1
}

#[test]
fn test_memory_footprint() {
    let obs = QuadrupoleObservation::new_with_capacity([100.0, 150.0], 10, 100, 1000);

    let memory_bytes = obs.memory_footprint_bytes();
    assert!(memory_bytes > 0);

    // Should include fixed sizes and empty Vec capacities
    let expected_min_size = std::mem::size_of::<[f32; 2]>()
        + std::mem::size_of::<usize>()
        + 3 * std::mem::size_of::<Vec<u32>>(); // 3 Vec overheads
    assert!(memory_bytes >= expected_min_size);
}
