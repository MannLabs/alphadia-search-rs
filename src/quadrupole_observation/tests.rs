use super::*;
use crate::mz_index::MZIndex;
use numpy::ndarray::{Array1, ArrayViewMut1};

#[test]
fn test_quadrupole_observation() {
    // Create a new MZIndex
    let mz_index = MZIndex::new();
    
    // Create a new QuadrupoleObservation
    let mut quad_obs = QuadrupoleObservation::new(&mz_index);
    
    // Test initial state
    assert_eq!(quad_obs.isolation_window, [0.0, 0.0]);
    assert_eq!(quad_obs.xic_slices.len(), mz_index.len());
    
    // Add some test peaks
    let test_mz = 500.0; // Example m/z value
    let test_intensity = 1000.0;
    let test_cycle_idx = 1;
    
    quad_obs.add_peak(test_mz, test_intensity, test_cycle_idx, &mz_index);
    
    // Find the closest index for our test m/z
    let closest_idx = mz_index.find_closest_index(test_mz);
    
    // Verify the peak was added to the correct XIC slice
    assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 1);
    assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 1);
    assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[0], test_cycle_idx);
    assert_eq!(quad_obs.xic_slices[closest_idx].intensity[0], test_intensity);
    
    // Add another peak to the same m/z bin
    quad_obs.add_peak(test_mz, test_intensity * 2.0, test_cycle_idx + 1, &mz_index);
    
    // Verify both peaks are in the same XIC slice
    assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 2);
    assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 2);
    assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[1], test_cycle_idx + 1);
    assert_eq!(quad_obs.xic_slices[closest_idx].intensity[1], test_intensity * 2.0);
}

#[test]
fn test_fill_xic_slice() {
    // Create a new MZIndex
    let mz_index = MZIndex::new();
    
    // Create a new QuadrupoleObservation
    let mut quad_obs = QuadrupoleObservation::new(&mz_index);
    
    // Add some test peaks at different m/z values and cycles
    let test_mz = 500.0;
    let mass_tolerance = 10.0; // 10 ppm tolerance
    
    // Add peaks with different cycle indices
    quad_obs.add_peak(test_mz, 1000.0, 10, &mz_index); // Cycle 10
    quad_obs.add_peak(test_mz, 2000.0, 11, &mz_index); // Cycle 11
    quad_obs.add_peak(test_mz, 3000.0, 12, &mz_index); // Cycle 12
    quad_obs.add_peak(test_mz, 4000.0, 13, &mz_index); // Cycle 13
    quad_obs.add_peak(test_mz, 5000.0, 14, &mz_index); // Cycle 14
    
    // Add a peak with slightly different m/z but within tolerance
    let close_mz = test_mz * (1.0 + 5e-6); // 5 ppm higher
    quad_obs.add_peak(close_mz, 1500.0, 12, &mz_index); // Cycle 12
    
    // Add a peak outside the tolerance range
    let outside_mz = test_mz * (1.0 + 15e-6); // 15 ppm higher
    quad_obs.add_peak(outside_mz, 9000.0, 12, &mz_index); // Cycle 12
    
    // Create a buffer to store the XIC
    let cycle_start = 10;
    let cycle_stop = 15;
    let num_cycles = cycle_stop - cycle_start;
    let mut buffer = Array1::<f32>::zeros(num_cycles);
    let mut dense_xic = buffer.view_mut();
    
    // Fill the XIC slice
    quad_obs.fill_xic_slice(&mz_index, &mut dense_xic, cycle_start, cycle_stop, mass_tolerance, test_mz);
    
    // Check that the intensity values were correctly added
    // Cycle 10 (index 0): 1000.0
    assert_eq!(dense_xic[0], 1000.0);
    // Cycle 11 (index 1): 2000.0
    assert_eq!(dense_xic[1], 2000.0);
    // Cycle 12 (index 2): 3000.0 + 1500.0 (from close_mz) = 4500.0
    assert_eq!(dense_xic[2], 4500.0);
    // Cycle 13 (index 3): 4000.0
    assert_eq!(dense_xic[3], 4000.0);
    // Cycle 14 (index 4): 5000.0
    assert_eq!(dense_xic[4], 5000.0);
    
    // The outside_mz peak should not be included since it's outside tolerance
    
    // Test with a different range
    let mut buffer2 = Array1::<f32>::zeros(2);
    let mut dense_xic2 = buffer2.view_mut();
    
    // Fill the XIC slice for cycles 11-13 only
    quad_obs.fill_xic_slice(&mz_index, &mut dense_xic2, 11, 13, mass_tolerance, test_mz);
    
    // Check correct cycles were included
    assert_eq!(dense_xic2[0], 2000.0);
    assert_eq!(dense_xic2[1], 4500.0);
} 