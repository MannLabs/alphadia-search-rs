use super::*;

fn create_realistic_index() -> RTIndex {
    let rt_values: Vec<f32> = (0..=100).map(|i| i as f32 * 0.5).collect();
    let rt_array = Array1::from_vec(rt_values);
    let delta_t = RTIndex::calculate_mean_delta_t(&rt_array);

    RTIndex {
        rt: rt_array,
        delta_t,
    }
}

#[test]
fn test_minimum_tolerance_enforced() {
    let rt_index = create_realistic_index();
    // Small tolerance should trigger minimum enforcement
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 2.0, 5);

    assert!(lower >= 33 && lower <= 37);
    assert!(upper >= 63 && upper <= 67);
}

#[test]
fn test_large_tolerance_not_enforced() {
    let rt_index = create_realistic_index();
    // Large tolerance should be used as-is
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 10.0, 5);

    assert!(lower >= 28 && lower <= 32);
    assert!(upper >= 68 && upper <= 72);
}

#[test]
fn test_padding_increases_range() {
    let rt_index = create_realistic_index();
    let target_rt = 25.0;
    let small_tolerance = 1.0;

    let (l1, u1) = rt_index.get_cycle_idx_limits(target_rt, small_tolerance, 2);
    let (l2, u2) = rt_index.get_cycle_idx_limits(target_rt, small_tolerance, 10);

    // Higher padding increases minimum tolerance
    assert!(u2 - l2 > u1 - l1);

    let range1 = u1 - l1;
    let range2 = u2 - l2;

    assert!(range1 >= 22 && range1 <= 26);
    assert!(range2 >= 38 && range2 <= 42);
}

#[test]
fn test_precise_boundary_calculation() {
    let rt_index = create_realistic_index();
    // Large tolerance avoids minimum enforcement for precise testing
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 15.0, 0);

    assert_eq!(lower, 20);
    assert_eq!(upper, 80);
}

#[test]
fn test_target_near_boundaries() {
    let rt_index = create_realistic_index();

    // Range gets clipped to data boundaries
    let (lower, upper) = rt_index.get_cycle_idx_limits(5.0, 15.0, 0);
    assert_eq!(lower, 0);
    assert_eq!(upper, 40);

    let (lower, upper) = rt_index.get_cycle_idx_limits(45.0, 15.0, 0);
    assert_eq!(lower, 60);
    assert_eq!(upper, 101);
}

#[test]
fn test_target_outside_range() {
    let rt_index = create_realistic_index();

    let (lower, upper) = rt_index.get_cycle_idx_limits(-20.0, 5.0, 0);
    assert_eq!(lower, 0);
    assert_eq!(upper, 0);

    let (lower, upper) = rt_index.get_cycle_idx_limits(70.0, 5.0, 0);
    assert_eq!(lower, 101);
    assert_eq!(upper, 101);
}

#[test]
fn test_empty_index() {
    let empty_rt_index = RTIndex::new();
    let (lower, upper) = empty_rt_index.get_cycle_idx_limits(25.0, 5.0, 5);
    assert_eq!(lower, 0);
    assert_eq!(upper, 0);
}
