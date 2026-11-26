use super::*;

#[test]
fn test_without_overlap_every_window_is_one_group() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [425.0, 450.0]]);

    // when / then
    assert_eq!(groups.quadrupole_observations_of(410.0), &[0]);
    assert_eq!(groups.quadrupole_observations_of(430.0), &[1]);
    assert_ne!(groups.group_of(410.0), groups.group_of(430.0));
    assert_eq!(groups.group_of(400.0), groups.group_of(424.9));
    assert_eq!(groups.num_groups(), 3);
}

#[test]
fn test_upper_limit_is_exclusive() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [425.0, 450.0]]);

    // when / then
    assert_eq!(groups.quadrupole_observations_of(425.0), &[1]);
    assert_eq!(groups.group_of(450.0), NO_WINDOW_GROUP);
    assert_eq!(groups.quadrupole_observations_of(450.0), &[] as &[usize]);
}

#[test]
fn test_overlap_is_a_group_of_its_own() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [424.0, 450.0]]);

    // when / then
    assert_eq!(groups.quadrupole_observations_of(410.0), &[0]);
    assert_eq!(groups.quadrupole_observations_of(424.5), &[0, 1]);
    assert_eq!(groups.quadrupole_observations_of(425.0), &[1]);
    let distinct: std::collections::HashSet<u32> = [410.0, 424.5, 430.0]
        .iter()
        .map(|&mz| groups.group_of(mz))
        .collect();
    assert_eq!(distinct.len(), 3);
}

#[test]
fn test_segments_with_the_same_windows_share_a_group() {
    // given: window 1 lies inside window 0, so window 0 alone covers both sides of it
    let groups = WindowGroups::new(&[[400.0, 450.0], [420.0, 430.0]]);

    // when / then
    assert_eq!(groups.group_of(410.0), groups.group_of(440.0));
    assert_eq!(groups.quadrupole_observations_of(425.0), &[0, 1]);
    assert_eq!(groups.num_groups(), 3);
}

#[test]
fn test_repeated_window_reads_all_its_quadrupole_observations() {
    // given: the same isolation window is acquired twice per cycle
    let groups = WindowGroups::new(&[[400.0, 425.0], [400.0, 425.0]]);

    // when / then
    assert_eq!(groups.quadrupole_observations_of(410.0), &[0, 1]);
}

#[test]
fn test_outside_all_windows_is_no_window_group() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [500.0, 525.0]]);

    // when / then
    for mz in [100.0, 450.0, 600.0, f32::NAN] {
        assert_eq!(groups.group_of(mz), NO_WINDOW_GROUP, "{mz}");
    }
    assert_eq!(
        groups.quadrupole_observations(NO_WINDOW_GROUP),
        &[] as &[usize]
    );
}

#[test]
fn test_empty_and_nan_windows_contain_nothing() {
    // given: an MS1 scan without isolation, a NaN limit and a valid window
    let groups = WindowGroups::new(&[[-1.0, -1.0], [f32::NAN, 500.0], [400.0, 425.0]]);

    // when / then
    assert_eq!(groups.quadrupole_observations_of(410.0), &[2]);
    assert_eq!(groups.group_of(-1.0), NO_WINDOW_GROUP);
}

#[test]
fn test_without_windows_everything_is_no_window_group() {
    // given
    let groups = WindowGroups::default();

    // when / then
    assert_eq!(groups.group_of(500.0), NO_WINDOW_GROUP);
    assert_eq!(groups.num_groups(), 1);
}
