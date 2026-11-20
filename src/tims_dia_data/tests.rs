use super::*;
use crate::dense_xic_observation::DenseXICObservation;
use crate::traits::{DIADataTrait, MobilityFilter};

const FRAGMENT_MZ: f32 = 600.0;

/// Two window groups over 4 scans (1/K0 1.5, 1.25, 1.0, 0.75, exact in f32 so a mobility
/// window can end exactly on a scan). Every scan of both groups has one peak at `FRAGMENT_MZ`.
/// - group 0: scans 0-2 isolate 400-500, scan 3 isolates 900-1000; peaks in cycle 1 with
///   intensity 10^scan, so each scan's contribution is visible in the sum.
/// - group 1: every scan isolates 700-800; peaks in cycle 2 with intensity 2 * 10^scan.
fn tims_data() -> TimsDIAData {
    let group_0 = observation(
        vec![
            Some([400.0, 500.0]),
            Some([400.0, 500.0]),
            Some([400.0, 500.0]),
            Some([900.0, 1000.0]),
        ],
        1,
        [1.0, 10.0, 100.0, 1000.0],
    );
    let group_1 = observation(vec![Some([700.0, 800.0]); 4], 2, [2.0, 20.0, 200.0, 2000.0]);

    TimsDIAData {
        rt_index: RTIndex::from_rt(Array1::from_vec(vec![0.0, 1.0, 2.0])),
        quadrupole_observations: vec![group_0, group_1],
        scan_mobility: vec![1.5, 1.25, 1.0, 0.75],
        rt_values: Array1::zeros(0),
        cycle: Array4::zeros((0, 0, 0, 0)),
        num_groups: 2,
    }
}

/// One window group with one peak at `FRAGMENT_MZ` on each of its 4 scans, all in `cycle`.
fn observation(
    scan_windows: Vec<Option<[f32; 2]>>,
    cycle: u16,
    intensities: [f32; 4],
) -> TimsQuadrupoleObservation {
    let mz_index = MZIndex::global();
    let mut obs = TimsQuadrupoleObservation::new(scan_windows);
    let bin = mz_index.find_closest_index(FRAGMENT_MZ);
    obs.slice_starts = (0..=mz_index.len())
        .map(|i| if i <= bin { 0 } else { 4 })
        .collect();
    obs.cycle_indices = vec![cycle; 4];
    obs.scan_indices = vec![0, 1, 2, 3];
    obs.intensities = intensities.to_vec();
    obs
}

/// The XIC of `FRAGMENT_MZ` over all 3 cycles.
fn xic(data: &TimsDIAData, precursor_mz: f32, mobility: MobilityFilter) -> Vec<f32> {
    xic_cycles(data, precursor_mz, mobility, 0, 3)
}

fn xic_cycles(
    data: &TimsDIAData,
    precursor_mz: f32,
    mobility: MobilityFilter,
    cycle_start: usize,
    cycle_stop: usize,
) -> Vec<f32> {
    DenseXICObservation::new(
        data,
        precursor_mz,
        mobility,
        cycle_start,
        cycle_stop,
        10.0,
        &[FRAGMENT_MZ],
    )
    .dense_xic
    .row(0)
    .to_vec()
}

#[test]
fn test_without_mobility_all_isolating_scans_count() {
    let data = tims_data();
    assert_eq!(
        xic(&data, 450.0, MobilityFilter::Any),
        vec![0.0, 111.0, 0.0]
    );
}

#[test]
fn test_mobility_filter_keeps_only_scans_in_range() {
    let data = tims_data();
    let mobility = MobilityFilter::Window {
        mobility: 1.25,
        tolerance: 0.05,
    };
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 10.0, 0.0]);
}

#[test]
fn test_scans_not_isolating_the_precursor_are_ignored() {
    let data = tims_data();
    let mobility = MobilityFilter::Window {
        mobility: 0.75,
        tolerance: 0.05,
    };
    // scan 3 is in mobility range but isolates 900-1000, not 450
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 0.0, 0.0]);
    assert_eq!(xic(&data, 950.0, mobility), vec![0.0, 1000.0, 0.0]);
}

#[test]
fn test_precursor_outside_all_windows_has_no_observation() {
    let data = tims_data();
    assert!(data
        .select_observations(600.0, MobilityFilter::Any)
        .is_empty());
    let selected = data.select_observations(450.0, MobilityFilter::Any);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].0, 0);
}

#[test]
fn test_no_library_mobility_or_an_infinite_tolerance_does_not_restrict_scans() {
    let data = tims_data();
    for (mobility, tolerance) in [(f32::NAN, 0.05), (1.25, f32::INFINITY)] {
        let filter = MobilityFilter::from_library(mobility, tolerance).unwrap();
        assert_eq!(xic(&data, 450.0, filter), vec![0.0, 111.0, 0.0]);
    }
}

#[test]
fn test_each_observation_reads_its_own_peaks() {
    let data = tims_data();
    let selected = data.select_observations(750.0, MobilityFilter::Any);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].0, 1);
    assert_eq!(
        xic(&data, 750.0, MobilityFilter::Any),
        vec![0.0, 0.0, 2222.0]
    );
}

#[test]
fn test_xic_cycles_are_relative_to_cycle_start() {
    let data = tims_data();
    assert_eq!(
        xic_cycles(&data, 450.0, MobilityFilter::Any, 1, 3),
        vec![111.0, 0.0]
    );
    assert_eq!(
        xic_cycles(&data, 750.0, MobilityFilter::Any, 1, 3),
        vec![0.0, 2222.0]
    );
    assert_eq!(
        xic_cycles(&data, 750.0, MobilityFilter::Any, 2, 3),
        vec![2222.0]
    );
}

#[test]
fn test_mobility_window_edges_are_inclusive() {
    let data = tims_data();
    // 1.25 ± 0.25 is exactly [1.0, 1.5]: scans 0 and 2 sit on its edges
    let upper = MobilityFilter::Window {
        mobility: 1.25,
        tolerance: 0.25,
    };
    assert_eq!(xic(&data, 750.0, upper), vec![0.0, 0.0, 222.0]);
    // 0.875 ± 0.125 is exactly [0.75, 1.0]: scans 2 and 3, the last scan
    let lower = MobilityFilter::Window {
        mobility: 0.875,
        tolerance: 0.125,
    };
    assert_eq!(xic(&data, 750.0, lower), vec![0.0, 0.0, 2200.0]);
}

#[test]
fn test_panic_message_keeps_the_panic_text() {
    let payload: Box<dyn std::any::Any + Send> = Box::new("bad metadata");
    assert_eq!(panic_message(payload.as_ref()), "bad metadata");
    let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("bad frame"));
    assert_eq!(panic_message(payload.as_ref()), "bad frame");
    let payload: Box<dyn std::any::Any + Send> = Box::new(42);
    assert_eq!(panic_message(payload.as_ref()), "no panic message");
}

mod empty_frames {
    use super::super::builder::empty_frame_as_none;
    use timsrust::core::{FrameIons, FrameReaderError};

    #[test]
    fn test_an_empty_blob_is_an_empty_frame() {
        let result = empty_frame_as_none(Err(FrameReaderError::new("No binary data")));
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn test_frames_read_without_error_are_kept() {
        let ions = FrameIons::new(vec![0, 0], vec![], vec![]);
        let result = empty_frame_as_none(Ok(ions.clone()));
        assert_eq!(result, Ok(Some(ions)));
    }

    #[test]
    fn test_other_read_errors_are_still_errors() {
        let result = empty_frame_as_none(Err(FrameReaderError::new("Corrupt Frame")));
        assert!(result.unwrap_err().contains("Corrupt Frame"));
    }
}

mod fill_group {
    use super::super::builder::fill_group;
    use super::*;

    const NUM_BINS: usize = 3;

    #[test]
    fn test_peaks_on_the_same_bin_cycle_and_scan_are_summed() {
        // bin 0, cycle 1 has two peaks on scan 0 (from two frames) and one on scan 1;
        // bin 2 has one peak on scan 3
        let staged = vec![
            vec![(0, 1, 0, 1.0), (0, 1, 1, 2.0)],
            vec![(0, 1, 0, 3.0), (2, 0, 3, 0.5)],
        ];
        let mut obs = TimsQuadrupoleObservation::new(vec![Some([400.0, 500.0]); 4]);
        fill_group(&mut obs, staged, NUM_BINS).unwrap();
        assert_eq!(obs.slice_starts, vec![0, 2, 2, 3]);
        assert_eq!(obs.cycle_indices, vec![1, 1, 0]);
        assert_eq!(obs.scan_indices, vec![0, 1, 3]);
        assert_eq!(obs.intensities, vec![4.0, 2.0, 0.5]);
    }
}

#[test]
fn test_zero_mobility_tolerance_keeps_the_matching_scan() {
    let data = tims_data();
    let mobility = MobilityFilter::Window {
        mobility: 1.25,
        tolerance: 0.0,
    };
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 10.0, 0.0]);
}

#[test]
fn test_mobility_outside_the_scan_range_has_no_observation() {
    let data = tims_data();
    let mobility = MobilityFilter::Window {
        mobility: 2.0,
        tolerance: 0.05,
    };
    assert!(data.select_observations(450.0, mobility).is_empty());
}

mod layout {
    use super::super::layout::{assign_cycles, frame_rt_seconds, window_per_scan, FrameMeta};
    use timsrust::core::{IsolationWindow, Mz, QuadrupoleSettings};

    fn ms1(frame_index: usize, rt_seconds: f32) -> FrameMeta {
        FrameMeta {
            frame_index,
            is_ms1: true,
            settings_id: 0,
            rt_seconds,
        }
    }

    fn ms2(frame_index: usize, settings_id: usize, rt_seconds: f32) -> FrameMeta {
        FrameMeta {
            frame_index,
            is_ms1: false,
            settings_id,
            rt_seconds,
        }
    }

    #[test]
    fn test_frames_are_assigned_to_cycles_and_groups() {
        // an MS2 frame before the first MS1 frame, then two cycles; group ids 7 and 3
        // become groups 0 and 1 in order of first appearance, and cycle 1 lacks group 3
        let frames = vec![
            ms2(1, 3, 0.5),
            ms1(2, 1.0),
            ms2(3, 7, 1.1),
            ms2(4, 3, 1.2),
            ms1(5, 2.0),
            ms2(6, 7, 2.1),
        ];
        let cycles = assign_cycles(frames).unwrap();
        assert_eq!(cycles.cycle_rt_seconds, vec![1.0, 2.0]);
        assert_eq!(cycles.settings_ids, vec![7, 3]);
        let placed: Vec<_> = cycles
            .ms2_frames
            .iter()
            .map(|f| (f.frame_index, f.cycle, f.group))
            .collect();
        assert_eq!(placed, vec![(3, 0, 0), (4, 0, 1), (6, 1, 0)]);
        // the missing frame slot keeps its cycle's MS1 RT
        assert_eq!(
            frame_rt_seconds(&cycles),
            vec![1.0, 1.1, 1.2, 2.0, 2.1, 2.0]
        );
    }

    #[test]
    fn test_two_ms1_frames_per_cycle_are_rejected() {
        let frames = vec![
            ms1(1, 1.0),
            ms2(2, 1, 1.1),
            ms2(3, 2, 1.2),
            ms1(4, 1.3),
            ms2(5, 3, 1.4),
            ms2(6, 4, 1.5),
            ms1(7, 2.0),
            ms2(8, 1, 2.1),
            ms2(9, 2, 2.2),
            ms1(10, 2.3),
            ms2(11, 3, 2.4),
            ms2(12, 4, 2.5),
        ];
        let error = assign_cycles(frames).err().unwrap();
        assert!(
            error.contains("cycle 0 is missing a window group"),
            "{error}"
        );
    }

    #[test]
    fn test_a_group_twice_in_one_cycle_is_rejected() {
        let frames = vec![
            ms1(1, 1.0),
            ms2(2, 7, 1.1),
            ms2(3, 3, 1.2),
            ms1(4, 2.0),
            ms2(5, 7, 2.1),
            ms2(6, 3, 2.2),
            ms2(7, 7, 2.3),
            ms1(8, 3.0),
            ms2(9, 7, 3.1),
            ms2(10, 3, 3.2),
        ];
        let error = assign_cycles(frames).err().unwrap();
        assert!(
            error.contains("window group 7 appears more than once in cycle 1"),
            "{error}"
        );
    }

    #[test]
    fn test_only_the_last_cycle_may_be_incomplete() {
        let last_cut_off = vec![
            ms1(1, 1.0),
            ms2(2, 7, 1.1),
            ms2(3, 3, 1.2),
            ms1(4, 2.0),
            ms2(5, 7, 2.1),
        ];
        assert!(assign_cycles(last_cut_off).is_ok());

        let middle_cut_off = vec![
            ms1(1, 1.0),
            ms2(2, 7, 1.1),
            ms2(3, 3, 1.2),
            ms1(4, 2.0),
            ms2(5, 7, 2.1),
            ms1(6, 3.0),
            ms2(7, 7, 3.1),
            ms2(8, 3, 3.2),
        ];
        let error = assign_cycles(middle_cut_off).err().unwrap();
        assert!(
            error.contains("cycle 1 is missing a window group"),
            "{error}"
        );
    }

    #[test]
    fn test_runs_without_cycles_or_windows_are_rejected() {
        assert!(assign_cycles(vec![ms1(1, 1.0), ms1(2, 2.0)]).is_err());
        assert!(assign_cycles(vec![ms2(1, 1, 1.0)]).is_err());
    }

    #[test]
    fn test_more_cycles_than_u16_is_an_error() {
        let mut frames: Vec<_> = (0..=u16::MAX as usize + 1)
            .map(|i| ms1(i, i as f32))
            .collect();
        frames.push(ms2(frames.len(), 1, 0.0));
        assert!(assign_cycles(frames).is_err());
    }

    #[test]
    fn test_scan_ends_are_exclusive_and_clamped() {
        let window =
            |lo: f64, hi: f64| IsolationWindow::new_from_bounds(Mz::from(lo), Mz::from(hi), 0.0);
        let settings = QuadrupoleSettings {
            index: 1,
            scan_starts: vec![1, 3],
            scan_ends: vec![3, 9],
            isolation_windows: vec![window(800.0, 825.0), window(600.0, 625.0)],
        };
        let windows = window_per_scan(&settings, 5);
        assert_eq!(
            windows,
            vec![
                None,
                Some([800.0, 825.0]),
                Some([800.0, 825.0]),
                Some([600.0, 625.0]),
                Some([600.0, 625.0]),
            ]
        );
    }

    #[test]
    fn test_settings_without_windows_isolate_nothing() {
        let windows = window_per_scan(&QuadrupoleSettings::default(), 3);
        assert_eq!(windows, vec![None; 3]);
    }
}

#[test]
fn test_ms1_observation_is_never_an_ms2_window_and_comes_first_in_the_cycle() {
    let mut data = tims_data();
    data.quadrupole_observations
        .push(TimsQuadrupoleObservation::new_ms1(4));
    assert!(data.has_ms1());
    assert_eq!(data.ms1_observations(), vec![2]);
    for mz in [450.0, 750.0, 950.0] {
        assert!(data
            .select_observations(mz, MobilityFilter::Any)
            .iter()
            .all(|(obs, _)| *obs != 2));
    }
    assert_eq!(data.cycle_position(2), 0.0);
    assert_eq!(data.cycle_position(0), 1.0 / 3.0);
    assert_eq!(
        data.ms1_scan_filter(
            2,
            MobilityFilter::Window {
                mobility: 1.0,
                tolerance: 0.3
            }
        ),
        vec![false, true, true, true]
    );
}

fn precursor_at(mz: f32, mobility: f32) -> crate::precursor::Precursor {
    crate::precursor::Precursor {
        precursor_idx: 0,
        mz,
        mz_library: mz,
        rt: 0.0,
        rt_library: 0.0,
        mobility,
        mobility_tolerance_scale: 1.0,
        naa: 8,
        fragment_mz: vec![FRAGMENT_MZ],
        fragment_mz_library: vec![FRAGMENT_MZ],
        fragment_intensity: vec![1.0],
        fragment_cardinality: vec![1],
        fragment_charge: vec![1],
        fragment_loss_type: vec![0],
        fragment_number: vec![1],
        fragment_position: vec![0],
        fragment_type: vec![0],
    }
}

#[test]
fn test_recentering_moves_the_window_to_the_intensity_weighted_apex() {
    use crate::peak_group_scoring::mobility::recentered_mobility_filter;
    // group 0 isolates 450 on scans 0-2 (1/K0 1.5, 1.25, 1.0) with intensities 1, 10, 100 in cycle 1
    let data = tims_data();
    let precursor = precursor_at(450.0, 1.25);
    let wide = precursor.mobility_filter(0.3);

    let filter = recentered_mobility_filter(&data, &precursor, 1, wide, 20.0, 0.05, 1.0);

    let expected = (1.5 + 12.5 + 100.0) / 111.0;
    match filter {
        MobilityFilter::Window {
            mobility,
            tolerance,
        } => {
            assert!((mobility - expected).abs() < 1e-5, "{mobility}");
            assert_eq!(tolerance, 0.05);
        }
        MobilityFilter::Any => panic!("expected a window"),
    }
}

#[test]
fn test_recentering_keeps_the_wide_window_when_off_or_without_signal() {
    use crate::peak_group_scoring::mobility::recentered_mobility_filter;
    let data = tims_data();
    let precursor = precursor_at(450.0, 1.25);
    let wide = precursor.mobility_filter(0.3);

    assert_eq!(
        recentered_mobility_filter(&data, &precursor, 1, wide, 20.0, 0.0, 1.0),
        wide
    );
    // only scan 3 (1/K0 0.75) isolates 950, outside the window around 1.5
    let far = precursor_at(950.0, 1.5);
    let far_wide = far.mobility_filter(0.1);
    assert_eq!(
        recentered_mobility_filter(&data, &far, 1, far_wide, 20.0, 0.05, 1.0),
        far_wide
    );
}
