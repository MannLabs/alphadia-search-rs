use super::*;
use crate::constants::{FragmentType, Loss};
use crate::dia_data::AlphaRawView;
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::{ArrayView1, ArrayView4};

/// One MS2 window [400, 600] per cycle, three cycles; fragments at 300 (1+) and 450 (2+) with their isotopes.
fn three_cycle_data() -> DIAData {
    let spectrum_delta_scan_idx = [0i64, 0, 0];
    let isolation_lower_mz = [400.0f32, 400.0, 400.0];
    let isolation_upper_mz = [600.0f32, 600.0, 600.0];
    let spectrum_peak_start_idx = [0i64, 2, 6];
    let spectrum_peak_stop_idx = [2i64, 6, 8];
    let spectrum_cycle_idx = [0i64, 1, 2];
    let spectrum_rt = [1.0f32, 2.0, 3.0];
    let peak_mz = [
        300.0f32, 450.0, 300.003, 301.003_35, 300.501_7, 450.501_7, 300.0, 450.0,
    ];
    let peak_intensity = [10.0f32, 5.0, 100.0, 40.0, 7.0, 3.0, 20.0, 9.0];
    let cycle = [2.0f32];
    let view = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&spectrum_delta_scan_idx),
        isolation_lower_mz: ArrayView1::from(&isolation_lower_mz),
        isolation_upper_mz: ArrayView1::from(&isolation_upper_mz),
        spectrum_peak_start_idx: ArrayView1::from(&spectrum_peak_start_idx),
        spectrum_peak_stop_idx: ArrayView1::from(&spectrum_peak_stop_idx),
        spectrum_cycle_idx: ArrayView1::from(&spectrum_cycle_idx),
        spectrum_rt: ArrayView1::from(&spectrum_rt),
        peak_mz: ArrayView1::from(&peak_mz),
        peak_intensity: ArrayView1::from(&peak_intensity),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &cycle).unwrap(),
    };
    DIADataBuilder::from_alpha_raw(&view)
}

/// One precursor at 500 with fragments 450 2+ (intensity 0.5), 300 1+ (1.0) and a y1 (2.0, never exported).
fn one_precursor_speclib() -> SpecLibFlat {
    SpecLibFlat::from_vecs(
        vec![0],
        vec![500.0],
        vec![500.0],
        vec![2.0],
        vec![2.0],
        vec![9],
        vec![0],
        vec![3],
        vec![450.0, 300.0, 200.0],
        vec![450.0, 300.0, 200.0],
        vec![0.5, 1.0, 2.0],
        vec![1, 1, 1],
        vec![2, 1, 1],
        vec![Loss::NONE, Loss::NONE, Loss::NONE],
        vec![7, 3, 1],
        vec![7, 3, 1],
        vec![FragmentType::B, FragmentType::B, FragmentType::Y],
    )
}

#[test]
fn companion_is_half_spacing_for_1plus_and_full_spacing_otherwise() {
    assert!((companion_offset(1) - C13_MASS_DELTA / 2.0).abs() < 1e-6);
    assert!((companion_offset(2) - C13_MASS_DELTA).abs() < 1e-6);
}

#[test]
fn traces_follow_slot_order_and_window() {
    let dia_data = three_cycle_data();
    let lib = one_precursor_speclib();
    let tic = observation_tic(&dia_data, 3);
    assert_eq!(tic.row(0).to_vec(), vec![15.0, 150.0, 29.0]);

    let params = TraceParameters {
        mass_tolerance: 20.0,
        top_k: 3,
        extend: 1,
        window: 5,
        neighbour_ppm: 50.0,
    };
    let candidate = Candidate::new(0, 0, 1.0, 0, 1, 3);
    let t = candidate_traces(&dia_data, &lib, &candidate, &tic.view(), &params);

    assert_eq!(t.window_start, -1);
    // slot 0 = most intense non-y1 fragment (300 1+), slot 1 = 450 2+, slot 2 empty
    assert_eq!(t.fragment_mz_library, vec![300.0, 450.0, 0.0]);
    assert_eq!(t.fragment_charge, vec![1, 2, 0]);
    // window cycles -1..4 map to slots 0..5; data cycles 0, 1, 2 sit at 1, 2, 3
    assert_eq!(&t.xic[0..5], &[0.0, 10.0, 100.0, 20.0, 0.0]);
    assert_eq!(&t.xic[5..10], &[0.0, 5.0, 0.0, 9.0, 0.0]);
    assert!((t.mz_error_ppm[2] - 10.0).abs() < 1.5);
    assert!(t.mz_error_ppm[0].is_nan());
    assert_eq!(t.isotope_xic[2], 40.0);
    assert_eq!(t.companion_xic[2], 7.0);
    // the 2+ fragment's isotope sits at +C13/2 = its companion of a 1+ fragment
    assert_eq!(t.isotope_xic[5 + 2], 3.0);
    assert_eq!(t.apex_tic, 150.0);
    // 300.0 has no peak at the apex cycle, but 300.003 (10 ppm) does
    assert_eq!(t.apex_neighbours[0], 1);
    assert_eq!(t.apex_neighbours[2], 0);
}
