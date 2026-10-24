use super::*;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::{AlphaRawView, DIAData};
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::{Array4, ArrayView1, ArrayView4};

const ISOTOPES: [f32; 4] = [0.5, 0.3, 0.15, 0.05];
/// A heavy peptide: M1 is the most abundant isotope
const HEAVY_ISOTOPES: [f32; 4] = [0.2, 0.35, 0.3, 0.15];
const FRAGMENT_MZ: f32 = 400.0;
const LOWER_WINDOW: [f32; 2] = [498.0, 500.0];
const UPPER_WINDOW: [f32; 2] = [500.0, 502.0];

/// Two adjacent 2 Th windows, [498, 500] and [500, 502], in one cycle, recording the given (m/z, intensity) peaks.
fn two_window_data(lower: &[(f32, f32)], upper: &[(f32, f32)]) -> DIAData {
    let delta_scan_idx = [0_i64, 1];
    let isolation_lower = [LOWER_WINDOW[0], UPPER_WINDOW[0]];
    let isolation_upper = [LOWER_WINDOW[1], UPPER_WINDOW[1]];
    let peak_start = [0_i64, lower.len() as i64];
    let peak_stop = [lower.len() as i64, (lower.len() + upper.len()) as i64];
    let cycle_idx = [0_i64, 0];
    let rt = [10.0_f32, 10.1];
    let peak_mz: Vec<f32> = lower.iter().chain(upper).map(|p| p.0).collect();
    let peak_intensity: Vec<f32> = lower.iter().chain(upper).map(|p| p.1).collect();
    let cycle = [1.0_f32];
    let view = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&delta_scan_idx),
        isolation_lower_mz: ArrayView1::from(&isolation_lower),
        isolation_upper_mz: ArrayView1::from(&isolation_upper),
        spectrum_peak_start_idx: ArrayView1::from(&peak_start),
        spectrum_peak_stop_idx: ArrayView1::from(&peak_stop),
        spectrum_cycle_idx: ArrayView1::from(&cycle_idx),
        spectrum_rt: ArrayView1::from(&rt),
        peak_mz: ArrayView1::from(&peak_mz),
        peak_intensity: ArrayView1::from(&peak_intensity),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &cycle).unwrap(),
    };
    DIADataBuilder::from_alpha_raw(&view)
}

fn empty_data() -> DIAData {
    two_window_data(&[], &[])
}

fn precursor(mz: f32, charge: u8, isotope_intensity: Vec<f32>) -> Precursor {
    Precursor {
        precursor_idx: 0,
        mz,
        mz_library: mz,
        rt: 10.0,
        rt_library: 10.0,
        naa: 9,
        charge,
        isotope_intensity,
        fragment_mz: vec![FRAGMENT_MZ],
        fragment_mz_library: vec![FRAGMENT_MZ],
        fragment_intensity: vec![1.0],
        fragment_cardinality: vec![1],
        fragment_charge: vec![1],
        fragment_loss_type: vec![0],
        fragment_number: vec![3],
        fragment_position: vec![3],
        fragment_type: vec![121],
    }
}

/// Expected gain of fragment isotope j in a window holding precursor isotopes `from..` only (rectangle model).
fn rectangle_gain(isotopes: &[f32], from: usize, j: usize, share: f32) -> f32 {
    let reference: f32 = isotopes
        .iter()
        .enumerate()
        .map(|(k, &abundance)| abundance * binomial(k, 0, share))
        .sum();
    isotopes
        .iter()
        .enumerate()
        .skip(from)
        .map(|(k, &abundance)| abundance * binomial(k, j, share))
        .sum::<f32>()
        / reference
}

#[test]
fn test_transmission_is_one_inside_half_at_edges_zero_outside() {
    assert!((transmission(501.0, 500.0, 502.0) - 1.0).abs() < 1e-3);
    assert!((transmission(500.0, 500.0, 502.0) - 0.5).abs() < 1e-3);
    assert!((transmission(502.0, 500.0, 502.0) - 0.5).abs() < 1e-3);
    assert!(transmission(498.5, 500.0, 502.0) < 1e-3);
    assert_eq!(transmission(450.0, 500.0, 502.0), 0.0);
}

#[test]
fn test_binomial_sums_to_one_over_fragment_isotopes() {
    for k in 0..5 {
        let total: f32 = (0..=k).map(|j| binomial(k, j, 0.3)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
    assert_eq!(binomial(1, 2, 0.3), 0.0);
    assert!((binomial(2, 1, 0.3) - 2.0 * 0.3 * 0.7).abs() < 1e-6);
}

#[test]
fn test_monoisotopic_plan_uses_windows_containing_precursor() {
    let dia_data = empty_data();

    let plan = ExtractionPlan::monoisotopic(&dia_data, 499.0, &[300.0, 400.0]);

    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert_eq!(plan.primary.channels.len(), 2);
    assert!(plan.primary.channels.iter().all(|c| c.mz_shift == 0.0));
    assert_eq!(plan.primary.fragment_gain, vec![1.0, 1.0]);
    assert!(plan.secondary.is_none());
}

#[test]
fn test_unusable_isotopes_fall_back_to_monoisotopic_plan() {
    let dia_data = empty_data();
    for (charge, isotopes) in [
        (0, Vec::new()),
        (2, vec![f32::NAN, 0.3]),
        (2, vec![0.0, 0.0]),
    ] {
        let plan = ExtractionPlan::new(&dia_data, &precursor(499.8, charge, isotopes));

        // M0 at 499.8 is in the lower window only
        assert_eq!(plan.primary.observation_indices(), vec![0]);
        assert_eq!(plan.primary.fragment_gain, vec![1.0]);
        assert!(plan.secondary.is_none());
    }
}

#[test]
fn test_envelope_inside_one_window_is_extracted_there_at_gain_one() {
    // Given: M0..M3 of a 2+ precursor at 498.6..500.1 lie almost fully in [498, 500]
    let plan = ExtractionPlan::new(&empty_data(), &precursor(498.6, 2, ISOTOPES.to_vec()));

    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert_eq!(plan.primary.channels[0].mz_shift, 0.0);
    assert!(
        plan.primary.fragment_gain[0] > 0.9,
        "{:?}",
        plan.primary.fragment_gain
    );
    // the upper window holds less than SECONDARY_MIN_SIGNAL_RATIO of the envelope
    assert!(plan.secondary.is_none());
}

#[test]
fn test_primary_window_is_the_one_with_most_of_the_envelope() {
    // Given: M0 (499.7) of a heavy 2+ precursor in the lower window, M1..M3 (500.2..501.2) in the upper one
    let plan = ExtractionPlan::new(&empty_data(), &precursor(499.7, 2, HEAVY_ISOTOPES.to_vec()));

    // Then: the upper window is primary, the lower one secondary
    assert_eq!(plan.primary.observation_indices(), vec![1]);
    let secondary = plan.secondary.as_ref().expect("secondary window");
    assert_eq!(secondary.observation_indices(), vec![0]);
    assert!(
        plan.primary.precursor_share > 0.7,
        "{}",
        plan.primary.precursor_share
    );
    assert!(
        secondary.precursor_share > 0.15,
        "{}",
        secondary.precursor_share
    );
}

#[test]
fn test_each_fragment_is_extracted_at_its_strongest_isotope() {
    // Given: the heavy precursor of the previous test, a short fragment (share ~0.2) and a long one (~0.6)
    let precursor_mz = 499.7;
    let precursor_mass = (precursor_mz - PROTON_MASS) * 2.0;
    let fragments = [200.0, 600.0];
    let shares: Vec<f32> = fragments
        .iter()
        .map(|&mz| (mz - PROTON_MASS) / precursor_mass)
        .collect();

    let plan = ExtractionPlan::isotope_envelope(
        &empty_data(),
        precursor_mz,
        2,
        &HEAVY_ISOTOPES,
        &fragments,
        &[1, 1],
    );

    // Then: the short fragment keeps its monoisotopic m/z, the long one moves to its first isotope
    let shifts: Vec<f32> = plan.primary.channels.iter().map(|c| c.mz_shift).collect();
    assert_eq!(shifts, vec![0.0, ISOTOPE_SPACING]);
    assert!((plan.primary.channels[1].mz - (600.0 + ISOTOPE_SPACING)).abs() < 1e-4);
    // and the gains follow the isotope binomial of the transmitted precursor isotopes M1..M3
    let short_gain = rectangle_gain(&HEAVY_ISOTOPES, 1, 0, shares[0]);
    let long_gain = rectangle_gain(&HEAVY_ISOTOPES, 1, 1, shares[1]);
    assert!((plan.primary.fragment_gain[0] - short_gain).abs() < 0.05);
    assert!((plan.primary.fragment_gain[1] - long_gain).abs() < 0.05);
}

#[test]
fn test_expected_intensity_scales_library_by_gain() {
    let window = WindowChannels {
        channels: Vec::new(),
        fragment_gain: vec![1.0, 0.5, 0.0],
        precursor_share: 1.0,
    };

    assert_eq!(
        window.expected_intensity(&[2.0, 2.0, 2.0]),
        vec![2.0, 1.0, 0.0]
    );
}

#[test]
fn test_dense_mz_reports_monoisotopic_mz_for_isotope_channels() {
    // Given: the long fragment's first isotope recorded in the upper (primary) window
    let dia_data = two_window_data(&[], &[(600.0 + ISOTOPE_SPACING, 500.0)]);
    let plan =
        ExtractionPlan::isotope_envelope(&dia_data, 499.7, 2, &HEAVY_ISOTOPES, &[600.0], &[1]);

    // When
    let dense = DenseXICMZObservation::new(&dia_data, &plan.primary, 0, 1, 20.0);

    // Then: its signal is found and its m/z is reported at the monoisotopic fragment
    assert_eq!(dense.dense_xic[[0, 0]], 500.0);
    let mz_error_ppm = (dense.dense_mz[[0, 0]] - 600.0) / 600.0 * 1e6;
    assert!(mz_error_ppm.abs() < 5.0, "{mz_error_ppm} ppm");
}

#[test]
fn test_primary_window_mz_is_inside_the_primary_window() {
    let windows = vec![[-1.0, -1.0], LOWER_WINDOW, UPPER_WINDOW];

    assert_eq!(
        primary_window_mz(499.7, 2, &HEAVY_ISOTOPES, &windows),
        501.0
    );
    assert_eq!(primary_window_mz(498.6, 2, &ISOTOPES, &windows), 499.0);
    assert_eq!(
        primary_window_mz(499.7, 0, &HEAVY_ISOTOPES, &windows),
        499.7
    );
    assert_eq!(primary_window_mz(499.7, 2, &[0.0, 0.0], &windows), 499.7);
    assert_eq!(primary_window_mz(300.0, 2, &ISOTOPES, &windows), 300.0);
}

#[test]
fn test_cycle_windows_span_all_scans() {
    let mut cycle = Array4::<f32>::zeros((1, 2, 2, 2));
    cycle[[0, 0, 0, 0]] = -1.0;
    cycle[[0, 0, 0, 1]] = -1.0;
    cycle[[0, 0, 1, 0]] = -1.0;
    cycle[[0, 0, 1, 1]] = -1.0;
    cycle[[0, 1, 0, 0]] = 500.0;
    cycle[[0, 1, 0, 1]] = 501.0;
    cycle[[0, 1, 1, 0]] = 500.5;
    cycle[[0, 1, 1, 1]] = 502.0;

    assert_eq!(
        cycle_windows(cycle.view()),
        vec![[-1.0, -1.0], [500.0, 502.0]]
    );
}
