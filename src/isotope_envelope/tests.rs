use super::*;
use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::dia_data::{AlphaRawView, DIAData};
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::{ArrayView1, ArrayView4};

const ISOTOPES: [f32; 4] = [0.5, 0.3, 0.15, 0.05];
const FRAGMENT_MZ: f32 = 400.0;
const LOWER_WINDOW: [f32; 2] = [498.0, 500.0];
const UPPER_WINDOW: [f32; 2] = [500.0, 502.0];

/// Two adjacent 2 Th windows, [498, 500] and [500, 502], in one cycle.
/// Each records the monoisotopic fragment and its first isotope with the given intensities.
fn two_window_data(lower: [f32; 2], upper: [f32; 2]) -> DIAData {
    let delta_scan_idx = [0_i64, 1];
    let isolation_lower = [LOWER_WINDOW[0], UPPER_WINDOW[0]];
    let isolation_upper = [LOWER_WINDOW[1], UPPER_WINDOW[1]];
    let peak_start = [0_i64, 2];
    let peak_stop = [2_i64, 4];
    let cycle_idx = [0_i64, 0];
    let rt = [10.0_f32, 10.1];
    let isotope_mz = FRAGMENT_MZ + ISOTOPE_SPACING;
    let peak_mz = [FRAGMENT_MZ, isotope_mz, FRAGMENT_MZ, isotope_mz];
    let peak_intensity = [lower[0], lower[1], upper[0], upper[1]];
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

/// Plan with monoisotopic and first-isotope fragment channels.
fn plan_at(dia_data: &DIAData, precursor_mz: f32) -> ExtractionPlan {
    ExtractionPlan::isotope_envelope(
        dia_data,
        precursor_mz,
        2,
        &ISOTOPES,
        &[FRAGMENT_MZ],
        &[1],
        1,
    )
}

fn sum_weight_times_gain(plan: &ExtractionPlan, fragment_idx: usize) -> f32 {
    plan.channels
        .iter()
        .filter(|channel| channel.fragment_idx == fragment_idx)
        .map(|channel| channel.weight * channel.gain)
        .sum()
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
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);

    let plan = ExtractionPlan::monoisotopic(&dia_data, 499.0, &[300.0, 400.0]);

    assert_eq!(plan.observation_indices, vec![0]);
    assert_eq!(plan.channels.len(), 2);
    assert!(plan
        .channels
        .iter()
        .all(|channel| channel.weight == 1.0 && channel.mz_shift == 0.0));
}

#[test]
fn test_precursor_without_isotopes_gets_monoisotopic_plan() {
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);
    let precursor = Precursor {
        precursor_idx: 0,
        mz: 499.8,
        mz_library: 499.8,
        rt: 10.0,
        rt_library: 10.0,
        naa: 9,
        charge: 0,
        isotope_intensity: Vec::new(),
        fragment_mz: vec![FRAGMENT_MZ],
        fragment_mz_library: vec![FRAGMENT_MZ],
        fragment_intensity: vec![1.0],
        fragment_cardinality: vec![1],
        fragment_charge: vec![1],
        fragment_loss_type: vec![0],
        fragment_number: vec![3],
        fragment_position: vec![3],
        fragment_type: vec![121],
    };

    let plan = ExtractionPlan::new(&dia_data, &precursor);

    assert_eq!(plan.observation_indices, vec![0]);
    assert_eq!(plan.channels.len(), 1);
}

#[test]
fn test_envelope_inside_one_window_keeps_monoisotopic_gain_one() {
    // Given: M0..M3 of a 2+ precursor at 498.6..500.1 lie almost fully in [498, 500]
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);

    let plan = plan_at(&dia_data, 498.6);

    // Then: the lower window's monoisotopic channel carries the reference signal
    let mono = plan
        .channels
        .iter()
        .find(|channel| channel.observation_idx == 0 && channel.mz_shift == 0.0)
        .unwrap();
    assert!(mono.gain > 0.95, "gain {}", mono.gain);
    assert!((sum_weight_times_gain(&plan, 0) - 1.0).abs() < 1e-5);
}

#[test]
fn test_envelope_across_windows_uses_both() {
    // Given: M0 at 499.8 in the lower window; M1 (500.30), M2 (500.80) in the upper window
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);

    let plan = plan_at(&dia_data, 499.8);

    // Then: both windows are extracted, the upper one also at the fragment's first isotope
    assert_eq!(plan.observation_indices, vec![0, 1]);
    assert!(plan
        .channels
        .iter()
        .any(|channel| channel.observation_idx == 1 && channel.mz_shift == 0.0));
    let upper_isotope = plan
        .channels
        .iter()
        .find(|channel| channel.observation_idx == 1 && channel.mz_shift > 0.0)
        .unwrap();
    assert!((upper_isotope.mz - (FRAGMENT_MZ + ISOTOPE_SPACING)).abs() < 1e-4);
    assert!((sum_weight_times_gain(&plan, 0) - 1.0).abs() < 1e-5);
}

#[test]
fn test_gain_follows_isotope_and_fragment_share() {
    // Given: a 1+ precursor with M0 (499.5) in the lower window and M1+ (500.5, ...) in the upper one
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);
    let precursor_mz = 499.5;
    let share = (FRAGMENT_MZ - PROTON_MASS) / (precursor_mz - PROTON_MASS);
    let reference: f32 = ISOTOPES
        .iter()
        .enumerate()
        .map(|(k, &abundance)| abundance * (1.0 - share).powi(k as i32))
        .sum();

    let plan = ExtractionPlan::isotope_envelope(
        &dia_data,
        precursor_mz,
        1,
        &ISOTOPES,
        &[FRAGMENT_MZ],
        &[1],
        1,
    );

    // Then: the lower window holds M0 only, so its monoisotopic gain is I_0 / reference
    let lower_mono = plan
        .channels
        .iter()
        .find(|channel| channel.observation_idx == 0 && channel.mz_shift == 0.0)
        .unwrap();
    assert!((lower_mono.gain - ISOTOPES[0] / reference).abs() < 1e-2);
    // and no fragment isotope channel: M0 fragments carry no extra neutron
    assert!(!plan
        .channels
        .iter()
        .any(|channel| channel.observation_idx == 0 && channel.mz_shift > 0.0));
    // The upper window's monoisotopic gain is sum_k>=1 I_k (1-f)^k / reference
    let upper_mono_expected: f32 = ISOTOPES
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &abundance)| abundance * (1.0 - share).powi(k as i32))
        .sum::<f32>()
        / reference;
    let upper_mono = plan
        .channels
        .iter()
        .find(|channel| channel.observation_idx == 1 && channel.mz_shift == 0.0)
        .unwrap();
    assert!((upper_mono.gain - upper_mono_expected).abs() < 1e-2);
}

#[test]
fn test_dense_xic_estimates_reference_signal_from_split_envelope() {
    // Given: a fragment signal s split across windows and fragment isotopes as the plan expects
    let precursor_mz = 499.8;
    let signal = 1000.0;
    let layout = two_window_data([0.0, 0.0], [0.0, 0.0]);
    let plan = plan_at(&layout, precursor_mz);
    let expected = |observation_idx: usize, isotope: bool| -> f32 {
        plan.channels
            .iter()
            .filter(|channel| {
                channel.observation_idx == observation_idx && (channel.mz_shift > 0.0) == isotope
            })
            .map(|channel| channel.gain * signal)
            .sum()
    };
    let dia_data = two_window_data(
        [expected(0, false), expected(0, true)],
        [expected(1, false), expected(1, true)],
    );

    // When
    let dense = DenseXICObservation::new(&dia_data, &plan, 0, 1, 20.0);

    // Then: the XIC holds the reference (fully isolated, monoisotopic) signal
    assert!(
        (dense.dense_xic[[0, 0]] - signal).abs() < 1.0,
        "{}",
        dense.dense_xic[[0, 0]]
    );
}

#[test]
fn test_dense_mz_reports_monoisotopic_mz_for_isotope_channels() {
    // Given: only the upper window's fragment isotope has signal
    let dia_data = two_window_data([0.0, 0.0], [0.0, 500.0]);
    let plan = plan_at(&dia_data, 499.8);

    // When
    let dense = DenseXICMZObservation::new(&dia_data, &plan, 0, 1, 20.0);

    // Then: its m/z is reported at the monoisotopic fragment
    assert!(dense.dense_xic[[0, 0]] > 0.0);
    let mz_error_ppm = (dense.dense_mz[[0, 0]] - FRAGMENT_MZ) / FRAGMENT_MZ * 1e6;
    assert!(mz_error_ppm.abs() < 5.0, "{mz_error_ppm} ppm");
}

#[test]
fn test_missing_isotope_values_fall_back_to_monoisotopic_plan() {
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);
    for isotopes in [vec![f32::NAN, 0.3], vec![0.0, 0.0]] {
        let precursor = Precursor {
            precursor_idx: 0,
            mz: 499.8,
            mz_library: 499.8,
            rt: 10.0,
            rt_library: 10.0,
            naa: 9,
            charge: 2,
            isotope_intensity: isotopes,
            fragment_mz: vec![FRAGMENT_MZ],
            fragment_mz_library: vec![FRAGMENT_MZ],
            fragment_intensity: vec![1.0],
            fragment_cardinality: vec![1],
            fragment_charge: vec![1],
            fragment_loss_type: vec![0],
            fragment_number: vec![3],
            fragment_position: vec![3],
            fragment_type: vec![121],
        };

        let plan = ExtractionPlan::new(&dia_data, &precursor);

        assert_eq!(plan.channels.len(), 1);
        assert_eq!(plan.channels[0].weight, 1.0);
    }
}

#[test]
fn test_monoisotopic_fragments_only_without_fragment_isotopes() {
    let dia_data = two_window_data([0.0, 0.0], [0.0, 0.0]);

    let plan =
        ExtractionPlan::isotope_envelope(&dia_data, 499.8, 2, &ISOTOPES, &[FRAGMENT_MZ], &[1], 0);

    assert_eq!(plan.observation_indices, vec![0, 1]);
    assert!(plan.channels.iter().all(|channel| channel.mz_shift == 0.0));
    assert!((sum_weight_times_gain(&plan, 0) - 1.0).abs() < 1e-5);
}
