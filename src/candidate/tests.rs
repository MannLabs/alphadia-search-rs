use super::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_get_feature_names() {
    let feature_names = CandidateFeatureCollection::get_feature_names();

    // Verify we have the expected number of f32 features (23 base + 8 ranked + fwhm_rt)
    assert_eq!(feature_names.len(), 41);

    // Verify some key feature names are present
    assert!(feature_names.contains(&"score".to_string()));
    assert!(feature_names.contains(&"mean_correlation".to_string()));
    assert!(feature_names.contains(&"median_correlation".to_string()));
    assert!(feature_names.contains(&"correlation_std".to_string()));
    assert!(feature_names.contains(&"intensity_correlation".to_string()));
    assert!(feature_names.contains(&"num_fragments".to_string()));
    assert!(feature_names.contains(&"num_scans".to_string()));
    assert!(feature_names.contains(&"num_over_0".to_string()));
    assert!(feature_names.contains(&"rt_observed".to_string()));
    assert!(feature_names.contains(&"delta_rt".to_string()));
    assert!(feature_names.contains(&"longest_b_series".to_string()));
    assert!(feature_names.contains(&"longest_y_series".to_string()));
    assert!(feature_names.contains(&"naa".to_string()));
    assert!(feature_names.contains(&"hyperscore_inverse_mass_error".to_string()));
    assert!(feature_names.contains(&"weighted_mass_error".to_string()));
    assert!(feature_names.contains(&"log10_b_ion_intensity".to_string()));
    assert!(feature_names.contains(&"log10_y_ion_intensity".to_string()));
    assert!(feature_names.contains(&"fwhm_rt".to_string()));

    // Verify that non-f32 columns are NOT included
    assert!(!feature_names.contains(&"precursor_idx".to_string()));
    assert!(!feature_names.contains(&"rank".to_string()));

    // Verify all names are unique
    let mut sorted_names = feature_names.clone();
    sorted_names.sort();
    sorted_names.dedup();
    assert_eq!(sorted_names.len(), feature_names.len());
}

#[test]
fn test_candidate_collection_to_arrays_dtypes_and_order() {
    pyo3::prepare_freethreaded_python();
    let candidate = Candidate {
        precursor_idx: 42,
        rank: 7,
        score: 1.5,
        scan_center: 101,
        scan_start: 100,
        scan_stop: 102,
        cycle_center: 201,
        cycle_start: 200,
        cycle_stop: 202,
    };
    let collection = CandidateCollection::from_vec(vec![candidate]);

    Python::with_gil(|py| {
        let (
            precursor_idxs,
            ranks,
            scores,
            scan_center,
            scan_start,
            scan_stop,
            cycle_center,
            cycle_start,
            cycle_stop,
        ) = collection.to_arrays(py).expect("to_arrays should succeed");

        let precursor_idxs: Vec<u64> = precursor_idxs.extract(py).unwrap();
        let ranks: Vec<u64> = ranks.extract(py).unwrap();
        let scores: Vec<f32> = scores.extract(py).unwrap();
        let scan_center: Vec<u64> = scan_center.extract(py).unwrap();
        let scan_start: Vec<u64> = scan_start.extract(py).unwrap();
        let scan_stop: Vec<u64> = scan_stop.extract(py).unwrap();
        let cycle_center: Vec<u64> = cycle_center.extract(py).unwrap();
        let cycle_start: Vec<u64> = cycle_start.extract(py).unwrap();
        let cycle_stop: Vec<u64> = cycle_stop.extract(py).unwrap();

        assert_eq!(precursor_idxs.len(), 1);
        assert_eq!(ranks.len(), 1);
        assert_eq!(scores.len(), 1);
        assert_eq!(scan_center.len(), 1);
        assert_eq!(scan_start.len(), 1);
        assert_eq!(scan_stop.len(), 1);
        assert_eq!(cycle_center.len(), 1);
        assert_eq!(cycle_start.len(), 1);
        assert_eq!(cycle_stop.len(), 1);

        assert_eq!(precursor_idxs[0], 42u64);
        assert_eq!(ranks[0], 7u64);
        assert!((scores[0] - 1.5).abs() < 1e-6);
        assert_eq!(scan_center[0], 101u64);
        assert_eq!(scan_start[0], 100u64);
        assert_eq!(scan_stop[0], 102u64);
        assert_eq!(cycle_center[0], 201u64);
        assert_eq!(cycle_start[0], 200u64);
        assert_eq!(cycle_stop[0], 202u64);
    });
}

#[test]
fn test_candidate_collection_from_arrays_roundtrip() {
    pyo3::prepare_freethreaded_python();
    let precursor_idxs = vec![1u64, 2u64];
    let ranks = vec![10u64, 20u64];
    let scores = vec![0.1f32, 0.2f32];
    let scan_center = vec![11u64, 21u64];
    let scan_start = vec![12u64, 22u64];
    let scan_stop = vec![13u64, 23u64];
    let cycle_center = vec![14u64, 24u64];
    let cycle_start = vec![15u64, 25u64];
    let cycle_stop = vec![16u64, 26u64];

    let collection = CandidateCollection::from_arrays(
        precursor_idxs.clone(),
        ranks.clone(),
        scores.clone(),
        scan_center.clone(),
        scan_start.clone(),
        scan_stop.clone(),
        cycle_center.clone(),
        cycle_start.clone(),
        cycle_stop.clone(),
    )
    .expect("from_arrays should succeed");

    assert_eq!(collection.len(), 2);
    assert!(!collection.is_empty());

    Python::with_gil(|py| {
        let (p, r, s, sc, ss, so, cc, cs, co) = collection.to_arrays(py).unwrap();

        let p: Vec<u64> = p.extract(py).unwrap();
        let r: Vec<u64> = r.extract(py).unwrap();
        let s: Vec<f32> = s.extract(py).unwrap();
        let sc: Vec<u64> = sc.extract(py).unwrap();
        let ss: Vec<u64> = ss.extract(py).unwrap();
        let so: Vec<u64> = so.extract(py).unwrap();
        let cc: Vec<u64> = cc.extract(py).unwrap();
        let cs: Vec<u64> = cs.extract(py).unwrap();
        let co: Vec<u64> = co.extract(py).unwrap();

        assert_eq!(p, precursor_idxs);
        assert_eq!(r, ranks);
        assert_eq!(s, scores);
        assert_eq!(sc, scan_center);
        assert_eq!(ss, scan_start);
        assert_eq!(so, scan_stop);
        assert_eq!(cc, cycle_center);
        assert_eq!(cs, cycle_start);
        assert_eq!(co, cycle_stop);
    });
}

#[test]
fn test_candidate_feature_collection_to_dict_arrays_dtypes_and_values() {
    pyo3::prepare_freethreaded_python();

    let feature = CandidateFeature::new(
        5,     // precursor_idx
        2,     // rank
        0.95,  // score
        0.8,   // mean_correlation
        0.75,  // median_correlation
        0.05,  // correlation_std
        0.9,   // intensity_correlation
        12.0,  // num_fragments
        30.0,  // num_scans
        3.0,   // num_over_95
        5.0,   // num_over_90
        8.0,   // num_over_80
        15.0,  // num_over_50
        20.0,  // num_over_0
        5.0,   // num_over_0_rank_0_5
        6.0,   // num_over_0_rank_6_11
        5.0,   // num_over_0_rank_12_17
        4.0,   // num_over_0_rank_18_23
        3.0,   // num_over_50_rank_0_5
        4.0,   // num_over_50_rank_6_11
        4.0,   // num_over_50_rank_12_17
        4.0,   // num_over_50_rank_18_23
        100.0, // hyperscore_intensity_observation
        120.0, // hyperscore_intensity_library
        80.0,  // hyperscore_inverse_mass_error
        123.4, // rt_observed
        -2.5,  // delta_rt
        4.0,   // longest_b_series
        6.0,   // longest_y_series
        10.0,  // naa
        2.5,   // weighted_mass_error
        3.2,   // log10_b_ion_intensity
        3.8,   // log10_y_ion_intensity
        15.5,  // fwhm_rt
        50.0,  // idf_hyperscore
        10.5,  // idf_xic_dot_product
        8.2,   // idf_intensity_dot_product
        100.0, // median_profile_sum
        95.0,  // median_profile_sum_filtered
        12.0,  // num_profiles
        11.0,  // num_profiles_filtered
        5.0,   // num_over_0_top6_idf
        3.0,   // num_over_50_top6_idf
    );
    let collection = CandidateFeatureCollection::from_vec(vec![feature]);

    Python::with_gil(|py| {
        let obj = collection
            .to_dict_arrays(py)
            .expect("to_dict_arrays should succeed");
        let dict = obj.downcast_bound::<PyDict>(py).unwrap();

        // u64-typed
        let precursor_idx: Vec<u64> = dict
            .get_item("precursor_idx")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rank: Vec<u64> = dict.get_item("rank").unwrap().unwrap().extract().unwrap();

        // f32-typed
        let score: Vec<f32> = dict.get_item("score").unwrap().unwrap().extract().unwrap();
        let mean_corr: Vec<f32> = dict
            .get_item("mean_correlation")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let median_corr: Vec<f32> = dict
            .get_item("median_correlation")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let corr_std: Vec<f32> = dict
            .get_item("correlation_std")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let intensity_corr: Vec<f32> = dict
            .get_item("intensity_correlation")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_fragments: Vec<f32> = dict
            .get_item("num_fragments")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_scans: Vec<f32> = dict
            .get_item("num_scans")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_over_95: Vec<f32> = dict
            .get_item("num_over_95")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_over_90: Vec<f32> = dict
            .get_item("num_over_90")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_over_80: Vec<f32> = dict
            .get_item("num_over_80")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_over_50: Vec<f32> = dict
            .get_item("num_over_50")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let num_over_0: Vec<f32> = dict
            .get_item("num_over_0")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let hypo_obs: Vec<f32> = dict
            .get_item("hyperscore_intensity_observation")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let hypo_lib: Vec<f32> = dict
            .get_item("hyperscore_intensity_library")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rt_observed: Vec<f32> = dict
            .get_item("rt_observed")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let delta_rt: Vec<f32> = dict
            .get_item("delta_rt")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let longest_b_series: Vec<f32> = dict
            .get_item("longest_b_series")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let longest_y_series: Vec<f32> = dict
            .get_item("longest_y_series")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let naa: Vec<f32> = dict.get_item("naa").unwrap().unwrap().extract().unwrap();
        let hypo_inv_error: Vec<f32> = dict
            .get_item("hyperscore_inverse_mass_error")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let weighted_mass_error: Vec<f32> = dict
            .get_item("weighted_mass_error")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let log10_b_ion: Vec<f32> = dict
            .get_item("log10_b_ion_intensity")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let log10_y_ion: Vec<f32> = dict
            .get_item("log10_y_ion_intensity")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let fwhm_rt: Vec<f32> = dict
            .get_item("fwhm_rt")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(precursor_idx, vec![5u64]);
        assert_eq!(rank, vec![2u64]);
        assert!((score[0] - 0.95).abs() < 1e-6);
        assert!((mean_corr[0] - 0.8).abs() < 1e-6);
        assert!((median_corr[0] - 0.75).abs() < 1e-6);
        assert!((corr_std[0] - 0.05).abs() < 1e-6);
        assert!((intensity_corr[0] - 0.9).abs() < 1e-6);
        assert!((num_fragments[0] - 12.0).abs() < 1e-6);
        assert!((num_scans[0] - 30.0).abs() < 1e-6);
        assert!((num_over_95[0] - 3.0).abs() < 1e-6);
        assert!((num_over_90[0] - 5.0).abs() < 1e-6);
        assert!((num_over_80[0] - 8.0).abs() < 1e-6);
        assert!((num_over_50[0] - 15.0).abs() < 1e-6);
        assert!((num_over_0[0] - 20.0).abs() < 1e-6);
        assert!((hypo_obs[0] - 100.0).abs() < 1e-6);
        assert!((hypo_lib[0] - 120.0).abs() < 1e-6);
        assert!((hypo_inv_error[0] - 80.0).abs() < 1e-6);
        assert!((rt_observed[0] - 123.4).abs() < 1e-6);
        assert!((delta_rt[0] + 2.5).abs() < 1e-6);
        assert!((longest_b_series[0] - 4.0).abs() < 1e-6);
        assert!((longest_y_series[0] - 6.0).abs() < 1e-6);
        assert!((naa[0] - 10.0).abs() < 1e-6);
        assert!((weighted_mass_error[0] - 2.5).abs() < 1e-6);
        assert!((log10_b_ion[0] - 3.2).abs() < 1e-6);
        assert!((log10_y_ion[0] - 3.8).abs() < 1e-6);
        assert!((fwhm_rt[0] - 15.5).abs() < 1e-6);
    });
}
