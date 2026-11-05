use super::*;
use pyo3::types::PyDict;

const NUM_CALIBRATION_FEATURES: usize = 20;
const NUM_FULL_FEATURES: usize = 37;

/// Core fields hold their 1-based column position, so that a column at the wrong position shows its own value.
fn numbered_core() -> CoreFeatures {
    CoreFeatures {
        score: 1.0,
        mean_correlation: 2.0,
        median_correlation: 3.0,
        correlation_std: 4.0,
        num_over_80: 5.0,
        num_over_0_rank_0_5: 6.0,
        hyperscore_intensity_observation: 7.0,
        hyperscore_intensity_library: 8.0,
        hyperscore_inverse_mass_error: 9.0,
        rt_observed: 10.0,
        delta_rt: 11.0,
        longest_b_series: 12.0,
        longest_y_series: 13.0,
        naa: 14.0,
        weighted_mass_error: 15.0,
        log10_b_ion_intensity: 16.0,
        log10_y_ion_intensity: 17.0,
        fwhm_rt: 18.0,
        median_profile_sum: 19.0,
        num_profiles_filtered: 20.0,
    }
}

/// A calibration feature whose fields hold their 1-based column position
fn numbered_calibration_feature() -> CandidateFeature {
    CandidateFeature {
        precursor_idx: 5,
        rank: 2,
        core: numbered_core(),
        ion_coverage: None,
        library_agreement: None,
    }
}

/// A full feature whose fields hold their 1-based column position
fn numbered_full_feature() -> CandidateFeature {
    CandidateFeature {
        precursor_idx: 5,
        rank: 2,
        core: numbered_core(),
        ion_coverage: Some(IonCoverageFeatures {
            ion_position_mean: 21.0,
            short_ion_intensity_share: 22.0,
            num_over_50_b_short: 23.0,
            num_over_50_b_long: 24.0,
            num_over_50_y_short: 25.0,
            num_over_50_y_mid: 26.0,
            num_over_50_y_long: 27.0,
            num_over_50_z2: 28.0,
            lib_share_over_50: 29.0,
            xic_libweighted: 30.0,
            xic_obsweighted: 31.0,
            ion_fisher_lib: 32.0,
        }),
        library_agreement: Some(LibraryAgreementFeatures {
            log_ratio_std: 33.0,
            y_intensity_correlation: 34.0,
            lib_cosine_over_50: 35.0,
            ent_sim: 36.0,
            ent_sim_corrweighted: 37.0,
        }),
    }
}

fn assert_unique_without_candidate_keys(names: &[&str]) {
    let mut unique = names.to_vec();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), names.len());
    assert!(!names.contains(&"precursor_idx"));
    assert!(!names.contains(&"rank"));
}

/// Checks that the dict of a one-feature collection holds the keys and every feature column at its position.
fn assert_every_column_at_its_position(
    collection: &CandidateFeatureCollection,
    num_features: usize,
) {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let obj = collection.to_dict_arrays(py).unwrap();
        let dict = obj.cast_bound::<PyDict>(py).unwrap();

        let precursor_idx: Vec<u64> = dict
            .get_item("precursor_idx")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rank: Vec<u64> = dict.get_item("rank").unwrap().unwrap().extract().unwrap();
        assert_eq!(precursor_idx, vec![5]);
        assert_eq!(rank, vec![2]);
        assert_eq!(dict.len(), num_features + 2);
        let names = collection.feature_set().feature_names();
        for (position, name) in names.iter().enumerate() {
            let column: Vec<f32> = dict.get_item(*name).unwrap().unwrap().extract().unwrap();
            assert_eq!(column, vec![(position + 1) as f32], "{name}");
        }
    });
}

#[test]
fn test_calibration_feature_names_are_the_core_block() {
    // given / when
    let names = FeatureSet::Calibration.feature_names();

    // then
    assert_eq!(names, CoreFeatures::NAMES);
    assert_eq!(names.len(), NUM_CALIBRATION_FEATURES);
    assert_unique_without_candidate_keys(&names);
}

#[test]
fn test_full_feature_names_are_core_ion_coverage_and_library_agreement() {
    // given / when
    let names = FeatureSet::Full.feature_names();

    // then
    assert_eq!(
        names,
        [
            CoreFeatures::NAMES,
            IonCoverageFeatures::NAMES,
            LibraryAgreementFeatures::NAMES
        ]
        .concat()
    );
    assert_eq!(names.len(), NUM_FULL_FEATURES);
    assert_unique_without_candidate_keys(&names);
}

#[test]
fn test_calibration_leaves_out_the_ion_coverage_and_library_agreement_features() {
    // given / when
    let names = FeatureSet::Calibration.feature_names();

    // then
    for name in [IonCoverageFeatures::NAMES, LibraryAgreementFeatures::NAMES].concat() {
        assert!(!names.contains(&name), "{name}");
    }
}

#[test]
fn test_block_column_names_are_the_field_names() {
    // given
    let feature = numbered_full_feature();

    // when
    let names = FeatureSet::Full.feature_names();
    let values = feature.values(FeatureSet::Full).unwrap();

    // then
    let value_of = |name: &str| values[names.iter().position(|n| *n == name).unwrap()];
    assert_eq!(value_of("fwhm_rt"), feature.core.fwhm_rt);
    assert_eq!(
        value_of("ion_fisher_lib"),
        feature.ion_coverage.unwrap().ion_fisher_lib
    );
    assert_eq!(
        value_of("ent_sim_corrweighted"),
        feature.library_agreement.unwrap().ent_sim_corrweighted
    );
}

#[test]
fn test_calibration_to_dict_arrays_puts_every_value_in_its_column() {
    // given
    let collection = CandidateFeatureCollection::from_vec(
        FeatureSet::Calibration,
        vec![numbered_calibration_feature()],
    );

    // when / then
    assert_every_column_at_its_position(&collection, NUM_CALIBRATION_FEATURES);
}

#[test]
fn test_full_to_dict_arrays_puts_every_value_in_its_column() {
    // given
    let collection =
        CandidateFeatureCollection::from_vec(FeatureSet::Full, vec![numbered_full_feature()]);

    // when / then
    assert_every_column_at_its_position(&collection, NUM_FULL_FEATURES);
}

#[test]
fn test_calibration_values_leave_out_the_full_blocks() {
    // given
    let feature = numbered_full_feature();

    // when
    let values = feature.values(FeatureSet::Calibration).unwrap();

    // then
    assert_eq!(values, feature.core.values());
}

#[test]
fn test_full_values_are_none_if_a_block_is_missing() {
    // given
    let calibration_feature = numbered_calibration_feature();

    // when
    let values = calibration_feature.values(FeatureSet::Full);

    // then
    assert_eq!(values, None);
}

#[test]
fn test_to_dict_arrays_rejects_a_candidate_missing_a_block_of_the_set() {
    pyo3::Python::initialize();

    // given
    let collection = CandidateFeatureCollection::from_vec(
        FeatureSet::Full,
        vec![numbered_calibration_feature()],
    );

    Python::attach(|py| {
        // when
        let result = collection.to_dict_arrays(py);

        // then
        assert!(result.is_err());
    });
}
