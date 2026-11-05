use super::*;
use pyo3::types::PyDict;

const NUM_FEATURES: usize = 53;

/// Every field holds its 1-based column position, so that a column at the wrong position shows its own value.
fn numbered_feature() -> CandidateFeature {
    CandidateFeature {
        precursor_idx: 5,
        rank: 2,
        core: CoreFeatures {
            score: 1.0,
            mean_correlation: 2.0,
            median_correlation: 3.0,
            correlation_std: 4.0,
            intensity_correlation: 5.0,
            num_fragments: 6.0,
            num_scans: 7.0,
            num_over_95: 8.0,
            num_over_90: 9.0,
            num_over_80: 10.0,
            num_over_50: 11.0,
            num_over_0: 12.0,
            num_over_0_rank_0_5: 13.0,
            num_over_0_rank_6_11: 14.0,
            num_over_0_rank_12_17: 15.0,
            num_over_0_rank_18_23: 16.0,
            num_over_50_rank_0_5: 17.0,
            num_over_50_rank_6_11: 18.0,
            num_over_50_rank_12_17: 19.0,
            num_over_50_rank_18_23: 20.0,
            hyperscore_intensity_observation: 21.0,
            hyperscore_intensity_library: 22.0,
            hyperscore_inverse_mass_error: 23.0,
            rt_observed: 24.0,
            delta_rt: 25.0,
            longest_b_series: 26.0,
            longest_y_series: 27.0,
            naa: 28.0,
            weighted_mass_error: 29.0,
            log10_b_ion_intensity: 30.0,
            log10_y_ion_intensity: 31.0,
            fwhm_rt: 32.0,
            median_profile_sum: 33.0,
            median_profile_sum_filtered: 34.0,
            num_profiles: 35.0,
            num_profiles_filtered: 36.0,
        },
        ion_coverage: IonCoverageFeatures {
            ion_position_mean: 37.0,
            short_ion_intensity_share: 38.0,
            num_over_50_b_short: 39.0,
            num_over_50_b_long: 40.0,
            num_over_50_y_short: 41.0,
            num_over_50_y_mid: 42.0,
            num_over_50_y_long: 43.0,
            num_over_50_z2: 44.0,
            lib_share_over_50: 45.0,
            xic_libweighted: 46.0,
            xic_obsweighted: 47.0,
            ion_fisher_lib: 48.0,
        },
        library_agreement: LibraryAgreementFeatures {
            log_ratio_std: 49.0,
            y_intensity_correlation: 50.0,
            lib_cosine_over_50: 51.0,
            ent_sim: 52.0,
            ent_sim_corrweighted: 53.0,
        },
    }
}

#[test]
fn test_feature_names_are_the_blocks_in_column_order() {
    // given / when
    let names = CandidateFeature::feature_names();

    // then
    let expected = [
        CoreFeatures::NAMES,
        IonCoverageFeatures::NAMES,
        LibraryAgreementFeatures::NAMES,
    ]
    .concat();
    assert_eq!(names, expected);
    assert_eq!(names.len(), NUM_FEATURES);
}

#[test]
fn test_feature_names_are_unique_and_leave_out_the_candidate_keys() {
    // given / when
    let names = CandidateFeature::feature_names();

    // then
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), names.len());
    assert!(!names.contains(&"precursor_idx"));
    assert!(!names.contains(&"rank"));
}

#[test]
fn test_block_column_names_are_the_field_names() {
    // given
    let feature = numbered_feature();

    // when
    let names = CandidateFeature::feature_names();
    let values = feature.values();

    // then
    let value_of = |name: &str| values[names.iter().position(|n| *n == name).unwrap()];
    assert_eq!(value_of("fwhm_rt"), feature.core.fwhm_rt);
    assert_eq!(
        value_of("ion_fisher_lib"),
        feature.ion_coverage.ion_fisher_lib
    );
    assert_eq!(
        value_of("ent_sim_corrweighted"),
        feature.library_agreement.ent_sim_corrweighted
    );
}

#[test]
fn test_to_dict_arrays_puts_every_value_in_its_column() {
    pyo3::Python::initialize();

    // given
    let collection = CandidateFeatureCollection::from_vec(vec![numbered_feature()]);

    Python::attach(|py| {
        // when
        let obj = collection.to_dict_arrays(py).unwrap();
        let dict = obj.cast_bound::<PyDict>(py).unwrap();

        // then
        let precursor_idx: Vec<u64> = dict
            .get_item("precursor_idx")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rank: Vec<u64> = dict.get_item("rank").unwrap().unwrap().extract().unwrap();
        assert_eq!(precursor_idx, vec![5]);
        assert_eq!(rank, vec![2]);
        assert_eq!(dict.len(), NUM_FEATURES + 2);
        for (position, name) in CandidateFeature::feature_names().iter().enumerate() {
            let column: Vec<f32> = dict.get_item(*name).unwrap().unwrap().extract().unwrap();
            assert_eq!(column, vec![(position + 1) as f32], "{name}");
        }
    });
}
