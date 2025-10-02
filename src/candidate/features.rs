use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

// Single source of truth for feature names (f32 features)
pub const FEATURE_NAMES: &[&str] = &[
    "score",
    "mean_correlation",
    "median_correlation",
    "correlation_std",
    "intensity_correlation",
    "num_fragments",
    "num_scans",
    "num_over_95",
    "num_over_90",
    "num_over_80",
    "num_over_50",
    "num_over_0",
    "num_over_0_rank_0_5",
    "num_over_0_rank_6_11",
    "num_over_0_rank_12_17",
    "num_over_0_rank_18_23",
    "num_over_50_rank_0_5",
    "num_over_50_rank_6_11",
    "num_over_50_rank_12_17",
    "num_over_50_rank_18_23",
    "hyperscore_intensity_observation",
    "hyperscore_intensity_library",
    "hyperscore_inverse_mass_error",
    "rt_observed",
    "delta_rt",
    "longest_b_series",
    "longest_y_series",
    "naa",
    "weighted_mass_error",
    "log10_b_ion_intensity",
    "log10_y_ion_intensity",
    "fwhm_rt",
    "idf_hyperscore",
    "idf_xic_dot_product",
    "idf_intensity_dot_product",
    "median_profile_sum",
    "median_profile_sum_filtered",
    "num_profiles",
    "num_profiles_filtered",
    "num_over_0_top6_idf",
    "num_over_50_top6_idf",
];

#[derive(Debug, Clone)]
pub struct CandidateFeature {
    pub precursor_idx: usize,
    pub rank: usize,
    pub score: f32,
    pub mean_correlation: f32,
    pub median_correlation: f32,
    pub correlation_std: f32,
    pub intensity_correlation: f32,
    pub num_fragments: f32,
    pub num_scans: f32,
    pub num_over_95: f32,
    pub num_over_90: f32,
    pub num_over_80: f32,
    pub num_over_50: f32,
    pub num_over_0: f32,
    pub num_over_0_rank_0_5: f32,
    pub num_over_0_rank_6_11: f32,
    pub num_over_0_rank_12_17: f32,
    pub num_over_0_rank_18_23: f32,
    pub num_over_50_rank_0_5: f32,
    pub num_over_50_rank_6_11: f32,
    pub num_over_50_rank_12_17: f32,
    pub num_over_50_rank_18_23: f32,
    pub hyperscore_intensity_observation: f32,
    pub hyperscore_intensity_library: f32,
    pub hyperscore_inverse_mass_error: f32,
    pub rt_observed: f32,
    pub delta_rt: f32,
    pub longest_b_series: f32,
    pub longest_y_series: f32,
    pub naa: f32,
    pub weighted_mass_error: f32,
    pub log10_b_ion_intensity: f32,
    pub log10_y_ion_intensity: f32,
    pub fwhm_rt: f32,
    pub idf_hyperscore: f32,
    pub idf_xic_dot_product: f32,
    pub idf_intensity_dot_product: f32,
    pub median_profile_sum: f32,
    pub median_profile_sum_filtered: f32,
    pub num_profiles: f32,
    pub num_profiles_filtered: f32,
    pub num_over_0_top6_idf: f32,
    pub num_over_50_top6_idf: f32,
}

impl CandidateFeature {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        precursor_idx: usize,
        rank: usize,
        score: f32,
        mean_correlation: f32,
        median_correlation: f32,
        correlation_std: f32,
        intensity_correlation: f32,
        num_fragments: f32,
        num_scans: f32,
        num_over_95: f32,
        num_over_90: f32,
        num_over_80: f32,
        num_over_50: f32,
        num_over_0: f32,
        num_over_0_rank_0_5: f32,
        num_over_0_rank_6_11: f32,
        num_over_0_rank_12_17: f32,
        num_over_0_rank_18_23: f32,
        num_over_50_rank_0_5: f32,
        num_over_50_rank_6_11: f32,
        num_over_50_rank_12_17: f32,
        num_over_50_rank_18_23: f32,
        hyperscore_intensity_observation: f32,
        hyperscore_intensity_library: f32,
        hyperscore_inverse_mass_error: f32,
        rt_observed: f32,
        delta_rt: f32,
        longest_b_series: f32,
        longest_y_series: f32,
        naa: f32,
        weighted_mass_error: f32,
        log10_b_ion_intensity: f32,
        log10_y_ion_intensity: f32,
        fwhm_rt: f32,
        idf_hyperscore: f32,
        idf_xic_dot_product: f32,
        idf_intensity_dot_product: f32,
        median_profile_sum: f32,
        median_profile_sum_filtered: f32,
        num_profiles: f32,
        num_profiles_filtered: f32,
        num_over_0_top6_idf: f32,
        num_over_50_top6_idf: f32,
    ) -> Self {
        Self {
            precursor_idx,
            rank,
            score,
            mean_correlation,
            median_correlation,
            correlation_std,
            intensity_correlation,
            num_fragments,
            num_scans,
            num_over_95,
            num_over_90,
            num_over_80,
            num_over_50,
            num_over_0,
            num_over_0_rank_0_5,
            num_over_0_rank_6_11,
            num_over_0_rank_12_17,
            num_over_0_rank_18_23,
            num_over_50_rank_0_5,
            num_over_50_rank_6_11,
            num_over_50_rank_12_17,
            num_over_50_rank_18_23,
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            hyperscore_inverse_mass_error,
            rt_observed,
            delta_rt,
            longest_b_series,
            longest_y_series,
            naa,
            weighted_mass_error,
            log10_b_ion_intensity,
            log10_y_ion_intensity,
            fwhm_rt,
            idf_hyperscore,
            idf_xic_dot_product,
            idf_intensity_dot_product,
            median_profile_sum,
            median_profile_sum_filtered,
            num_profiles,
            num_profiles_filtered,
            num_over_0_top6_idf,
            num_over_50_top6_idf,
        }
    }
}

#[pyclass]
pub struct CandidateFeatureCollection {
    features: Vec<CandidateFeature>,
}

impl Default for CandidateFeatureCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl CandidateFeatureCollection {
    #[new]
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn to_dict_arrays(&self, py: Python) -> PyResult<PyObject> {
        let n = self.features.len();

        let mut precursor_idxs = Array1::<u64>::zeros(n);
        let mut ranks = Array1::<u64>::zeros(n);
        let mut scores = Array1::<f32>::zeros(n);
        let mut mean_correlations = Array1::<f32>::zeros(n);
        let mut median_correlations = Array1::<f32>::zeros(n);
        let mut correlation_stds = Array1::<f32>::zeros(n);
        let mut intensity_correlations = Array1::<f32>::zeros(n);
        let mut num_fragments = Array1::<f32>::zeros(n);
        let mut num_scans = Array1::<f32>::zeros(n);
        let mut num_over_95 = Array1::<f32>::zeros(n);
        let mut num_over_90 = Array1::<f32>::zeros(n);
        let mut num_over_80 = Array1::<f32>::zeros(n);
        let mut num_over_50 = Array1::<f32>::zeros(n);
        let mut num_over_0 = Array1::<f32>::zeros(n);
        let mut num_over_0_rank_0_5 = Array1::<f32>::zeros(n);
        let mut num_over_0_rank_6_11 = Array1::<f32>::zeros(n);
        let mut num_over_0_rank_12_17 = Array1::<f32>::zeros(n);
        let mut num_over_0_rank_18_23 = Array1::<f32>::zeros(n);
        let mut num_over_50_rank_0_5 = Array1::<f32>::zeros(n);
        let mut num_over_50_rank_6_11 = Array1::<f32>::zeros(n);
        let mut num_over_50_rank_12_17 = Array1::<f32>::zeros(n);
        let mut num_over_50_rank_18_23 = Array1::<f32>::zeros(n);
        let mut hyperscore_intensity_observations = Array1::<f32>::zeros(n);
        let mut hyperscore_intensity_libraries = Array1::<f32>::zeros(n);
        let mut hyperscore_inverse_mass_errors = Array1::<f32>::zeros(n);
        let mut rt_observeds = Array1::<f32>::zeros(n);
        let mut delta_rts = Array1::<f32>::zeros(n);
        let mut longest_b_series = Array1::<f32>::zeros(n);
        let mut longest_y_series = Array1::<f32>::zeros(n);
        let mut naa = Array1::<f32>::zeros(n);
        let mut weighted_mass_errors = Array1::<f32>::zeros(n);
        let mut log10_b_ion_intensity = Array1::<f32>::zeros(n);
        let mut log10_y_ion_intensity = Array1::<f32>::zeros(n);
        let mut fwhm_rt = Array1::<f32>::zeros(n);
        let mut idf_hyperscore = Array1::<f32>::zeros(n);
        let mut idf_xic_dot_product = Array1::<f32>::zeros(n);
        let mut idf_intensity_dot_product = Array1::<f32>::zeros(n);
        let mut median_profile_sum = Array1::<f32>::zeros(n);
        let mut median_profile_sum_filtered = Array1::<f32>::zeros(n);
        let mut num_profiles = Array1::<f32>::zeros(n);
        let mut num_profiles_filtered = Array1::<f32>::zeros(n);
        let mut num_over_0_top6_idf = Array1::<f32>::zeros(n);
        let mut num_over_50_top6_idf = Array1::<f32>::zeros(n);

        for (i, feature) in self.features.iter().enumerate() {
            precursor_idxs[i] = feature.precursor_idx as u64;
            ranks[i] = feature.rank as u64;
            scores[i] = feature.score;
            mean_correlations[i] = feature.mean_correlation;
            median_correlations[i] = feature.median_correlation;
            correlation_stds[i] = feature.correlation_std;
            intensity_correlations[i] = feature.intensity_correlation;
            num_fragments[i] = feature.num_fragments;
            num_scans[i] = feature.num_scans;
            num_over_95[i] = feature.num_over_95;
            num_over_90[i] = feature.num_over_90;
            num_over_80[i] = feature.num_over_80;
            num_over_50[i] = feature.num_over_50;
            num_over_0[i] = feature.num_over_0;
            num_over_0_rank_0_5[i] = feature.num_over_0_rank_0_5;
            num_over_0_rank_6_11[i] = feature.num_over_0_rank_6_11;
            num_over_0_rank_12_17[i] = feature.num_over_0_rank_12_17;
            num_over_0_rank_18_23[i] = feature.num_over_0_rank_18_23;
            num_over_50_rank_0_5[i] = feature.num_over_50_rank_0_5;
            num_over_50_rank_6_11[i] = feature.num_over_50_rank_6_11;
            num_over_50_rank_12_17[i] = feature.num_over_50_rank_12_17;
            num_over_50_rank_18_23[i] = feature.num_over_50_rank_18_23;
            hyperscore_intensity_observations[i] = feature.hyperscore_intensity_observation;
            hyperscore_intensity_libraries[i] = feature.hyperscore_intensity_library;
            hyperscore_inverse_mass_errors[i] = feature.hyperscore_inverse_mass_error;
            rt_observeds[i] = feature.rt_observed;
            delta_rts[i] = feature.delta_rt;
            longest_b_series[i] = feature.longest_b_series;
            longest_y_series[i] = feature.longest_y_series;
            naa[i] = feature.naa;
            weighted_mass_errors[i] = feature.weighted_mass_error;
            log10_b_ion_intensity[i] = feature.log10_b_ion_intensity;
            log10_y_ion_intensity[i] = feature.log10_y_ion_intensity;
            fwhm_rt[i] = feature.fwhm_rt;
            idf_hyperscore[i] = feature.idf_hyperscore;
            idf_xic_dot_product[i] = feature.idf_xic_dot_product;
            idf_intensity_dot_product[i] = feature.idf_intensity_dot_product;
            median_profile_sum[i] = feature.median_profile_sum;
            median_profile_sum_filtered[i] = feature.median_profile_sum_filtered;
            num_profiles[i] = feature.num_profiles;
            num_profiles_filtered[i] = feature.num_profiles_filtered;
            num_over_0_top6_idf[i] = feature.num_over_0_top6_idf;
            num_over_50_top6_idf[i] = feature.num_over_50_top6_idf;
        }

        let dict = PyDict::new(py);
        dict.set_item("precursor_idx", precursor_idxs.into_pyarray(py))?;
        dict.set_item("rank", ranks.into_pyarray(py))?;
        dict.set_item("score", scores.into_pyarray(py))?;
        dict.set_item("mean_correlation", mean_correlations.into_pyarray(py))?;
        dict.set_item("median_correlation", median_correlations.into_pyarray(py))?;
        dict.set_item("correlation_std", correlation_stds.into_pyarray(py))?;
        dict.set_item(
            "intensity_correlation",
            intensity_correlations.into_pyarray(py),
        )?;
        dict.set_item("num_fragments", num_fragments.into_pyarray(py))?;
        dict.set_item("num_scans", num_scans.into_pyarray(py))?;
        dict.set_item("num_over_95", num_over_95.into_pyarray(py))?;
        dict.set_item("num_over_90", num_over_90.into_pyarray(py))?;
        dict.set_item("num_over_80", num_over_80.into_pyarray(py))?;
        dict.set_item("num_over_50", num_over_50.into_pyarray(py))?;
        dict.set_item("num_over_0", num_over_0.into_pyarray(py))?;
        dict.set_item("num_over_0_rank_0_5", num_over_0_rank_0_5.into_pyarray(py))?;
        dict.set_item(
            "num_over_0_rank_6_11",
            num_over_0_rank_6_11.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_0_rank_12_17",
            num_over_0_rank_12_17.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_0_rank_18_23",
            num_over_0_rank_18_23.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_50_rank_0_5",
            num_over_50_rank_0_5.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_50_rank_6_11",
            num_over_50_rank_6_11.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_50_rank_12_17",
            num_over_50_rank_12_17.into_pyarray(py),
        )?;
        dict.set_item(
            "num_over_50_rank_18_23",
            num_over_50_rank_18_23.into_pyarray(py),
        )?;
        dict.set_item(
            "hyperscore_intensity_observation",
            hyperscore_intensity_observations.into_pyarray(py),
        )?;
        dict.set_item(
            "hyperscore_intensity_library",
            hyperscore_intensity_libraries.into_pyarray(py),
        )?;
        dict.set_item(
            "hyperscore_inverse_mass_error",
            hyperscore_inverse_mass_errors.into_pyarray(py),
        )?;
        dict.set_item("rt_observed", rt_observeds.into_pyarray(py))?;
        dict.set_item("delta_rt", delta_rts.into_pyarray(py))?;
        dict.set_item("longest_b_series", longest_b_series.into_pyarray(py))?;
        dict.set_item("longest_y_series", longest_y_series.into_pyarray(py))?;
        dict.set_item("naa", naa.into_pyarray(py))?;
        dict.set_item("weighted_mass_error", weighted_mass_errors.into_pyarray(py))?;
        dict.set_item(
            "log10_b_ion_intensity",
            log10_b_ion_intensity.into_pyarray(py),
        )?;
        dict.set_item(
            "log10_y_ion_intensity",
            log10_y_ion_intensity.into_pyarray(py),
        )?;
        dict.set_item("fwhm_rt", fwhm_rt.into_pyarray(py))?;
        dict.set_item("idf_hyperscore", idf_hyperscore.into_pyarray(py))?;
        dict.set_item("idf_xic_dot_product", idf_xic_dot_product.into_pyarray(py))?;
        dict.set_item(
            "idf_intensity_dot_product",
            idf_intensity_dot_product.into_pyarray(py),
        )?;
        dict.set_item("median_profile_sum", median_profile_sum.into_pyarray(py))?;
        dict.set_item(
            "median_profile_sum_filtered",
            median_profile_sum_filtered.into_pyarray(py),
        )?;
        dict.set_item("num_profiles", num_profiles.into_pyarray(py))?;
        dict.set_item(
            "num_profiles_filtered",
            num_profiles_filtered.into_pyarray(py),
        )?;
        dict.set_item("num_over_0_top6_idf", num_over_0_top6_idf.into_pyarray(py))?;
        dict.set_item(
            "num_over_50_top6_idf",
            num_over_50_top6_idf.into_pyarray(py),
        )?;

        Ok(dict.into())
    }

    #[staticmethod]
    pub fn get_feature_names() -> Vec<String> {
        FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
    }
}

impl CandidateFeatureCollection {
    pub fn from_vec(features: Vec<CandidateFeature>) -> Self {
        Self { features }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CandidateFeature> {
        self.features.iter()
    }

    pub fn add_feature_internal(&mut self, feature: &CandidateFeature) {
        self.features.push(feature.clone());
    }
}
