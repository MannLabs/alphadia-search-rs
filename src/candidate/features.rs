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
    // shape of the per-fragment correlation distribution, beyond its mean/median/std
    "correlation_min",
    "correlation_iqr",
    "weighted_mean_correlation",
    "top3_correlation",
    // mass error consistency, beyond the existing weighted mean absolute error
    "mass_error_std",
    "mass_error_mean_signed",
    "mass_error_max_abs",
    // shape of the observed fragment intensity pattern
    "spectral_entropy",
    "intensity_cosine_library",
    "intensity_top_fraction",
    "n_fragments_80pct_intensity",
    // fragment co-elution and chromatographic peak shape
    "apex_delta_std",
    "mean_abs_apex_delta",
    "profile_edge_ratio",
    "profile_asymmetry",
];

/// Number of f32 features per candidate; `feature_values` is checked against it.
pub const NUM_FEATURES: usize = FEATURE_NAMES.len();

#[derive(Debug, Clone, Default)]
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
    pub correlation_min: f32,
    pub correlation_iqr: f32,
    pub weighted_mean_correlation: f32,
    pub top3_correlation: f32,
    pub mass_error_std: f32,
    pub mass_error_mean_signed: f32,
    pub mass_error_max_abs: f32,
    pub spectral_entropy: f32,
    pub intensity_cosine_library: f32,
    pub intensity_top_fraction: f32,
    pub n_fragments_80pct_intensity: f32,
    pub apex_delta_std: f32,
    pub mean_abs_apex_delta: f32,
    pub profile_edge_ratio: f32,
    pub profile_asymmetry: f32,
}

impl CandidateFeature {
    /// Feature values in `FEATURE_NAMES` order.
    ///
    /// The fixed-size array is the ordering contract: adding a feature without adding a
    /// name here (or vice versa) fails to compile rather than silently shifting columns.
    pub fn feature_values(&self) -> [f32; NUM_FEATURES] {
        [
            self.score,
            self.mean_correlation,
            self.median_correlation,
            self.correlation_std,
            self.intensity_correlation,
            self.num_fragments,
            self.num_scans,
            self.num_over_95,
            self.num_over_90,
            self.num_over_80,
            self.num_over_50,
            self.num_over_0,
            self.num_over_0_rank_0_5,
            self.num_over_0_rank_6_11,
            self.num_over_0_rank_12_17,
            self.num_over_0_rank_18_23,
            self.num_over_50_rank_0_5,
            self.num_over_50_rank_6_11,
            self.num_over_50_rank_12_17,
            self.num_over_50_rank_18_23,
            self.hyperscore_intensity_observation,
            self.hyperscore_intensity_library,
            self.hyperscore_inverse_mass_error,
            self.rt_observed,
            self.delta_rt,
            self.longest_b_series,
            self.longest_y_series,
            self.naa,
            self.weighted_mass_error,
            self.log10_b_ion_intensity,
            self.log10_y_ion_intensity,
            self.fwhm_rt,
            self.idf_hyperscore,
            self.idf_xic_dot_product,
            self.idf_intensity_dot_product,
            self.median_profile_sum,
            self.median_profile_sum_filtered,
            self.num_profiles,
            self.num_profiles_filtered,
            self.num_over_0_top6_idf,
            self.num_over_50_top6_idf,
            self.correlation_min,
            self.correlation_iqr,
            self.weighted_mean_correlation,
            self.top3_correlation,
            self.mass_error_std,
            self.mass_error_mean_signed,
            self.mass_error_max_abs,
            self.spectral_entropy,
            self.intensity_cosine_library,
            self.intensity_top_fraction,
            self.n_fragments_80pct_intensity,
            self.apex_delta_std,
            self.mean_abs_apex_delta,
            self.profile_edge_ratio,
            self.profile_asymmetry,
        ]
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

    pub fn to_dict_arrays(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n = self.features.len();
        let dict = PyDict::new(py);

        let mut precursor_idxs = Array1::<u64>::zeros(n);
        let mut ranks = Array1::<u64>::zeros(n);
        // one column per entry of FEATURE_NAMES, filled in the same order as feature_values
        let mut columns = vec![Array1::<f32>::zeros(n); NUM_FEATURES];

        for (row, feature) in self.features.iter().enumerate() {
            precursor_idxs[row] = feature.precursor_idx as u64;
            ranks[row] = feature.rank as u64;
            for (column, value) in columns.iter_mut().zip(feature.feature_values()) {
                column[row] = value;
            }
        }

        dict.set_item("precursor_idx", precursor_idxs.into_pyarray(py))?;
        dict.set_item("rank", ranks.into_pyarray(py))?;
        for (name, column) in FEATURE_NAMES.iter().zip(columns) {
            dict.set_item(*name, column.into_pyarray(py))?;
        }

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
