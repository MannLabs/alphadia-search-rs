//! Candidate scoring and feature extraction for neural network classification.
//!
//! All features are standardized to f32 type for memory efficiency and consistency
//! in neural network training. While some features like fragment counts are naturally
//! integer values, using f32 throughout:
//! - Reduces memory usage compared to mixed u64/usize types
//! - Eliminates type conversion overhead in ML pipelines
//! - Provides consistent tensor dimensions for batch processing
//! - Maintains sufficient precision for all feature ranges

use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Candidate {
    /// Identifier linking to precursor
    pub precursor_idx: usize,
    /// Rank of candidate (1-based)
    pub rank: usize,
    /// Score indicating confidence
    pub score: f32,

    pub scan_center: usize,
    pub scan_start: usize,
    pub scan_stop: usize,

    pub cycle_center: usize,
    pub cycle_start: usize,
    pub cycle_stop: usize,
}

impl Candidate {
    pub fn new(
        precursor_idx: usize,
        rank: usize,
        score: f32,
        cycle_start: usize,
        cycle_center: usize,
        cycle_stop: usize,
    ) -> Self {
        Self {
            precursor_idx,
            rank,
            score,
            scan_center: 0,
            scan_start: 0,
            scan_stop: 0,
            cycle_start,
            cycle_center,
            cycle_stop,
        }
    }
}

/// Features calculated for a candidate during scoring
#[derive(Debug, Clone)]
pub struct CandidateFeature {
    /// Original candidate precursor index
    pub precursor_idx: usize,
    /// Original candidate rank
    pub rank: usize,
    /// Original candidate score
    pub score: f32,
    /// Mean correlation across all fragments
    pub mean_correlation: f32,
    /// Median correlation across all fragments
    pub median_correlation: f32,
    /// Standard deviation of correlations
    pub correlation_std: f32,
    /// Intensity correlation between observed and library intensities
    pub intensity_correlation: f32,
    /// Number of fragments used in scoring
    pub num_fragments: f32,
    /// Number of scans/cycles used in scoring
    pub num_scans: f32,
    /// Number of correlations above 0.95
    pub num_over_95: f32,
    /// Number of correlations above 0.90
    pub num_over_90: f32,
    /// Number of correlations above 0.80
    pub num_over_80: f32,
    /// Number of correlations above 0.50
    pub num_over_50: f32,
    /// Hyperscore calculated from observed intensities
    pub hyperscore_intensity_observation: f32,
    /// Hyperscore calculated from library intensities
    pub hyperscore_intensity_library: f32,
    /// Observed retention time in seconds (from cycle center)
    pub rt_observed: f32,
    /// Delta retention time (observed - library) in seconds
    pub delta_rt: f32,
    /// Longest continuous b-ion series length
    pub longest_b_series: f32,
    /// Longest continuous y-ion series length
    pub longest_y_series: f32,
    /// Number of amino acids in the precursor sequence
    pub naa: f32,
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
        hyperscore_intensity_observation: f32,
        hyperscore_intensity_library: f32,
        rt_observed: f32,
        delta_rt: f32,
        longest_b_series: f32,
        longest_y_series: f32,
        naa: f32,
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
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            rt_observed,
            delta_rt,
            longest_b_series,
            longest_y_series,
            naa,
        }
    }
}

/// Collection of candidate features
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

    /// Convert the collection to a dictionary of arrays for Python
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
        let mut hyperscore_intensity_observations = Array1::<f32>::zeros(n);
        let mut hyperscore_intensity_libraries = Array1::<f32>::zeros(n);
        let mut rt_observeds = Array1::<f32>::zeros(n);
        let mut delta_rts = Array1::<f32>::zeros(n);
        let mut longest_b_series = Array1::<f32>::zeros(n);
        let mut longest_y_series = Array1::<f32>::zeros(n);
        let mut naa = Array1::<f32>::zeros(n);

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
            hyperscore_intensity_observations[i] = feature.hyperscore_intensity_observation;
            hyperscore_intensity_libraries[i] = feature.hyperscore_intensity_library;
            rt_observeds[i] = feature.rt_observed;
            delta_rts[i] = feature.delta_rt;
            longest_b_series[i] = feature.longest_b_series;
            longest_y_series[i] = feature.longest_y_series;
            naa[i] = feature.naa;
        }

        // Create Python dictionary
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
        dict.set_item(
            "hyperscore_intensity_observation",
            hyperscore_intensity_observations.into_pyarray(py),
        )?;
        dict.set_item(
            "hyperscore_intensity_library",
            hyperscore_intensity_libraries.into_pyarray(py),
        )?;
        dict.set_item("rt_observed", rt_observeds.into_pyarray(py))?;
        dict.set_item("delta_rt", delta_rts.into_pyarray(py))?;
        dict.set_item("longest_b_series", longest_b_series.into_pyarray(py))?;
        dict.set_item("longest_y_series", longest_y_series.into_pyarray(py))?;
        dict.set_item("naa", naa.into_pyarray(py))?;

        Ok(dict.into())
    }

    /// Get the names of all f32 feature columns
    #[staticmethod]
    pub fn get_feature_names() -> Vec<String> {
        vec![
            "score".to_string(),
            "mean_correlation".to_string(),
            "median_correlation".to_string(),
            "correlation_std".to_string(),
            "intensity_correlation".to_string(),
            "num_fragments".to_string(),
            "num_scans".to_string(),
            "num_over_95".to_string(),
            "num_over_90".to_string(),
            "num_over_80".to_string(),
            "num_over_50".to_string(),
            "hyperscore_intensity_observation".to_string(),
            "hyperscore_intensity_library".to_string(),
            "rt_observed".to_string(),
            "delta_rt".to_string(),
            "longest_b_series".to_string(),
            "longest_y_series".to_string(),
            "naa".to_string(),
        ]
    }
}

impl CandidateFeatureCollection {
    pub fn from_vec(features: Vec<CandidateFeature>) -> Self {
        Self { features }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CandidateFeature> {
        self.features.iter()
    }

    /// Add a feature to the collection (internal use only)
    pub fn add_feature_internal(&mut self, feature: &CandidateFeature) {
        self.features.push(feature.clone());
    }
}

/// Collection of candidates from a search
#[pyclass]
pub struct CandidateCollection {
    candidates: Vec<Candidate>,
}

impl Default for CandidateCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl CandidateCollection {
    #[new]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Create a CandidateCollection from separate arrays
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        precursor_idxs: Vec<u64>,
        ranks: Vec<u64>,
        scores: Vec<f32>,
        scan_center: Vec<u64>,
        scan_start: Vec<u64>,
        scan_stop: Vec<u64>,
        cycle_center: Vec<u64>,
        cycle_start: Vec<u64>,
        cycle_stop: Vec<u64>,
    ) -> PyResult<Self> {
        let n = precursor_idxs.len();

        // Validate all arrays have the same length
        if ![
            ranks.len(),
            scores.len(),
            scan_center.len(),
            scan_start.len(),
            scan_stop.len(),
            cycle_center.len(),
            cycle_start.len(),
            cycle_stop.len(),
        ]
        .iter()
        .all(|&len| len == n)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        let mut candidates = Vec::with_capacity(n);
        for i in 0..n {
            candidates.push(Candidate {
                precursor_idx: precursor_idxs[i] as usize,
                rank: ranks[i] as usize,
                score: scores[i],
                scan_center: scan_center[i] as usize,
                scan_start: scan_start[i] as usize,
                scan_stop: scan_stop[i] as usize,
                cycle_center: cycle_center[i] as usize,
                cycle_start: cycle_start[i] as usize,
                cycle_stop: cycle_stop[i] as usize,
            });
        }

        Ok(Self { candidates })
    }

    /// Convert the collection to separate arrays for all fields
    #[allow(clippy::type_complexity)]
    pub fn to_arrays(
        &self,
        py: Python,
    ) -> PyResult<(
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
    )> {
        let n = self.candidates.len();
        let mut precursor_idxs = Array1::<u64>::zeros(n);
        let mut ranks = Array1::<u64>::zeros(n);
        let mut scores = Array1::<f32>::zeros(n);
        let mut scan_center = Array1::<u64>::zeros(n);
        let mut scan_start = Array1::<u64>::zeros(n);
        let mut scan_stop = Array1::<u64>::zeros(n);
        let mut cycle_start = Array1::<u64>::zeros(n);
        let mut cycle_center = Array1::<u64>::zeros(n);
        let mut cycle_stop = Array1::<u64>::zeros(n);

        for (i, candidate) in self.candidates.iter().enumerate() {
            precursor_idxs[i] = candidate.precursor_idx as u64;
            ranks[i] = candidate.rank as u64;
            scores[i] = candidate.score;
            scan_center[i] = candidate.scan_center as u64;
            scan_start[i] = candidate.scan_start as u64;
            scan_stop[i] = candidate.scan_stop as u64;

            cycle_start[i] = candidate.cycle_start as u64;
            cycle_center[i] = candidate.cycle_center as u64;
            cycle_stop[i] = candidate.cycle_stop as u64;
        }

        Ok((
            precursor_idxs.into_pyarray(py).into(),
            ranks.into_pyarray(py).into(),
            scores.into_pyarray(py).into(),
            scan_center.into_pyarray(py).into(),
            scan_start.into_pyarray(py).into(),
            scan_stop.into_pyarray(py).into(),
            cycle_center.into_pyarray(py).into(),
            cycle_start.into_pyarray(py).into(),
            cycle_stop.into_pyarray(py).into(),
        ))
    }
}

impl CandidateCollection {
    pub fn from_vec(candidates: Vec<Candidate>) -> Self {
        Self { candidates }
    }

    /// Get an iterator over the candidates
    pub fn iter(&self) -> std::slice::Iter<'_, Candidate> {
        self.candidates.iter()
    }
}

impl<'a> IntoParallelRefIterator<'a> for CandidateCollection {
    type Iter = rayon::slice::Iter<'a, Candidate>;
    type Item = &'a Candidate;

    fn par_iter(&'a self) -> Self::Iter {
        self.candidates.par_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_feature_names() {
        let feature_names = CandidateFeatureCollection::get_feature_names();

        // Verify we have the expected number of f32 features
        assert_eq!(feature_names.len(), 18);

        // Verify some key feature names are present
        assert!(feature_names.contains(&"score".to_string()));
        assert!(feature_names.contains(&"mean_correlation".to_string()));
        assert!(feature_names.contains(&"median_correlation".to_string()));
        assert!(feature_names.contains(&"correlation_std".to_string()));
        assert!(feature_names.contains(&"intensity_correlation".to_string()));
        assert!(feature_names.contains(&"num_fragments".to_string()));
        assert!(feature_names.contains(&"num_scans".to_string()));
        assert!(feature_names.contains(&"rt_observed".to_string()));
        assert!(feature_names.contains(&"delta_rt".to_string()));
        assert!(feature_names.contains(&"longest_b_series".to_string()));
        assert!(feature_names.contains(&"longest_y_series".to_string()));
        assert!(feature_names.contains(&"naa".to_string()));

        // Verify that non-f32 columns are NOT included
        assert!(!feature_names.contains(&"precursor_idx".to_string()));
        assert!(!feature_names.contains(&"rank".to_string()));

        // Verify all names are unique
        let mut sorted_names = feature_names.clone();
        sorted_names.sort();
        sorted_names.dedup();
        assert_eq!(sorted_names.len(), feature_names.len());
    }
}
