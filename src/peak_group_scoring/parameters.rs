use crate::traits::{check_mobility_tolerance, DEFAULT_MOBILITY_TOLERANCE};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::peak_group_scoring::feature_set::FeatureSet;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ScoringParameters {
    #[pyo3(get)]
    pub mass_tolerance: f32,
    #[pyo3(get)]
    pub mobility_tolerance: f32,
    #[pyo3(get)]
    pub mobility_peak_tolerance: f32,
    #[pyo3(get)]
    pub top_k_fragments: usize,
    #[pyo3(get)]
    pub feature_set: FeatureSet,
    #[pyo3(get)]
    pub ms1_mass_tolerance: f32,
    #[pyo3(get)]
    pub detection_features: bool,
}

#[pymethods]
impl ScoringParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            mass_tolerance: 7.0,
            // maximum ion mobility (1/K0) deviation from the precursor's library value, ignored for data without mobility.
            mobility_tolerance: DEFAULT_MOBILITY_TOLERANCE,
            // half width (1/K0) of the window around each candidate's own apex mobility its XICs are read from; 0 reads
            // the whole `mobility_tolerance` window around the library mobility
            mobility_peak_tolerance: 0.0,
            // maximum number of fragments to use for scoring per precursor. depends on the number of fragments in the precursor.
            top_k_fragments: 100,
            feature_set: FeatureSet::Full,
            // maximum mass error for matching precursor isotopes in the MS1 scans, in ppm (MS1 block and `score_ms1_calibration` only)
            ms1_mass_tolerance: 10.0,
            // score the detection block when the feature set has it (on by default, so the feature set alone decides);
            // an opt-out that skips the run-wide detection model to save compute, the block's features are then 0
            detection_features: true,
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("mass_tolerance")? {
            self.mass_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("mobility_tolerance")? {
            self.mobility_tolerance =
                check_mobility_tolerance(value.extract::<f32>()?).map_err(PyValueError::new_err)?;
        }
        if let Some(value) = config.get_item("mobility_peak_tolerance")? {
            self.mobility_peak_tolerance =
                check_mobility_tolerance(value.extract::<f32>()?).map_err(PyValueError::new_err)?;
        }
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("feature_set")? {
            self.feature_set = value.extract::<FeatureSet>()?;
        }
        if let Some(value) = config.get_item("ms1_mass_tolerance")? {
            self.ms1_mass_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("detection_features")? {
            self.detection_features = value.extract::<bool>()?;
        }
        Ok(())
    }
}

impl Default for ScoringParameters {
    fn default() -> Self {
        Self::new()
    }
}
