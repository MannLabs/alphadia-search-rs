//! The scored features of the candidates: one struct per candidate holding its feature blocks, and the collection
//! that `PeakGroupScoring::score` returns to Python.

use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::detection::DetectionFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::feature_set::{FeatureBlockId, FeatureSet};
use crate::peak_group_scoring::fragment_isotope::FragmentIsotopeFeatures;
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;
use crate::peak_group_scoring::ms1::Ms1Features;
use crate::peak_group_scoring::window_share::WindowShareFeatures;

#[cfg(test)]
mod tests;

/// The features of one candidate: one field per block, `None` unless the feature set has the block.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFeature {
    pub precursor_idx: usize,
    pub rank: usize,
    pub core: CoreFeatures,
    pub ion_coverage: Option<IonCoverageFeatures>,
    pub library_agreement: Option<LibraryAgreementFeatures>,
    pub fragment_isotope: Option<FragmentIsotopeFeatures>,
    pub window_share: Option<WindowShareFeatures>,
    pub ms1: Option<Ms1Features>,
    pub detection: Option<DetectionFeatures>,
}

impl CandidateFeature {
    /// Feature values in the order of `feature_set.feature_names()`, or `None` if a block of the set is missing.
    pub fn values(&self, feature_set: FeatureSet) -> Option<Vec<f32>> {
        let blocks: Option<Vec<Vec<f32>>> = feature_set
            .blocks()
            .iter()
            .map(|block| self.block_values(*block))
            .collect();
        blocks.map(|blocks| blocks.concat())
    }

    fn block_values(&self, block: FeatureBlockId) -> Option<Vec<f32>> {
        match block {
            FeatureBlockId::Core => Some(self.core.values()),
            FeatureBlockId::IonCoverage => self.ion_coverage.map(|features| features.values()),
            FeatureBlockId::LibraryAgreement => {
                self.library_agreement.map(|features| features.values())
            }
            FeatureBlockId::FragmentIsotope => {
                self.fragment_isotope.map(|features| features.values())
            }
            FeatureBlockId::WindowShare => self.window_share.map(|features| features.values()),
            FeatureBlockId::Ms1 => self.ms1.map(|features| features.values()),
            FeatureBlockId::Detection => self.detection.map(|features| features.values()),
        }
    }
}

#[pyclass]
pub struct CandidateFeatureCollection {
    feature_set: FeatureSet,
    features: Vec<CandidateFeature>,
}

#[pymethods]
impl CandidateFeatureCollection {
    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    #[getter]
    pub fn feature_set(&self) -> FeatureSet {
        self.feature_set
    }

    pub fn to_dict_arrays(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut columns =
            FeatureColumns::new(self.feature_set.feature_names(), self.features.len());
        for (i, feature) in self.features.iter().enumerate() {
            let values = feature.values(self.feature_set).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Candidate with precursor_idx {} and rank {} misses a block of the {:?} feature set",
                    feature.precursor_idx, feature.rank, self.feature_set
                ))
            })?;
            columns.set_row(i, feature.precursor_idx, feature.rank, values);
        }
        columns.into_dict(py)
    }

    #[staticmethod]
    pub fn get_feature_names(feature_set: FeatureSet) -> Vec<String> {
        feature_set
            .feature_names()
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl CandidateFeatureCollection {
    pub fn from_vec(feature_set: FeatureSet, features: Vec<CandidateFeature>) -> Self {
        Self {
            feature_set,
            features,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CandidateFeature> {
        self.features.iter()
    }
}

/// The feature columns the scoring methods return to Python: `precursor_idx` and `rank` (u64), then one f32 column
/// per feature name, filled row by row with the values in the order of the names.
pub(crate) struct FeatureColumns {
    names: Vec<&'static str>,
    precursor_idx: Array1<u64>,
    rank: Array1<u64>,
    columns: Vec<Array1<f32>>,
}

impl FeatureColumns {
    pub(crate) fn new(names: Vec<&'static str>, num_rows: usize) -> Self {
        let columns = vec![Array1::<f32>::zeros(num_rows); names.len()];
        Self {
            names,
            precursor_idx: Array1::<u64>::zeros(num_rows),
            rank: Array1::<u64>::zeros(num_rows),
            columns,
        }
    }

    pub(crate) fn set_row(
        &mut self,
        row: usize,
        precursor_idx: usize,
        rank: usize,
        values: Vec<f32>,
    ) {
        self.precursor_idx[row] = precursor_idx as u64;
        self.rank[row] = rank as u64;
        for (column, value) in self.columns.iter_mut().zip(values) {
            column[row] = value;
        }
    }

    /// The dict of numpy arrays, keyed by column name.
    pub(crate) fn into_dict(self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("precursor_idx", self.precursor_idx.into_pyarray(py))?;
        dict.set_item("rank", self.rank.into_pyarray(py))?;
        for (name, column) in self.names.into_iter().zip(self.columns) {
            dict.set_item(name, column.into_pyarray(py))?;
        }
        Ok(dict.into())
    }
}
