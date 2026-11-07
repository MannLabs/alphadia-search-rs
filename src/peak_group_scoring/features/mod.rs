//! The scored features of the candidates: one struct per candidate holding its feature blocks, and the collection
//! that `PeakGroupScoring::score` returns to Python.

use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::feature_set::{FeatureBlockId, FeatureSet};
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;
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
    pub window_share: Option<WindowShareFeatures>,
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
            FeatureBlockId::WindowShare => self.window_share.map(|features| features.values()),
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
        let names = self.feature_set.feature_names();
        let n = self.features.len();

        let mut precursor_idxs = Array1::<u64>::zeros(n);
        let mut ranks = Array1::<u64>::zeros(n);
        let mut columns: Vec<Array1<f32>> = vec![Array1::<f32>::zeros(n); names.len()];

        for (i, feature) in self.features.iter().enumerate() {
            precursor_idxs[i] = feature.precursor_idx as u64;
            ranks[i] = feature.rank as u64;
            let values = feature.values(self.feature_set).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Candidate with precursor_idx {} and rank {} misses a block of the {:?} feature set",
                    feature.precursor_idx, feature.rank, self.feature_set
                ))
            })?;
            for (column, value) in columns.iter_mut().zip(values) {
                column[i] = value;
            }
        }

        let dict = PyDict::new(py);
        dict.set_item("precursor_idx", precursor_idxs.into_pyarray(py))?;
        dict.set_item("rank", ranks.into_pyarray(py))?;
        for (name, column) in names.iter().zip(columns) {
            dict.set_item(*name, column.into_pyarray(py))?;
        }
        Ok(dict.into())
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
