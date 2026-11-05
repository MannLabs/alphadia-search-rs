//! The scored features of the candidates: one struct per candidate holding its feature blocks, and the collection
//! that `PeakGroupScoring::score` returns to Python.

use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::feature_set::{Block, FeatureSet};
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;

#[cfg(test)]
mod tests;

/// The blocks that only the full feature set scores.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FullOnlyFeatures {
    pub ion_coverage: IonCoverageFeatures,
    pub library_agreement: LibraryAgreementFeatures,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFeature {
    pub precursor_idx: usize,
    pub rank: usize,
    pub core: CoreFeatures,
    /// `None` with the calibration feature set
    pub full_only: Option<FullOnlyFeatures>,
}

impl CandidateFeature {
    pub fn feature_set(&self) -> FeatureSet {
        if self.full_only.is_some() {
            FeatureSet::Full
        } else {
            FeatureSet::Calibration
        }
    }

    /// Feature values in the order of `self.feature_set().feature_names()`
    pub fn values(&self) -> Vec<f32> {
        self.feature_set()
            .blocks()
            .iter()
            .flat_map(|block| self.block_values(*block))
            .collect()
    }

    fn block_values(&self, block: Block) -> Vec<f32> {
        match block {
            Block::Core => self.core.values(),
            Block::IonCoverage => self.full_only().ion_coverage.values(),
            Block::LibraryAgreement => self.full_only().library_agreement.values(),
        }
    }

    fn full_only(&self) -> &FullOnlyFeatures {
        self.full_only
            .as_ref()
            .expect("only full-set candidates have the full-only blocks")
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
            for (column, value) in columns.iter_mut().zip(feature.values()) {
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
    /// Every feature must be of `feature_set`, so that it fills every column.
    pub fn from_vec(feature_set: FeatureSet, features: Vec<CandidateFeature>) -> PyResult<Self> {
        if let Some(feature) = features.iter().find(|f| f.feature_set() != feature_set) {
            return Err(PyValueError::new_err(format!(
                "Candidate with precursor_idx {} and rank {} is scored with {:?}, not with {:?}",
                feature.precursor_idx,
                feature.rank,
                feature.feature_set(),
                feature_set
            )));
        }
        Ok(Self {
            feature_set,
            features,
        })
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CandidateFeature> {
        self.features.iter()
    }
}
