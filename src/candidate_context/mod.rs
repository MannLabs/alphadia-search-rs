//! Cross-candidate competition and isolation-window context features.
//!
//! Instead of removing a candidate whose fragments a better candidate also claims, this module
//! measures that competition for every candidate before the classifier runs. Python passes the
//! same DIA data, library and candidates as for scoring; all calculations occur here.

mod algorithm;
#[cfg(test)]
mod tests;

pub use algorithm::{
    compute_context_features, ContextFeatures, ContextParameters, DEFAULT_CYCLE_RADIUS,
    DEFAULT_MIN_SHARED, FEATURES,
};

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::speclib_flat::SpecLibFlat;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

/// Candidates in, one row of context features per candidate out.
///
/// The description below comes from `src/candidate_context/DOCS.md`. It gives the problem, the
/// parameters, the output columns and the algorithm.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct CandidateContext {
    params: ContextParameters,
}

#[pymethods]
impl CandidateContext {
    /// * `mass_tolerance` - two fragment m/z are the same ion if they lie within this value in
    ///   ppm of each other.
    /// * `top_k_fragments` - fragments per candidate, the same value as used for scoring.
    /// * `min_shared` - two candidates compete if they share this many fragment m/z or more.
    /// * `cycle_radius` - two candidates are compared if their apex cycles are at most this
    ///   many cycles apart.
    ///
    /// Raises `ValueError` if the tolerance is not positive or if a count is zero.
    #[new]
    #[pyo3(signature = (
        mass_tolerance,
        top_k_fragments,
        min_shared = DEFAULT_MIN_SHARED,
        cycle_radius = DEFAULT_CYCLE_RADIUS,
    ))]
    fn new(
        mass_tolerance: f32,
        top_k_fragments: usize,
        min_shared: usize,
        cycle_radius: u32,
    ) -> PyResult<Self> {
        let params = ContextParameters {
            mass_tolerance,
            top_k_fragments,
            min_shared,
            cycle_radius,
        };
        params.validate().map_err(PyValueError::new_err)?;
        Ok(Self { params })
    }

    /// Gives a dict of arrays with one row per candidate, in the candidate order of the
    /// caller: `precursor_idx`, `rank` and the columns of `get_feature_names()`.
    ///
    /// Raises `ValueError` if the run is too large for the packed index (see `DOCS.md`).
    fn compute(
        &self,
        py: Python<'_>,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> PyResult<Py<PyAny>> {
        let features = compute_context_features(dia_data, lib, candidates, &self.params)
            .map_err(PyValueError::new_err)?;

        let dict = PyDict::new(py);
        let precursor_idx: Vec<u64> = features.iter().map(|f| f.precursor_idx as u64).collect();
        let rank: Vec<u64> = features.iter().map(|f| f.rank as u64).collect();
        dict.set_item("precursor_idx", precursor_idx.into_pyarray(py))?;
        dict.set_item("rank", rank.into_pyarray(py))?;
        for (name, value) in FEATURES {
            let values: Vec<f32> = features.iter().map(value).collect();
            dict.set_item(*name, values.into_pyarray(py))?;
        }
        Ok(dict.into())
    }

    #[staticmethod]
    pub fn get_feature_names() -> Vec<String> {
        FEATURES.iter().map(|(name, _)| name.to_string()).collect()
    }
}
