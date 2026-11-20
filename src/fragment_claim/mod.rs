//! Fragment claim features: how much of a candidate's matched fragment signal a higher-scoring
//! candidate in the same isolation window matches at the same time, and what the evidence
//! looks like without those fragments. Computed for every candidate, target and decoy alike,
//! before the classifier runs.

mod algorithm;
#[cfg(test)]
pub(crate) mod tests;

pub use algorithm::{
    compute_claim_features, enabled_features, ClaimFeatures, ClaimParameters, DEFAULT_CYCLE_RADIUS,
    HARD_FEATURES, SOFT_FEATURES,
};

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::speclib_flat::SpecLibFlat;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

/// Candidates in, one row of claim features per candidate out.
///
/// The description below comes from `src/fragment_claim/DOCS.md`.
#[doc = include_str!("DOCS.md")]
#[pyclass]
pub struct FragmentClaim {
    params: ClaimParameters,
}

#[pymethods]
impl FragmentClaim {
    /// * `mass_tolerance` - ppm tolerance of the trace extraction, the scoring tolerance.
    /// * `top_k_fragments` - fragments per candidate, the same value as used for scoring.
    /// * `claim_tolerance` - ppm distance within which two observed fragment m/z are the same
    ///   ion; `None` uses `mass_tolerance`.
    /// * `cycle_radius` - another candidate can claim a fragment if its apex cycle is at most
    ///   this far away.
    /// * `soft_claims` - also compute the soft claims (`clm_soft_*`): the share of every matched
    ///   peak the mixture model of the co-isolated precursors attributes to the candidate.
    ///
    /// Raises `ValueError` if a tolerance is not positive or `top_k_fragments` is zero.
    #[new]
    #[pyo3(signature = (mass_tolerance, top_k_fragments, claim_tolerance = None, cycle_radius = DEFAULT_CYCLE_RADIUS, soft_claims = false))]
    fn new(
        mass_tolerance: f32,
        top_k_fragments: usize,
        claim_tolerance: Option<f32>,
        cycle_radius: u32,
        soft_claims: bool,
    ) -> PyResult<Self> {
        let params = ClaimParameters {
            mass_tolerance,
            top_k_fragments,
            claim_tolerance: claim_tolerance.unwrap_or(mass_tolerance),
            cycle_radius,
            soft_claims,
        };
        params.validate().map_err(PyValueError::new_err)?;
        Ok(Self { params })
    }

    /// Gives a dict of arrays with one row per candidate, in the candidate order of the
    /// caller: `precursor_idx`, `rank` and the columns of
    /// `get_feature_names(soft_claims)` for this instance's `soft_claims`.
    ///
    /// Raises `ValueError` if the run or the candidates exceed the sizes of the index.
    fn compute(
        &self,
        py: Python<'_>,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> PyResult<Py<PyAny>> {
        let features = compute_claim_features(dia_data, lib, candidates, &self.params)
            .map_err(PyValueError::new_err)?;
        let dict = PyDict::new(py);
        let precursor_idx: Vec<u64> = features.iter().map(|f| f.precursor_idx as u64).collect();
        let rank: Vec<u64> = features.iter().map(|f| f.rank as u64).collect();
        dict.set_item("precursor_idx", precursor_idx.into_pyarray(py))?;
        dict.set_item("rank", rank.into_pyarray(py))?;
        for (name, value) in enabled_features(self.params.soft_claims) {
            let values: Vec<f32> = features.iter().map(value).collect();
            dict.set_item(*name, values.into_pyarray(py))?;
        }
        Ok(dict.into())
    }

    /// The feature columns `compute` returns: the hard claims, and with `soft_claims` the soft
    /// claims after them.
    #[staticmethod]
    #[pyo3(signature = (soft_claims = false))]
    pub fn get_feature_names(soft_claims: bool) -> Vec<String> {
        enabled_features(soft_claims)
            .map(|(name, _)| name.to_string())
            .collect()
    }
}
