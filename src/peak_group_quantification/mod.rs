//! Peak group quantification operates only on candidates scored at 1% FDR.
//!
//! Avoids storing memory-intensive fragment-level properties for all possible candidates
//! (e.g., 3M precursors × 3 candidates × target+decoy = 18M candidates) by quantifying
//! only the pre-filtered high-confidence subset.

use pyo3::prelude::*;

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::{SpecLibFlat, SpecLibFlatQuantified};

mod algorithm;
pub mod parameters;
#[cfg(test)]
mod tests;
pub use parameters::QuantificationParameters;

#[pyclass]
pub struct PeakGroupQuantification {
    params: QuantificationParameters,
}

#[pymethods]
impl PeakGroupQuantification {
    #[new]
    pub fn new(params: QuantificationParameters) -> Self {
        Self { params }
    }

    pub fn quantify(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        algorithm::quantify_generic(&self.params, dia_data, lib, candidates)
    }
}
