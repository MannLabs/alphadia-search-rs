use pyo3::prelude::*;

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::peak_group_scoring::features::CandidateFeatureCollection;
use crate::SpecLibFlat;

mod algorithm;
pub mod core;
pub mod feature_block;
pub mod feature_set;
pub mod features;
pub mod ion_coverage;
pub mod library_agreement;
pub mod observation;
pub mod parameters;
#[cfg(test)]
mod tests;
pub mod utils;
pub use feature_set::FeatureSet;
pub use parameters::ScoringParameters;

#[pyclass]
pub struct PeakGroupScoring {
    params: ScoringParameters,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self { params }
    }

    pub fn score(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> CandidateFeatureCollection {
        algorithm::score_candidates(&self.params, dia_data, lib, candidates)
    }
}
