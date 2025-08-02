use pyo3::prelude::*;

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::traits::DIADataTrait;
use crate::SpecLibFlat;

pub mod parameters;
pub use parameters::ScoringParameters;

#[pyclass]
#[allow(dead_code)]
pub struct PeakGroupScoring {
    params: ScoringParameters,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self { params }
    }

    pub fn score(&self, dia_data: &DIAData, lib: &SpecLibFlat, candidates: &CandidateCollection) {
        self.score_generic(dia_data, lib, candidates)
    }

    pub fn score_next_gen(
        &self,
        dia_data: &DIADataNextGen,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) {
        self.score_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupScoring {
    /// Generic scoring function that works with any type implementing DIADataTrait
    #[allow(unused_variables)]
    fn score_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) {
        // Scoring implementation will be added here
        println!(
            "Scoring {} candidates with PeakGroupScoring",
            candidates.len()
        );
    }
}
