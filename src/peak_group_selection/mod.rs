use pyo3::prelude::*;

use crate::candidate::CandidateCollection;
use crate::dia_data::DIAData;
use crate::SpecLibFlat;

mod algorithm;
pub mod parameters;
pub use parameters::SelectionParameters;

#[pyclass]
pub struct PeakGroupSelection {
    params: SelectionParameters,
}

#[pymethods]
impl PeakGroupSelection {
    #[new]
    pub fn new(params: SelectionParameters) -> Self {
        Self { params }
    }

    pub fn search(&self, dia_data: &DIAData, lib: &SpecLibFlat) -> CandidateCollection {
        algorithm::search_precursors(&self.params, dia_data, lib)
    }
}

#[cfg(test)]
mod tests;
