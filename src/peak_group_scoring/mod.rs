use numpy::ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::convolution::convolution;
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::kernel::GaussianKernel;
use crate::precursor::Precursor;
use crate::score::axis_log_dot_product;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;

pub mod parameters;
pub use parameters::ScoringParameters;

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self {
            params,
        }
    }

    pub fn score(&self, dia_data: &DIAData, lib: &SpecLibFlat, candidates: &CandidateCollection) {

    }
}