use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dia_data::DIAData;
use crate::dia_data_next_gen::DIADataNextGen;
use crate::precursor::Precursor;
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
    fn score_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) {
        let start_time = Instant::now();

        // Parallel iteration over candidates to score each one
        let _scored_candidates: Vec<_> = candidates
            .iter()
            .collect::<Vec<_>>()
            .par_iter()
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx(candidate.precursor_idx) {
                    Some(precursor) => {
                        Some(self.score_candidate_generic(dia_data, &precursor, candidate))
                    }
                    None => {
                        eprintln!(
                            "Warning: Candidate precursor_idx {} not found in library. Skipping.",
                            candidate.precursor_idx
                        );
                        None
                    }
                }
            })
            .collect();

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = candidates.len() as f32 / duration.as_secs_f32();
        println!(
            "Scored {} candidates at {:.2} candidates/second",
            candidates.len(),
            candidates_per_second
        );
    }

    /// Generic candidate scoring function that works with any type implementing DIADataTrait
    fn score_candidate_generic<T: DIADataTrait>(
        &self,
        _dia_data: &T,
        _precursor: &Precursor,
        candidate: &Candidate,
    ) -> f32 {
        // Scoring implementation for individual candidate will be added here
        // For now, return the original score
        candidate.score
    }
}
