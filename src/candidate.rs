use pyo3::prelude::*;
use numpy::{IntoPyArray, ndarray::Array1};


pub struct Candidate {
    /// Identifier linking to precursor
    pub precursor_idx: usize,
    /// Rank of candidate (1-based)
    pub rank: usize,
    /// Score indicating confidence
    pub score: f32,

    pub scan_center: usize,
    pub scan_start: usize,
    pub scan_stop: usize,

    pub cycle_center: usize,
    pub cycle_start: usize,
    pub cycle_stop: usize,
}



impl Candidate {
    pub fn new(
        precursor_idx: usize,
        rank: usize,
        score: f32,
        cycle_start: usize,
        cycle_center: usize,
        cycle_stop: usize,
    ) -> Self {
        Self {
            precursor_idx,
            rank,
            score,
            scan_center: 0,
            scan_start: 0,
            scan_stop: 0,
            cycle_start,
            cycle_center,
            cycle_stop,
        }
    }
}

/// Collection of candidates from a search
#[pyclass]
pub struct CandidateCollection {
    candidates: Vec<Candidate>,
}

impl Default for CandidateCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl CandidateCollection {
    #[new]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Convert the collection to separate arrays for all fields
    #[allow(clippy::type_complexity)]
    pub fn to_arrays(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject)> {
        let n = self.candidates.len();
        let mut precursor_idxs = Array1::<u64>::zeros(n);
        let mut ranks = Array1::<u64>::zeros(n);
        let mut scores = Array1::<f32>::zeros(n);
        let mut scan_center = Array1::<u64>::zeros(n);
        let mut scan_start = Array1::<u64>::zeros(n);
        let mut scan_stop = Array1::<u64>::zeros(n);
        let mut cycle_start = Array1::<u64>::zeros(n);
        let mut cycle_center = Array1::<u64>::zeros(n);
        let mut cycle_stop = Array1::<u64>::zeros(n);

        for (i, candidate) in self.candidates.iter().enumerate() {
            precursor_idxs[i] = candidate.precursor_idx as u64;
            ranks[i] = candidate.rank as u64;
            scores[i] = candidate.score;
            scan_center[i] = candidate.scan_center as u64;
            scan_start[i] = candidate.scan_start as u64;
            scan_stop[i] = candidate.scan_stop as u64;
            
            cycle_start[i] = candidate.cycle_start as u64;
            cycle_center[i] = candidate.cycle_center as u64;
            cycle_stop[i] = candidate.cycle_stop as u64;
        }

        Ok((
            precursor_idxs.into_pyarray(py).into(),
            ranks.into_pyarray(py).into(),
            scores.into_pyarray(py).into(),
            scan_center.into_pyarray(py).into(),
            scan_start.into_pyarray(py).into(),
            scan_stop.into_pyarray(py).into(),
            cycle_center.into_pyarray(py).into(),
            cycle_start.into_pyarray(py).into(),
            cycle_stop.into_pyarray(py).into(),
        ))
    }

}

impl CandidateCollection {
    pub fn from_vec(candidates: Vec<Candidate>) -> Self {
        Self { candidates }
    }
}
