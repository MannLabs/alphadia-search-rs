//! Which features `PeakGroupScoring` computes. A feature set is a list of feature blocks; the split follows the FDR
//! rounds of alphaDIA (MannLabs/alphadia#910):
//!
//! | block               | Calibration | Full |
//! |---------------------|-------------|------|
//! | `core`              | x           | x    |
//! | `ion_coverage`      |             | x    |
//! | `library_agreement` |             | x    |

use pyo3::prelude::*;

/// The feature set of a scoring run.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSet {
    Full,
    Calibration,
}
