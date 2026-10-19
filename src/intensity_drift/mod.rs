//! Python binding of the intensity drift correction, see [`algorithm`].

mod algorithm;
#[cfg(test)]
mod tests;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::intensity_drift::algorithm::{correct_drift, DriftParameters};

/// Divides the drift of every run out of `intensity`, in place.
///
/// `intensity` and `correlation` have the run-major shape `(n_runs, n_ions)`: `intensity`
/// holds the linear fragment intensities where zero marks a missing value, `correlation` the
/// cross-run fragment correlations. `rt` has the shape `(n_runs, n_precursors)` and holds the
/// observed retention time of every precursor, NaN where a run has no identification for it.
/// `precursor` gives the column of `rt` for every ion.
///
/// Runs with too few usable ions are left unchanged.
///
/// The arrays are borrowed rather than copied and must therefore be C-contiguous.
///
/// Raises `ValueError` if the shapes do not agree or a precursor index is out of range.
/// Raises `TypeError` if an array is not contiguous.
#[pyfunction]
pub fn correct_intensity_drift(
    mut intensity: PyReadwriteArray2<'_, f64>,
    correlation: PyReadonlyArray2<'_, f64>,
    rt: PyReadonlyArray2<'_, f64>,
    precursor: PyReadonlyArray1<'_, usize>,
) -> PyResult<()> {
    let shape = intensity.shape().to_vec();
    let (n_runs, n_ions) = (shape[0], shape[1]);
    if correlation.shape() != shape.as_slice() {
        return Err(PyValueError::new_err(format!(
            "intensity and correlation must have the same shape, got {:?} and {:?}",
            shape,
            correlation.shape()
        )));
    }
    if rt.shape()[0] != n_runs || precursor.len() != n_ions {
        return Err(PyValueError::new_err(format!(
            "rt must have {n_runs} runs and precursor {n_ions} ions, got {:?} and {}",
            rt.shape(),
            precursor.len()
        )));
    }
    let precursor = precursor.as_slice()?;
    let n_precursors = rt.shape()[1];
    if precursor.iter().any(|&index| index >= n_precursors) {
        return Err(PyValueError::new_err(format!(
            "precursor indices must be smaller than the {n_precursors} precursors of rt"
        )));
    }
    // the matrices are chunked by the number of ions, which must not be zero
    if n_ions == 0 {
        return Ok(());
    }

    correct_drift(
        intensity.as_slice_mut()?,
        correlation.as_slice()?,
        rt.as_slice()?,
        precursor,
        &DriftParameters::DEFAULT,
    );
    Ok(())
}
