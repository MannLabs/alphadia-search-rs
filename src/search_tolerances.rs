//! The tolerances a search runs with, as found by the optimization of a raw file.
//!
//! An immutable snapshot: the search parameter optimization produces a new one whenever it changes a tolerance.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Tolerances for selecting and scoring peak groups.
///
/// Values are `f64` so that a snapshot reproduces the values it was created from exactly; the search casts them to
/// the precision it computes in.
#[pyclass(frozen, eq, from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct SearchTolerances {
    /// Retention time tolerance in seconds.
    #[pyo3(get)]
    pub rt_error: f64,
    /// Precursor mass tolerance in ppm.
    #[pyo3(get)]
    pub ms1_error: f64,
    /// Fragment mass tolerance in ppm.
    #[pyo3(get)]
    pub ms2_error: f64,
    /// Ion mobility tolerance in 1/K0. A placeholder for data without ion mobility.
    #[pyo3(get)]
    pub mobility_error: f64,
    /// Full width at half maximum of the retention time peak in seconds.
    #[pyo3(get)]
    pub fwhm_rt: f64,
    /// Full width at half maximum of the ion mobility peak in 1/K0. A placeholder for data without ion mobility.
    #[pyo3(get)]
    pub fwhm_mobility: f64,
    /// Maximum number of peak group candidates per precursor.
    #[pyo3(get)]
    pub num_candidates: usize,
    /// Minimum selection score for a candidate to be kept.
    #[pyo3(get)]
    pub score_cutoff: f64,
}

impl SearchTolerances {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        rt_error: f64,
        ms1_error: f64,
        ms2_error: f64,
        mobility_error: f64,
        fwhm_rt: f64,
        fwhm_mobility: f64,
        num_candidates: usize,
        score_cutoff: f64,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("rt_error", rt_error),
            ("ms1_error", ms1_error),
            ("ms2_error", ms2_error),
            ("fwhm_rt", fwhm_rt),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be a positive number, got {value}"));
            }
        }
        // The mobility values are placeholders for data without ion mobility, e.g. a fwhm_mobility of -1.
        for (name, value) in [
            ("mobility_error", mobility_error),
            ("fwhm_mobility", fwhm_mobility),
            ("score_cutoff", score_cutoff),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be a finite number, got {value}"));
            }
        }
        if num_candidates == 0 {
            return Err("num_candidates must be at least 1".to_string());
        }

        Ok(Self {
            rt_error,
            ms1_error,
            ms2_error,
            mobility_error,
            fwhm_rt,
            fwhm_mobility,
            num_candidates,
            score_cutoff,
        })
    }
}

#[pymethods]
impl SearchTolerances {
    #[new]
    #[pyo3(signature = (
        *,
        rt_error,
        ms1_error,
        ms2_error,
        mobility_error,
        fwhm_rt,
        fwhm_mobility,
        num_candidates,
        score_cutoff,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        rt_error: f64,
        ms1_error: f64,
        ms2_error: f64,
        mobility_error: f64,
        fwhm_rt: f64,
        fwhm_mobility: f64,
        num_candidates: usize,
        score_cutoff: f64,
    ) -> PyResult<Self> {
        Self::try_new(
            rt_error,
            ms1_error,
            ms2_error,
            mobility_error,
            fwhm_rt,
            fwhm_mobility,
            num_candidates,
            score_cutoff,
        )
        .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchTolerances(rt_error={}, ms1_error={}, ms2_error={}, mobility_error={}, fwhm_rt={}, \
             fwhm_mobility={}, num_candidates={}, score_cutoff={})",
            self.rt_error,
            self.ms1_error,
            self.ms2_error,
            self.mobility_error,
            self.fwhm_rt,
            self.fwhm_mobility,
            self.num_candidates,
            self.score_cutoff,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> SearchTolerances {
        SearchTolerances::try_new(120.0, 30.0, 15.0, 0.1, 5.0, 0.01, 3, 0.0).unwrap()
    }

    fn with(modify: impl FnOnce(&mut SearchTolerances)) -> Result<SearchTolerances, String> {
        let mut t = valid();
        modify(&mut t);
        SearchTolerances::try_new(
            t.rt_error,
            t.ms1_error,
            t.ms2_error,
            t.mobility_error,
            t.fwhm_rt,
            t.fwhm_mobility,
            t.num_candidates,
            t.score_cutoff,
        )
    }

    #[test]
    fn test_try_new_keeps_the_values() {
        // given / when
        let tolerances = valid();

        // then
        assert_eq!(tolerances.rt_error, 120.0);
        assert_eq!(tolerances.ms1_error, 30.0);
        assert_eq!(tolerances.ms2_error, 15.0);
        assert_eq!(tolerances.mobility_error, 0.1);
        assert_eq!(tolerances.fwhm_rt, 5.0);
        assert_eq!(tolerances.fwhm_mobility, 0.01);
        assert_eq!(tolerances.num_candidates, 3);
        assert_eq!(tolerances.score_cutoff, 0.0);
    }

    #[test]
    fn test_try_new_rejects_non_positive_tolerances() {
        let cases: [(&str, fn(&mut SearchTolerances)); 4] = [
            ("rt_error", |t| t.rt_error = 0.0),
            ("ms1_error", |t| t.ms1_error = -1.0),
            ("ms2_error", |t| t.ms2_error = 0.0),
            ("fwhm_rt", |t| t.fwhm_rt = 0.0),
        ];
        for (name, modify) in cases {
            // when
            let result = with(modify);

            // then
            let error = result.expect_err(name);
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn test_try_new_rejects_non_finite_values() {
        // when
        let rt = with(|t| t.rt_error = f64::NAN);
        let cutoff = with(|t| t.score_cutoff = f64::INFINITY);

        // then
        assert!(rt.unwrap_err().contains("rt_error"));
        assert!(cutoff.unwrap_err().contains("score_cutoff"));
    }

    #[test]
    fn test_try_new_accepts_placeholder_mobility_values() {
        // given: data without ion mobility, for which alphaDIA records fwhm_mobility as -1
        // when
        let result = with(|t| {
            t.mobility_error = 0.0;
            t.fwhm_mobility = -1.0;
        });

        // then
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_new_rejects_non_finite_mobility_values() {
        // when
        let result = with(|t| t.fwhm_mobility = f64::NAN);

        // then
        assert!(result.unwrap_err().contains("fwhm_mobility"));
    }

    #[test]
    fn test_try_new_rejects_zero_candidates() {
        // when
        let result = with(|t| t.num_candidates = 0);

        // then
        assert!(result.unwrap_err().contains("num_candidates"));
    }

    #[test]
    fn test_try_new_accepts_any_finite_score_cutoff() {
        // when / then
        assert!(with(|t| t.score_cutoff = -2.5).is_ok());
    }
}
