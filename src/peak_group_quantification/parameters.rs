use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct QuantificationParameters {
    /// Mass tolerance in ppm for fragment matching
    #[pyo3(get, set)]
    pub tolerance_ppm: f32,

    /// Maximum number of fragments to use for quantification per precursor
    #[pyo3(get, set)]
    pub top_k_fragments: usize,

    /// How a flat background under the peak is removed from the summed fragment XIC:
    /// "none" (plain sum), "edges" (subtract the mean of the window's first and last cycle) or
    /// "profile" (fit each XIC as a * median profile + b and report a * sum(profile))
    #[pyo3(get, set)]
    pub background_subtraction: String,
}

pub const BACKGROUND_NONE: &str = "none";
pub const BACKGROUND_EDGES: &str = "edges";
pub const BACKGROUND_PROFILE: &str = "profile";
const BACKGROUND_MODES: [&str; 3] = [BACKGROUND_NONE, BACKGROUND_EDGES, BACKGROUND_PROFILE];

#[pymethods]
impl QuantificationParameters {
    #[new]
    pub fn new() -> Self {
        Self {
            // maximum mass error expected for fragment matching in part per million (ppm). depends on mass detector will usually be between 3 and 20ppm.
            tolerance_ppm: 7.0,
            // maximum number of fragments to use for quantification per precursor. depends on the number of fragments in the precursor.
            // very large number to capture them all by default
            top_k_fragments: 10000,
            background_subtraction: BACKGROUND_NONE.to_string(),
        }
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("tolerance_ppm")? {
            self.tolerance_ppm = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("background_subtraction")? {
            let mode = value.extract::<String>()?;
            if !BACKGROUND_MODES.contains(&mode.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "background_subtraction must be one of {BACKGROUND_MODES:?}, got {mode}"
                )));
            }
            self.background_subtraction = mode;
        }
        Ok(())
    }
}

impl Default for QuantificationParameters {
    fn default() -> Self {
        Self::new()
    }
}
