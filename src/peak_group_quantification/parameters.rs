use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Strategy used to reduce the dense XIC of a candidate window to one intensity per
/// fragment. See the [`integration`](super::integration) module for the algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuantificationMethod {
    /// Sum of every cycle in the candidate window. Unit: intensity × cycles.
    #[default]
    Sum,

    /// Trapezoidal rule over the whole candidate window. Unit: intensity × seconds.
    Trapezoid,

    /// Consensus peak boundaries plus a base-to-base linear baseline, then the
    /// trapezoidal rule between the boundaries.
    BoundaryTrapezoid,

    /// Least-squares projection of every fragment onto the consensus elution profile,
    /// with Huber reweighting to bound the influence of interfered cycles.
    ProfileProjection,

    /// Exponentially modified Gaussian fitted to the consensus elution profile, then a
    /// per-fragment amplitude and a sub-cycle integral of the fitted shape.
    EmgFit,
}

impl QuantificationMethod {
    /// Canonical lowercase name, the value round-tripped through the Python API.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Trapezoid => "trapezoid",
            Self::BoundaryTrapezoid => "boundary_trapezoid",
            Self::ProfileProjection => "profile_projection",
            Self::EmgFit => "emg_fit",
        }
    }

    /// Whether the strategy reads the consensus elution profile. `sum` and `trapezoid` do
    /// not, so building a refined profile for them would be wasted work on the default path.
    pub fn uses_elution_profile(&self) -> bool {
        match self {
            Self::Sum | Self::Trapezoid => false,
            Self::BoundaryTrapezoid | Self::ProfileProjection | Self::EmgFit => true,
        }
    }

    /// Parse a method name. Aliases are accepted so that configurations written against
    /// earlier naming keep working.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "trapezoid" | "trapezoidal" => Ok(Self::Trapezoid),
            "boundary_trapezoid" | "boundary" => Ok(Self::BoundaryTrapezoid),
            "profile_projection" | "projection" | "matched_filter" => Ok(Self::ProfileProjection),
            "emg_fit" | "emg" => Ok(Self::EmgFit),
            other => Err(format!(
                "Invalid quantification method '{other}'. Must be one of: \
                 'sum', 'trapezoid', 'boundary_trapezoid', 'profile_projection', 'emg_fit'."
            )),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct QuantificationParameters {
    /// Mass tolerance in ppm for fragment matching
    #[pyo3(get, set)]
    pub tolerance_ppm: f32,

    /// Maximum number of fragments to use for quantification per precursor
    #[pyo3(get, set)]
    pub top_k_fragments: usize,

    /// Strategy used to turn the dense XIC window into one intensity per fragment.
    /// Exposed to Python as the string returned by [`QuantificationMethod::as_str`].
    pub method: QuantificationMethod,

    /// All methods except `sum` and `trapezoid`: strength of the Whittaker-Henderson
    /// smoother applied to the consensus elution profile. Dimensionless — retention time is
    /// rescaled to unit window width first. Zero disables smoothing.
    #[pyo3(get, set)]
    pub template_smoothing_lambda: f32,

    /// All methods except `sum` and `trapezoid`: minimum correlation a fragment must have
    /// with the first-pass consensus profile to help define the refined one. Fragments below
    /// the cut are still quantified. Zero disables the refinement.
    #[pyo3(get, set)]
    pub template_min_correlation: f32,

    /// `boundary_trapezoid`: how far the consensus profile may climb above its running
    /// minimum before the boundary walk stops, as a factor. `1.15` allows a 15% rise, so
    /// noise does not end the walk early but a neighbouring peak does.
    #[pyo3(get, set)]
    pub boundary_valley_tolerance: f32,

    /// `boundary_trapezoid`: withhold the whole peak group when its unsubtracted area
    /// exceeds its baseline-subtracted area by more than this factor, which means the window
    /// sits on the shoulder of a neighbour rather than on a peak. Zero disables the test;
    /// see the module docs before enabling it.
    #[pyo3(get, set)]
    pub min_area_surviving_ratio: f32,

    /// `boundary_trapezoid`: subtract a linear baseline drawn between the intensities at
    /// the two peak boundaries before integrating.
    #[pyo3(get, set)]
    pub subtract_baseline: bool,

    /// `profile_projection` and `emg_fit`: number of iteratively reweighted least-squares
    /// passes used to bound the influence of interfered cycles. Zero gives plain
    /// least squares.
    #[pyo3(get, set)]
    pub robust_iterations: usize,

    /// `profile_projection` and `emg_fit`: fit a flat background alongside the amplitude, so
    /// that chemical background under the peak does not inflate the area. Costs one degree of
    /// freedom and can absorb part of a peak that fills the whole window.
    #[pyo3(get, set)]
    pub projection_fit_baseline: bool,

    /// `profile_projection` and `emg_fit`: Huber threshold in robust scale units. Smaller
    /// values reject interference more aggressively at the cost of efficiency on clean
    /// peaks.
    #[pyo3(get, set)]
    pub huber_k: f32,

    /// `emg_fit`: factor by which the retention time window is widened before the fitted
    /// peak shape is integrated. `1.0` integrates exactly over the observed window;
    /// larger values recover the tails of peaks truncated by the candidate window.
    #[pyo3(get, set)]
    pub emg_extrapolation_factor: f32,

    /// `emg_fit`: number of grid points per acquisition cycle used to integrate the
    /// fitted peak shape. Controls the sub-cycle resolution of the reported area.
    #[pyo3(get, set)]
    pub emg_upsample_factor: usize,
}

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
            // the historical behaviour, kept as the default so that existing searches are unaffected
            method: QuantificationMethod::EmgFit,
            // swept optimum reported for the same smoother in Pioneer
            template_smoothing_lambda: 5e-7,
            // a fragment correlating below 0.5 with the peak group disagrees about the shape
            template_min_correlation: 0.5,
            // 15% above the running minimum, as in Pioneer's boundary walk
            boundary_valley_tolerance: 1.15,
            // disabled: withholding a quantity currently drops the identification with it
            min_area_surviving_ratio: 0.0,
            subtract_baseline: true,
            // three passes are enough for the weights to settle on windows of this size
            robust_iterations: 3,
            projection_fit_baseline: true,
            // 1.5 robust scale units is the conventional Huber threshold
            huber_k: 1.5,
            // integrate over the observed window only; opt in to tail recovery explicitly
            emg_extrapolation_factor: 1.0,
            emg_upsample_factor: 8,
        }
    }

    /// Quantification strategy as a lowercase string, e.g. `"profile_projection"`.
    #[getter(method)]
    pub fn get_method(&self) -> &'static str {
        self.method.as_str()
    }

    #[setter(method)]
    pub fn set_method(&mut self, value: &str) -> PyResult<()> {
        self.method = QuantificationMethod::from_name(value)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    pub fn update(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(value) = config.get_item("tolerance_ppm")? {
            self.tolerance_ppm = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("top_k_fragments")? {
            self.top_k_fragments = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("method")? {
            self.set_method(&value.extract::<String>()?)?;
        }
        if let Some(value) = config.get_item("template_smoothing_lambda")? {
            self.template_smoothing_lambda = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("template_min_correlation")? {
            self.template_min_correlation = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("boundary_valley_tolerance")? {
            self.boundary_valley_tolerance = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("min_area_surviving_ratio")? {
            self.min_area_surviving_ratio = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("subtract_baseline")? {
            self.subtract_baseline = value.extract::<bool>()?;
        }
        if let Some(value) = config.get_item("robust_iterations")? {
            self.robust_iterations = value.extract::<usize>()?;
        }
        if let Some(value) = config.get_item("projection_fit_baseline")? {
            self.projection_fit_baseline = value.extract::<bool>()?;
        }
        if let Some(value) = config.get_item("huber_k")? {
            self.huber_k = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("emg_extrapolation_factor")? {
            self.emg_extrapolation_factor = value.extract::<f32>()?;
        }
        if let Some(value) = config.get_item("emg_upsample_factor")? {
            self.emg_upsample_factor = value.extract::<usize>()?;
        }
        Ok(())
    }
}

impl Default for QuantificationParameters {
    fn default() -> Self {
        Self::new()
    }
}
