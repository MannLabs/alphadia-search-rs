use crate::calibration::CalibrationEstimator;
use crate::dia_data::DIAData;
use crate::precursor::Precursor;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

mod columns;
mod hdf;
pub use columns::read_hdf_columns;
pub use hdf::read_flat_hdf;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SpecLibFlat {
    /// Precursor indices, MUST be sorted in ascending order for binary search to work correctly
    precursor_idx: Vec<usize>,

    /// Precursor m/z values, as originally stored in the library
    /// Needed for downstream optimizations where a calibration model learns mz_observed as function of mz_library
    precursor_mz_library: Vec<f32>,

    /// Precursor m/z values, sorted according to precursor_idx order
    /// Used for extraction of precursor XICs and selection of quadrupole windows
    /// It's left to the caller if these are precursor_mz_library or precursor_mz_calibrated values, depending on optimization and calibration
    precursor_mz: Vec<f32>,

    /// Precursor retention times, as originally stored in the library
    /// Needed for downstream optimizations where a calibration model learns rt_observed as function of rt_library
    precursor_rt_library: Vec<f32>,

    /// Precursor retention times, sorted according to precursor_idx order
    /// It's left to the caller if these are precursor_rt_library or precursor_rt_calibrated values, depending on optimization and calibration
    precursor_rt: Vec<f32>,

    /// Number of amino acids in the precursor sequence, sorted according to precursor_idx order
    precursor_naa: Vec<u8>,

    /// Start indices into fragment arrays for each precursor, sorted according to precursor_idx order
    flat_frag_start_idx: Vec<usize>,

    /// Stop indices into fragment arrays for each precursor, sorted according to precursor_idx order
    flat_frag_stop_idx: Vec<usize>,

    /// Fragment m/z values, as originally stored in the library
    /// Needed for downstream optimizations where a calibration model learns mz_observed as function of mz_library
    fragment_mz_library: Vec<f32>,

    /// Fragment m/z values, sorted as originally stored in the library
    /// These mz values are used for extraction of the fragment XIC
    /// It's left to the caller if these are fragment_mz_library or fragment_mz_calibrated values, depending on optimization and calibration
    /// Mass errors etc. will be calculated against these values
    fragment_mz: Vec<f32>,

    /// Fragment intensity values in original library order (NOT sorted, maintains original order within each precursor)
    fragment_intensity: Vec<f32>,

    /// Fragment cardinality values
    fragment_cardinality: Vec<u8>,

    /// Fragment charge values
    fragment_charge: Vec<u8>,

    /// Fragment loss type values
    fragment_loss_type: Vec<u8>,

    /// Fragment number values
    fragment_number: Vec<u8>,

    /// Fragment position values
    fragment_position: Vec<u8>,

    /// Fragment type values
    fragment_type: Vec<u8>,
}

#[pymethods]
impl SpecLibFlat {
    #[new]
    fn new() -> Self {
        Self {
            precursor_idx: Vec::new(),
            precursor_mz_library: Vec::new(),
            precursor_mz: Vec::new(),
            precursor_rt_library: Vec::new(),
            precursor_rt: Vec::new(),
            precursor_naa: Vec::new(),
            flat_frag_start_idx: Vec::new(),
            flat_frag_stop_idx: Vec::new(),
            fragment_mz_library: Vec::new(),
            fragment_mz: Vec::new(),
            fragment_intensity: Vec::new(),
            fragment_cardinality: Vec::new(),
            fragment_charge: Vec::new(),
            fragment_loss_type: Vec::new(),
            fragment_number: Vec::new(),
            fragment_position: Vec::new(),
            fragment_type: Vec::new(),
        }
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        precursor_idx: PyReadonlyArray1<'_, usize>,
        precursor_mz_library: PyReadonlyArray1<'_, f32>,
        precursor_mz: PyReadonlyArray1<'_, f32>,
        precursor_rt_library: PyReadonlyArray1<'_, f32>,
        precursor_rt: PyReadonlyArray1<'_, f32>,
        precursor_naa: PyReadonlyArray1<'_, u8>,
        flat_frag_start_idx: PyReadonlyArray1<'_, usize>,
        flat_frag_stop_idx: PyReadonlyArray1<'_, usize>,
        fragment_mz_library: PyReadonlyArray1<'_, f32>,
        fragment_mz: PyReadonlyArray1<'_, f32>,
        fragment_intensity: PyReadonlyArray1<'_, f32>,
        fragment_cardinality: PyReadonlyArray1<'_, u8>,
        fragment_charge: PyReadonlyArray1<'_, u8>,
        fragment_loss_type: PyReadonlyArray1<'_, u8>,
        fragment_number: PyReadonlyArray1<'_, u8>,
        fragment_position: PyReadonlyArray1<'_, u8>,
        fragment_type: PyReadonlyArray1<'_, u8>,
    ) -> Self {
        Self::from_vecs(
            precursor_idx.as_array().to_vec(),
            precursor_mz_library.as_array().to_vec(),
            precursor_mz.as_array().to_vec(),
            precursor_rt_library.as_array().to_vec(),
            precursor_rt.as_array().to_vec(),
            precursor_naa.as_array().to_vec(),
            flat_frag_start_idx.as_array().to_vec(),
            flat_frag_stop_idx.as_array().to_vec(),
            fragment_mz_library.as_array().to_vec(),
            fragment_mz.as_array().to_vec(),
            fragment_intensity.as_array().to_vec(),
            fragment_cardinality.as_array().to_vec(),
            fragment_charge.as_array().to_vec(),
            fragment_loss_type.as_array().to_vec(),
            fragment_number.as_array().to_vec(),
            fragment_position.as_array().to_vec(),
            fragment_type.as_array().to_vec(),
        )
    }

    /// Reads the flat spectral library that alphabase `SpecLibFlat.save_hdf` wrote to `path`.
    ///
    /// The library values are also used as the search values, as `from_arrays` does when it gets
    /// the library columns twice.
    #[staticmethod]
    fn from_hdf(path: PathBuf) -> PyResult<Self> {
        read_flat_hdf(&path).map_err(PyValueError::new_err)
    }

    /// Returns an independent copy, for changes in place that must not change this library.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Prepares the library for the search of `dia_data`, in place.
    ///
    /// 1. Scales the normalized library RT to the RT range of `dia_data`, see `normalize_rt`.
    /// 2. Removes the precursors whose library m/z is outside the isolation windows of the DIA cycle,
    ///    see `filter_precursor_mz`.
    ///
    /// The search values are reset to the library values.
    fn init_for_run(&mut self, dia_data: PyRef<'_, DIAData>) -> PyResult<()> {
        let rt_values = &dia_data.rt_values;
        if rt_values.is_empty() {
            return Err(PyValueError::new_err("The DIA data has no RT values"));
        }
        let (mz_lower, mz_upper) = dia_data
            .cycle
            .iter()
            .filter(|&&mz| mz > 0.0)
            .fold(None, |limits: Option<(f32, f32)>, &mz| match limits {
                None => Some((mz, mz)),
                Some((lower, upper)) => Some((lower.min(mz), upper.max(mz))),
            })
            .ok_or_else(|| PyValueError::new_err("The DIA cycle has no isolation window"))?;

        self.normalize_rt(rt_values[0], rt_values[rt_values.len() - 1]);
        self.filter_precursor_mz(mz_lower, mz_upper);
        Ok(())
    }

    /// Returns a new library with only the precursors in `precursor_idx` and their fragments.
    ///
    /// Raises `ValueError` if a precursor is not in the library.
    fn subset(&self, precursor_idx: PyReadonlyArray1<'_, usize>) -> PyResult<Self> {
        let precursor_idx = precursor_idx.as_array();
        let precursor_idx = precursor_idx
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| precursor_idx.iter().copied().collect());
        self.subset_by_idx(&precursor_idx)
            .map_err(PyValueError::new_err)
    }

    /// Sets the search values from the library values, in place.
    ///
    /// For each fitted estimator, the search value is the calibrated library value.
    /// For each estimator that is `None`, the search value is the library value.
    #[pyo3(signature = (rt=None, precursor_mz=None, fragment_mz=None))]
    fn calibrate(
        &mut self,
        rt: Option<PyRef<'_, CalibrationEstimator>>,
        precursor_mz: Option<PyRef<'_, CalibrationEstimator>>,
        fragment_mz: Option<PyRef<'_, CalibrationEstimator>>,
    ) -> PyResult<()> {
        let calibrated = |estimator: Option<&CalibrationEstimator>, input: &[f32]| match estimator {
            Some(estimator) => estimator
                .predict_values(input)
                .map_err(PyValueError::new_err),
            None => Ok(input.to_vec()),
        };
        self.precursor_rt = calibrated(rt.as_deref(), &self.precursor_rt_library)?;
        self.precursor_mz = calibrated(precursor_mz.as_deref(), &self.precursor_mz_library)?;
        self.fragment_mz = calibrated(fragment_mz.as_deref(), &self.fragment_mz_library)?;
        Ok(())
    }

    /// Returns the precursor and the fragment columns as two dicts of numpy arrays.
    fn to_dict_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let precursors = PyDict::new(py);
        precursors.set_item("precursor_idx", self.precursor_idx.clone().into_pyarray(py))?;
        precursors.set_item(
            "mz_library",
            self.precursor_mz_library.clone().into_pyarray(py),
        )?;
        precursors.set_item("mz", self.precursor_mz.clone().into_pyarray(py))?;
        precursors.set_item(
            "rt_library",
            self.precursor_rt_library.clone().into_pyarray(py),
        )?;
        precursors.set_item("rt", self.precursor_rt.clone().into_pyarray(py))?;
        precursors.set_item("nAA", self.precursor_naa.clone().into_pyarray(py))?;
        precursors.set_item(
            "flat_frag_start_idx",
            self.flat_frag_start_idx.clone().into_pyarray(py),
        )?;
        precursors.set_item(
            "flat_frag_stop_idx",
            self.flat_frag_stop_idx.clone().into_pyarray(py),
        )?;

        let fragments = PyDict::new(py);
        fragments.set_item(
            "mz_library",
            self.fragment_mz_library.clone().into_pyarray(py),
        )?;
        fragments.set_item("mz", self.fragment_mz.clone().into_pyarray(py))?;
        fragments.set_item(
            "intensity",
            self.fragment_intensity.clone().into_pyarray(py),
        )?;
        fragments.set_item(
            "cardinality",
            self.fragment_cardinality.clone().into_pyarray(py),
        )?;
        fragments.set_item("charge", self.fragment_charge.clone().into_pyarray(py))?;
        fragments.set_item(
            "loss_type",
            self.fragment_loss_type.clone().into_pyarray(py),
        )?;
        fragments.set_item("number", self.fragment_number.clone().into_pyarray(py))?;
        fragments.set_item("position", self.fragment_position.clone().into_pyarray(py))?;
        fragments.set_item("type", self.fragment_type.clone().into_pyarray(py))?;

        Ok((precursors, fragments))
    }

    #[getter]
    pub fn num_precursors(&self) -> usize {
        self.precursor_mz.len()
    }

    #[getter]
    pub fn num_fragments(&self) -> usize {
        self.fragment_mz.len()
    }
}

/// Apply fragment filtering and return filtered fragment vectors
type FragmentData = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
);

/// Highest ion number of the b and y ions that `filter_short_ions` drops.
const SHORT_ION_MAX_NUMBER: u8 = 2;

/// b1, b2, y1 and y2 are shared by many precursors and decoys, so they add selection signal at many wrong retention times.
fn is_short_ion(fragment_type: u8, number: u8) -> bool {
    (fragment_type == crate::constants::FragmentType::B
        || fragment_type == crate::constants::FragmentType::Y)
        && number <= SHORT_ION_MAX_NUMBER
}

/// Filters and sorts fragments based on intensity and fragment m/z.
///
/// This function performs the following operations:
/// 1. Optionally filters out fragments with zero intensity (if `non_zero` is true)
/// 2. Optionally filters out y1 ions and the short b1, b2, y1 and y2 ions
/// 3. Selects the top-k fragments by intensity using partial sorting
/// 4. Sorts the selected fragments by fragment_mz in ascending order
///
/// # Arguments
///
/// * `fragment_mz` - Fragment m/z values
/// * `fragment_mz_library` - Library fragment m/z values
/// * `fragment_intensity` - Fragment intensities
/// * `fragment_cardinality` - Fragment cardinality values
/// * `fragment_charge` - Fragment charge states
/// * `fragment_loss_type` - Fragment loss type annotations
/// * `fragment_number` - Fragment number annotations
/// * `fragment_position` - Fragment position annotations
/// * `fragment_type` - Fragment type annotations
/// * `non_zero` - If true, filters out fragments with zero intensity
/// * `filter_y1_ions` - If true, filters out y1 ions
/// * `filter_short_ions` - If true, filters out b1, b2, y1 and y2 before the top-k choice
/// * `top_k_fragments` - Maximum number of fragments to return (top-k by intensity)
///
/// # Returns
///
/// Returns a tuple of filtered and sorted fragment data vectors, ordered by fragment_mz ascending
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn filter_sort_fragments(
    fragment_mz: &[f32],
    fragment_mz_library: &[f32],
    fragment_intensity: &[f32],
    fragment_cardinality: &[u8],
    fragment_charge: &[u8],
    fragment_loss_type: &[u8],
    fragment_number: &[u8],
    fragment_position: &[u8],
    fragment_type: &[u8],
    non_zero: bool,
    filter_y1_ions: bool,
    filter_short_ions: bool,
    top_k_fragments: usize,
) -> FragmentData {
    let mut fragment_data: Vec<(f32, f32, f32, u8, u8, u8, u8, u8, u8, usize)> = (0..fragment_mz
        .len())
        .map(|idx| {
            (
                fragment_mz[idx],
                fragment_mz_library[idx],
                fragment_intensity[idx],
                fragment_cardinality[idx],
                fragment_charge[idx],
                fragment_loss_type[idx],
                fragment_number[idx],
                fragment_position[idx],
                fragment_type[idx],
                idx,
            )
        })
        .collect();

    // Filter non-zero intensities if requested
    if non_zero {
        fragment_data.retain(|(_, _, intensity, _, _, _, _, _, _, _)| *intensity > 0.0);
    }

    // Filter out y1 ions if requested (fragment_type = Y (121) AND fragment_number = 1)
    if filter_y1_ions {
        fragment_data.retain(|(_, _, _, _, _, _, number, _, fragment_type, _)| {
            !(*fragment_type == crate::constants::FragmentType::Y && *number == 1)
        });
    }

    if filter_short_ions {
        fragment_data.retain(|(_, _, _, _, _, _, number, _, fragment_type, _)| {
            !is_short_ion(*fragment_type, *number)
        });
    }

    // Use partial sorting for top-k selection - much faster than full sort
    let k = top_k_fragments.min(fragment_data.len());
    if k < fragment_data.len() {
        // Partial sort: only sort the k-th element and everything before it
        fragment_data.select_nth_unstable_by(k, |a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });
        fragment_data.truncate(k);
    }
    // Sort by fragment_mz in ascending order
    fragment_data.sort_by(
        |(a, _, _, _, _, _, _, _, _, _), (b, _, _, _, _, _, _, _, _, _)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        },
    );

    let mut mz_vec = Vec::new();
    let mut mz_library_vec = Vec::new();
    let mut intensity_vec = Vec::new();
    let mut cardinality_vec = Vec::new();
    let mut charge_vec = Vec::new();
    let mut loss_type_vec = Vec::new();
    let mut number_vec = Vec::new();
    let mut position_vec = Vec::new();
    let mut type_vec = Vec::new();

    for (
        mz,
        mz_library,
        intensity,
        cardinality,
        charge,
        loss_type,
        number,
        position,
        frag_type,
        _,
    ) in fragment_data
    {
        mz_vec.push(mz);
        mz_library_vec.push(mz_library);
        intensity_vec.push(intensity);
        cardinality_vec.push(cardinality);
        charge_vec.push(charge);
        loss_type_vec.push(loss_type);
        number_vec.push(number);
        position_vec.push(position);
        type_vec.push(frag_type);
    }

    (
        mz_vec,
        mz_library_vec,
        intensity_vec,
        cardinality_vec,
        charge_vec,
        loss_type_vec,
        number_vec,
        position_vec,
        type_vec,
    )
}

// Regular Rust implementation (not exposed to Python)
impl SpecLibFlat {
    /// Builds the library from plain vectors, without a Python interpreter.
    ///
    /// The precursor arrays are reordered by `precursor_idx` so that lookups can use binary
    /// search. The fragment arrays keep their library order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_vecs(
        precursor_idx: Vec<usize>,
        precursor_mz_library: Vec<f32>,
        precursor_mz: Vec<f32>,
        precursor_rt_library: Vec<f32>,
        precursor_rt: Vec<f32>,
        precursor_naa: Vec<u8>,
        flat_frag_start_idx: Vec<usize>,
        flat_frag_stop_idx: Vec<usize>,
        fragment_mz_library: Vec<f32>,
        fragment_mz: Vec<f32>,
        fragment_intensity: Vec<f32>,
        fragment_cardinality: Vec<u8>,
        fragment_charge: Vec<u8>,
        fragment_loss_type: Vec<u8>,
        fragment_number: Vec<u8>,
        fragment_position: Vec<u8>,
        fragment_type: Vec<u8>,
    ) -> Self {
        let mut indices: Vec<usize> = (0..precursor_idx.len()).collect();
        indices.sort_by_key(|&i| precursor_idx[i]);

        let sorted_precursor_idx: Vec<usize> = indices.iter().map(|&i| precursor_idx[i]).collect();
        let sorted_precursor_mz_library: Vec<f32> =
            indices.iter().map(|&i| precursor_mz_library[i]).collect();
        let sorted_precursor_mz: Vec<f32> = indices.iter().map(|&i| precursor_mz[i]).collect();
        let sorted_precursor_rt_library: Vec<f32> =
            indices.iter().map(|&i| precursor_rt_library[i]).collect();
        let sorted_precursor_rt: Vec<f32> = indices.iter().map(|&i| precursor_rt[i]).collect();
        let sorted_precursor_naa: Vec<u8> = indices.iter().map(|&i| precursor_naa[i]).collect();
        let sorted_flat_frag_start_idx: Vec<usize> =
            indices.iter().map(|&i| flat_frag_start_idx[i]).collect();
        let sorted_flat_frag_stop_idx: Vec<usize> =
            indices.iter().map(|&i| flat_frag_stop_idx[i]).collect();

        Self {
            precursor_idx: sorted_precursor_idx,
            precursor_mz_library: sorted_precursor_mz_library,
            precursor_mz: sorted_precursor_mz,
            precursor_rt_library: sorted_precursor_rt_library,
            precursor_rt: sorted_precursor_rt,
            precursor_naa: sorted_precursor_naa,
            flat_frag_start_idx: sorted_flat_frag_start_idx,
            flat_frag_stop_idx: sorted_flat_frag_stop_idx,
            fragment_mz_library,
            fragment_mz,
            fragment_intensity,
            fragment_cardinality,
            fragment_charge,
            fragment_loss_type,
            fragment_number,
            fragment_position,
            fragment_type,
        }
    }

    /// Scales the normalized library RT to the range from `rt_start` to `rt_stop`, in place.
    ///
    /// The minimum and the maximum library RT of all precursors become `rt_start` and `rt_stop`.
    /// The calculation is the same as the two `np.interp` calls of alphaDIA `_norm_to_rt`.
    /// The search RT is reset to the new library RT.
    pub fn normalize_rt(&mut self, rt_start: f32, rt_stop: f32) {
        let (rt_min, rt_max) = self
            .precursor_rt_library
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &rt| {
                (min.min(rt as f64), max.max(rt as f64))
            });
        let (rt_start, rt_stop) = (rt_start as f64, rt_stop as f64);

        for rt in self.precursor_rt_library.iter_mut() {
            let rt_norm = interp(*rt as f64, rt_min, rt_max, 0.0, 1.0);
            *rt = interp(rt_norm, 0.0, 1.0, rt_start, rt_stop) as f32;
        }
        self.precursor_rt = self.precursor_rt_library.clone();
    }

    /// Removes the precursors whose library m/z is not in the range from `mz_lower` to `mz_upper`, in place.
    ///
    /// The limits are included. The fragments of the removed precursors are also removed.
    /// The search m/z values are reset to the library values.
    pub fn filter_precursor_mz(&mut self, mz_lower: f32, mz_upper: f32) {
        let keep: Vec<usize> = (0..self.precursor_idx.len())
            .filter(|&i| {
                let mz = self.precursor_mz_library[i];
                mz >= mz_lower && mz <= mz_upper
            })
            .collect();
        *self = self.select(&keep);
        self.precursor_mz = self.precursor_mz_library.clone();
        self.fragment_mz = self.fragment_mz_library.clone();
    }

    /// Returns a new library with only the precursors in `precursor_idx` and their fragments.
    ///
    /// Returns `Err` if a precursor is not in the library.
    pub fn subset_by_idx(&self, precursor_idx: &[usize]) -> Result<Self, String> {
        let mut indices = Vec::with_capacity(precursor_idx.len());
        let mut num_missing = 0usize;
        for idx in precursor_idx {
            match self.precursor_idx.binary_search(idx) {
                Ok(index) => indices.push(index),
                Err(_) => num_missing += 1,
            }
        }
        if num_missing > 0 {
            return Err(format!(
                "{num_missing} of {} precursors are not in the spectral library",
                precursor_idx.len()
            ));
        }
        indices.sort_unstable();
        indices.dedup();
        Ok(self.select(&indices))
    }

    /// Returns a new library with the precursors at the ascending array `indices` and their fragments.
    ///
    /// The fragments are stored again without gaps, in the order of the precursors.
    fn select(&self, indices: &[usize]) -> Self {
        let num_fragments: usize = indices
            .iter()
            .map(|&i| self.flat_frag_stop_idx[i] - self.flat_frag_start_idx[i])
            .sum();

        let mut library = Self {
            precursor_idx: indices.iter().map(|&i| self.precursor_idx[i]).collect(),
            precursor_mz_library: indices
                .iter()
                .map(|&i| self.precursor_mz_library[i])
                .collect(),
            precursor_mz: indices.iter().map(|&i| self.precursor_mz[i]).collect(),
            precursor_rt_library: indices
                .iter()
                .map(|&i| self.precursor_rt_library[i])
                .collect(),
            precursor_rt: indices.iter().map(|&i| self.precursor_rt[i]).collect(),
            precursor_naa: indices.iter().map(|&i| self.precursor_naa[i]).collect(),
            flat_frag_start_idx: Vec::with_capacity(indices.len()),
            flat_frag_stop_idx: Vec::with_capacity(indices.len()),
            fragment_mz_library: Vec::with_capacity(num_fragments),
            fragment_mz: Vec::with_capacity(num_fragments),
            fragment_intensity: Vec::with_capacity(num_fragments),
            fragment_cardinality: Vec::with_capacity(num_fragments),
            fragment_charge: Vec::with_capacity(num_fragments),
            fragment_loss_type: Vec::with_capacity(num_fragments),
            fragment_number: Vec::with_capacity(num_fragments),
            fragment_position: Vec::with_capacity(num_fragments),
            fragment_type: Vec::with_capacity(num_fragments),
        };

        for &i in indices {
            let range = self.flat_frag_start_idx[i]..self.flat_frag_stop_idx[i];
            library.flat_frag_start_idx.push(library.fragment_mz.len());
            library
                .fragment_mz_library
                .extend_from_slice(&self.fragment_mz_library[range.clone()]);
            library
                .fragment_mz
                .extend_from_slice(&self.fragment_mz[range.clone()]);
            library
                .fragment_intensity
                .extend_from_slice(&self.fragment_intensity[range.clone()]);
            library
                .fragment_cardinality
                .extend_from_slice(&self.fragment_cardinality[range.clone()]);
            library
                .fragment_charge
                .extend_from_slice(&self.fragment_charge[range.clone()]);
            library
                .fragment_loss_type
                .extend_from_slice(&self.fragment_loss_type[range.clone()]);
            library
                .fragment_number
                .extend_from_slice(&self.fragment_number[range.clone()]);
            library
                .fragment_position
                .extend_from_slice(&self.fragment_position[range.clone()]);
            library
                .fragment_type
                .extend_from_slice(&self.fragment_type[range]);
            library.flat_frag_stop_idx.push(library.fragment_mz.len());
        }

        library
    }

    pub fn get_precursor(&self, index: usize) -> Precursor {
        let precursor_idx = self.precursor_idx[index];
        let precursor_mz = self.precursor_mz[index];
        let precursor_mz_library = self.precursor_mz_library[index];
        let precursor_rt = self.precursor_rt[index];
        let precursor_rt_library = self.precursor_rt_library[index];
        let precursor_naa = self.precursor_naa[index];
        let start_idx = self.flat_frag_start_idx[index];
        let stop_idx = self.flat_frag_stop_idx[index];

        let fragment_mz = self.fragment_mz[start_idx..stop_idx].to_vec();
        let fragment_mz_library = self.fragment_mz_library[start_idx..stop_idx].to_vec();
        let fragment_intensity = self.fragment_intensity[start_idx..stop_idx].to_vec();
        let fragment_cardinality = self.fragment_cardinality[start_idx..stop_idx].to_vec();
        let fragment_charge = self.fragment_charge[start_idx..stop_idx].to_vec();
        let fragment_loss_type = self.fragment_loss_type[start_idx..stop_idx].to_vec();
        let fragment_number = self.fragment_number[start_idx..stop_idx].to_vec();
        let fragment_position = self.fragment_position[start_idx..stop_idx].to_vec();
        let fragment_type = self.fragment_type[start_idx..stop_idx].to_vec();

        Precursor {
            precursor_idx,
            mz: precursor_mz,
            mz_library: precursor_mz_library,
            rt: precursor_rt,
            rt_library: precursor_rt_library,
            naa: precursor_naa,
            fragment_mz,
            fragment_mz_library,
            fragment_intensity,
            fragment_cardinality,
            fragment_charge,
            fragment_loss_type,
            fragment_number,
            fragment_position,
            fragment_type,
        }
    }

    pub fn get_precursor_filtered(
        &self,
        index: usize,
        non_zero: bool,
        filter_y1_ions: bool,
        filter_short_ions: bool,
        top_k_fragments: usize,
    ) -> Precursor {
        let precursor_idx = self.precursor_idx[index];
        let precursor_mz = self.precursor_mz[index];
        let precursor_mz_library = self.precursor_mz_library[index];
        let precursor_rt = self.precursor_rt[index];
        let precursor_rt_library = self.precursor_rt_library[index];
        let precursor_naa = self.precursor_naa[index];
        let start_idx = self.flat_frag_start_idx[index];
        let stop_idx = self.flat_frag_stop_idx[index];

        let raw_fragment_mz = &self.fragment_mz[start_idx..stop_idx];
        let raw_fragment_mz_library = &self.fragment_mz_library[start_idx..stop_idx];
        let raw_fragment_intensity = &self.fragment_intensity[start_idx..stop_idx];
        let raw_fragment_cardinality = &self.fragment_cardinality[start_idx..stop_idx];
        let raw_fragment_charge = &self.fragment_charge[start_idx..stop_idx];
        let raw_fragment_loss_type = &self.fragment_loss_type[start_idx..stop_idx];
        let raw_fragment_number = &self.fragment_number[start_idx..stop_idx];
        let raw_fragment_position = &self.fragment_position[start_idx..stop_idx];
        let raw_fragment_type = &self.fragment_type[start_idx..stop_idx];

        let (
            fragment_mz,
            fragment_mz_library,
            fragment_intensity,
            fragment_cardinality,
            fragment_charge,
            fragment_loss_type,
            fragment_number,
            fragment_position,
            fragment_type,
        ) = filter_sort_fragments(
            raw_fragment_mz,
            raw_fragment_mz_library,
            raw_fragment_intensity,
            raw_fragment_cardinality,
            raw_fragment_charge,
            raw_fragment_loss_type,
            raw_fragment_number,
            raw_fragment_position,
            raw_fragment_type,
            non_zero,
            filter_y1_ions,
            filter_short_ions,
            top_k_fragments,
        );

        Precursor {
            precursor_idx,
            mz: precursor_mz,
            mz_library: precursor_mz_library,
            rt: precursor_rt,
            rt_library: precursor_rt_library,
            naa: precursor_naa,
            fragment_mz,
            fragment_mz_library,
            fragment_intensity,
            fragment_cardinality,
            fragment_charge,
            fragment_loss_type,
            fragment_number,
            fragment_position,
            fragment_type,
        }
    }

    pub fn get_precursor_by_idx_filtered(
        &self,
        precursor_idx: usize,
        non_zero: bool,
        filter_y1_ions: bool,
        filter_short_ions: bool,
        top_k_fragments: usize,
    ) -> Option<Precursor> {
        // Use binary search since precursor_idx is now sorted
        match self.precursor_idx.binary_search(&precursor_idx) {
            Ok(array_index) => Some(self.get_precursor_filtered(
                array_index,
                non_zero,
                filter_y1_ions,
                filter_short_ions,
                top_k_fragments,
            )),
            Err(_) => None,
        }
    }
}

/// Linear interpolation from `[x_start, x_stop]` to `[y_start, y_stop]`, the same as `np.interp` with two points.
///
/// Values outside the range are clipped. A value at or above `x_stop` gives `y_stop`, also if the range is empty.
fn interp(x: f64, x_start: f64, x_stop: f64, y_start: f64, y_stop: f64) -> f64 {
    if x >= x_stop {
        y_stop
    } else if x <= x_start {
        y_start
    } else {
        (y_stop - y_start) / (x_stop - x_start) * (x - x_start) + y_start
    }
}

#[cfg(test)]
mod tests;
