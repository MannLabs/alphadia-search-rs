//! # Quantified Spectral Library Module
//!
//! This module provides the `SpecLibFlatQuantified` structure for storing spectral libraries
//! that have been quantified against experimental DIA (Data-Independent Acquisition) data.
//!
//! Precursors are being quntified as part of DIA search following the identification of candidates.
//! Therefore, additional columns are needed on top of the regular SpecLibFlat and some columns are used differently.
//!
//! `fragment_intensity` is the intensity of the fragment ion observed in the experimental data.
//! `fragment_mz_observed` is the m/z value of the fragment ion observed in the experimental data.
//! `fragment_correlation_observed` is the correlation coefficient between the fragment's elution profile and the median profile of all fragments for that precursor.
//! `fragment_mass_error_observed` is the mass error in Da between theoretical and observed fragment m/z values.
//!

use crate::precursor_quantified::PrecursorQuantified;
use pyo3::prelude::*;

#[pyclass]
pub struct SpecLibFlatQuantified {
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
    /// Precursor rank, sorted according to precursor_idx order
    precursor_rank: Vec<usize>,
    /// Observed retention time, sorted according to precursor_idx order
    precursor_rt_observed: Vec<f32>,
    /// Start indices into fragment arrays for each precursor, sorted according to precursor_idx order
    flat_frag_start_idx: Vec<usize>,
    /// Stop indices into fragment arrays for each precursor, sorted according to precursor_idx order
    flat_frag_stop_idx: Vec<usize>,
    /// Fragment m/z values, as originally stored in the library
    /// Needed for downstream optimizations where a calibration model learns mz_observed as function of mz_library
    fragment_mz_library: Vec<f32>,

    /// Fragment m/z values, expected to be sorted in ascending order within each precursor upon creation
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
    /// Observed fragment m/z values
    fragment_mz_observed: Vec<f32>,
    /// Observed fragment correlation values
    fragment_correlation_observed: Vec<f32>,
    /// Observed fragment mass error values
    fragment_mass_error_observed: Vec<f32>,
    /// Fragment precursor indices (expanded to fragment dimension)
    fragment_precursor_idx: Vec<usize>,
    /// Fragment precursor ranks (expanded to fragment dimension)
    fragment_precursor_rank: Vec<usize>,
}

#[pymethods]
impl SpecLibFlatQuantified {
    #[new]
    fn new() -> Self {
        Self {
            precursor_idx: Vec::new(),
            precursor_mz_library: Vec::new(),
            precursor_mz: Vec::new(),
            precursor_rt_library: Vec::new(),
            precursor_rt: Vec::new(),
            precursor_naa: Vec::new(),
            precursor_rank: Vec::new(),
            precursor_rt_observed: Vec::new(),
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
            fragment_mz_observed: Vec::new(),
            fragment_correlation_observed: Vec::new(),
            fragment_mass_error_observed: Vec::new(),
            fragment_precursor_idx: Vec::new(),
            fragment_precursor_rank: Vec::new(),
        }
    }

    #[getter]
    pub fn num_precursors(&self) -> usize {
        self.precursor_mz.len()
    }

    #[getter]
    pub fn num_fragments(&self) -> usize {
        self.fragment_mz.len()
    }

    pub fn to_dict_arrays(&self, py: Python) -> PyResult<pyo3::PyObject> {
        use numpy::IntoPyArray;
        use pyo3::types::{PyDict, PyDictMethods, PyTuple};

        let precursor_dict = PyDict::new(py);
        let fragment_dict = PyDict::new(py);

        // Precursor arrays
        precursor_dict.set_item("precursor_idx", self.precursor_idx.clone().into_pyarray(py))?;
        precursor_dict.set_item(
            "mz_library",
            self.precursor_mz_library.clone().into_pyarray(py),
        )?;
        precursor_dict.set_item("mz", self.precursor_mz.clone().into_pyarray(py))?;
        precursor_dict.set_item(
            "rt_library",
            self.precursor_rt_library.clone().into_pyarray(py),
        )?;
        precursor_dict.set_item("rt", self.precursor_rt.clone().into_pyarray(py))?;
        precursor_dict.set_item("naa", self.precursor_naa.clone().into_pyarray(py))?;
        precursor_dict.set_item("rank", self.precursor_rank.clone().into_pyarray(py))?;
        precursor_dict.set_item(
            "rt_observed",
            self.precursor_rt_observed.clone().into_pyarray(py),
        )?;
        precursor_dict.set_item(
            "flat_frag_start_idx",
            self.flat_frag_start_idx.clone().into_pyarray(py),
        )?;
        precursor_dict.set_item(
            "flat_frag_stop_idx",
            self.flat_frag_stop_idx.clone().into_pyarray(py),
        )?;

        // Fragment arrays (library data)
        fragment_dict.set_item(
            "mz_library",
            self.fragment_mz_library.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item("mz", self.fragment_mz.clone().into_pyarray(py))?;
        fragment_dict.set_item(
            "intensity",
            self.fragment_intensity.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item(
            "cardinality",
            self.fragment_cardinality.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item("charge", self.fragment_charge.clone().into_pyarray(py))?;
        fragment_dict.set_item(
            "loss_type",
            self.fragment_loss_type.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item("number", self.fragment_number.clone().into_pyarray(py))?;
        fragment_dict.set_item("position", self.fragment_position.clone().into_pyarray(py))?;
        fragment_dict.set_item("type", self.fragment_type.clone().into_pyarray(py))?;

        // Fragment arrays (quantified data)
        fragment_dict.set_item(
            "mz_observed",
            self.fragment_mz_observed.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item(
            "correlation_observed",
            self.fragment_correlation_observed.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item(
            "mass_error_observed",
            self.fragment_mass_error_observed.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item(
            "precursor_idx",
            self.fragment_precursor_idx.clone().into_pyarray(py),
        )?;
        fragment_dict.set_item(
            "rank",
            self.fragment_precursor_rank.clone().into_pyarray(py),
        )?;

        let precursor_obj: pyo3::PyObject = precursor_dict.into();
        let fragment_obj: pyo3::PyObject = fragment_dict.into();
        let result_tuple = PyTuple::new(py, &[precursor_obj, fragment_obj])?;
        Ok(result_tuple.into())
    }
}

impl SpecLibFlatQuantified {
    pub fn from_precursor_quantified_vec(precursors: Vec<PrecursorQuantified>) -> Self {
        if precursors.is_empty() {
            return Self::new();
        }

        let mut precursor_idx = Vec::new();
        let mut precursor_mz_library = Vec::new();
        let mut precursor_mz = Vec::new();
        let mut precursor_rt_library = Vec::new();
        let mut precursor_rt = Vec::new();
        let mut precursor_naa = Vec::new();
        let mut precursor_rank = Vec::new();
        let mut precursor_rt_observed = Vec::new();
        let mut flat_frag_start_idx = Vec::new();
        let mut flat_frag_stop_idx = Vec::new();
        let mut fragment_mz_library = Vec::new();
        let mut fragment_mz = Vec::new();
        let mut fragment_intensity = Vec::new();
        let mut fragment_cardinality = Vec::new();
        let mut fragment_charge = Vec::new();
        let mut fragment_loss_type = Vec::new();
        let mut fragment_number = Vec::new();
        let mut fragment_position = Vec::new();
        let mut fragment_type = Vec::new();
        let mut fragment_mz_observed = Vec::new();
        let mut fragment_correlation_observed = Vec::new();
        let mut fragment_mass_error_observed = Vec::new();
        let mut fragment_precursor_idx = Vec::new();
        let mut fragment_precursor_rank = Vec::new();

        let mut current_fragment_idx = 0;

        for precursor in precursors {
            precursor_idx.push(precursor.precursor_idx);
            precursor_mz_library.push(precursor.mz_library);
            precursor_mz.push(precursor.mz);
            precursor_rt_library.push(precursor.rt_library);
            precursor_rt.push(precursor.rt);
            precursor_naa.push(precursor.naa);
            precursor_rank.push(precursor.rank);
            precursor_rt_observed.push(precursor.rt_observed);

            let start_idx = current_fragment_idx;
            current_fragment_idx += precursor.fragment_mz.len();
            let stop_idx = current_fragment_idx;

            flat_frag_start_idx.push(start_idx);
            flat_frag_stop_idx.push(stop_idx);

            // Calculate fragment count before moving fragment_mz
            let fragment_count = precursor.fragment_mz.len();

            fragment_mz_library.extend(precursor.fragment_mz_library);
            fragment_mz.extend(precursor.fragment_mz);
            fragment_intensity.extend(precursor.fragment_intensity);
            fragment_cardinality.extend(precursor.fragment_cardinality);
            fragment_charge.extend(precursor.fragment_charge);
            fragment_loss_type.extend(precursor.fragment_loss_type);
            fragment_number.extend(precursor.fragment_number);
            fragment_position.extend(precursor.fragment_position);
            fragment_type.extend(precursor.fragment_type);
            fragment_mz_observed.extend(precursor.fragment_mz_observed);
            fragment_correlation_observed.extend(precursor.fragment_correlation_observed);
            fragment_mass_error_observed.extend(precursor.fragment_mass_error_observed);

            // Expand precursor values to fragment dimension
            fragment_precursor_idx.extend(vec![precursor.precursor_idx; fragment_count]);
            fragment_precursor_rank.extend(vec![precursor.rank; fragment_count]);
        }

        // Create indices for sorting
        let mut indices: Vec<usize> = (0..precursor_idx.len()).collect();

        // Sort indices by precursor_idx values
        indices.sort_by_key(|&i| precursor_idx[i]);

        // Reorder all precursor arrays according to sorted indices
        let sorted_precursor_idx: Vec<usize> = indices.iter().map(|&i| precursor_idx[i]).collect();
        let sorted_precursor_mz_library: Vec<f32> =
            indices.iter().map(|&i| precursor_mz_library[i]).collect();
        let sorted_precursor_mz: Vec<f32> = indices.iter().map(|&i| precursor_mz[i]).collect();
        let sorted_precursor_rt_library: Vec<f32> =
            indices.iter().map(|&i| precursor_rt_library[i]).collect();
        let sorted_precursor_rt: Vec<f32> = indices.iter().map(|&i| precursor_rt[i]).collect();
        let sorted_precursor_naa: Vec<u8> = indices.iter().map(|&i| precursor_naa[i]).collect();
        let sorted_precursor_rank: Vec<usize> =
            indices.iter().map(|&i| precursor_rank[i]).collect();
        let sorted_precursor_rt_observed: Vec<f32> =
            indices.iter().map(|&i| precursor_rt_observed[i]).collect();
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
            precursor_rank: sorted_precursor_rank,
            precursor_rt_observed: sorted_precursor_rt_observed,
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
            fragment_mz_observed,
            fragment_correlation_observed,
            fragment_mass_error_observed,
            fragment_precursor_idx,
            fragment_precursor_rank,
        }
    }
}

#[cfg(test)]
mod tests;
