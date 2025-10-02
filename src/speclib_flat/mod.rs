use crate::idf::InverseDocumentFrequency;
use crate::precursor::Precursor;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass]
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

    /// Fragment IDF calculator
    pub idf: InverseDocumentFrequency,
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
            idf: InverseDocumentFrequency::new(&[]),
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
        // Convert arrays to vectors
        let precursor_idx_vec = precursor_idx.as_array().to_vec();
        let precursor_mz_library_vec = precursor_mz_library.as_array().to_vec();
        let precursor_mz_vec = precursor_mz.as_array().to_vec();
        let precursor_rt_library_vec = precursor_rt_library.as_array().to_vec();
        let precursor_rt_vec = precursor_rt.as_array().to_vec();
        let precursor_naa_vec = precursor_naa.as_array().to_vec();
        let flat_frag_start_idx_vec = flat_frag_start_idx.as_array().to_vec();
        let flat_frag_stop_idx_vec = flat_frag_stop_idx.as_array().to_vec();
        let fragment_mz_library_vec = fragment_mz_library.as_array().to_vec();
        let fragment_mz_vec = fragment_mz.as_array().to_vec();
        let fragment_intensity_vec = fragment_intensity.as_array().to_vec();
        let fragment_cardinality_vec = fragment_cardinality.as_array().to_vec();
        let fragment_charge_vec = fragment_charge.as_array().to_vec();
        let fragment_loss_type_vec = fragment_loss_type.as_array().to_vec();
        let fragment_number_vec = fragment_number.as_array().to_vec();
        let fragment_position_vec = fragment_position.as_array().to_vec();
        let fragment_type_vec = fragment_type.as_array().to_vec();

        // Create indices for sorting
        let mut indices: Vec<usize> = (0..precursor_idx_vec.len()).collect();

        // Sort indices by precursor_idx values
        indices.sort_by_key(|&i| precursor_idx_vec[i]);

        // Reorder all precursor arrays according to sorted indices
        let sorted_precursor_idx: Vec<usize> =
            indices.iter().map(|&i| precursor_idx_vec[i]).collect();
        let sorted_precursor_mz_library: Vec<f32> = indices
            .iter()
            .map(|&i| precursor_mz_library_vec[i])
            .collect();
        let sorted_precursor_mz: Vec<f32> = indices.iter().map(|&i| precursor_mz_vec[i]).collect();
        let sorted_precursor_rt_library: Vec<f32> = indices
            .iter()
            .map(|&i| precursor_rt_library_vec[i])
            .collect();
        let sorted_precursor_rt: Vec<f32> = indices.iter().map(|&i| precursor_rt_vec[i]).collect();
        let sorted_precursor_naa: Vec<u8> = indices.iter().map(|&i| precursor_naa_vec[i]).collect();
        let sorted_flat_frag_start_idx: Vec<usize> = indices
            .iter()
            .map(|&i| flat_frag_start_idx_vec[i])
            .collect();
        let sorted_flat_frag_stop_idx: Vec<usize> =
            indices.iter().map(|&i| flat_frag_stop_idx_vec[i]).collect();

        // Create IDF from fragment m/z library values
        let idf = InverseDocumentFrequency::new(&fragment_mz_library_vec);

        Self {
            precursor_idx: sorted_precursor_idx,
            precursor_mz_library: sorted_precursor_mz_library,
            precursor_mz: sorted_precursor_mz,
            precursor_rt_library: sorted_precursor_rt_library,
            precursor_rt: sorted_precursor_rt,
            precursor_naa: sorted_precursor_naa,
            flat_frag_start_idx: sorted_flat_frag_start_idx,
            flat_frag_stop_idx: sorted_flat_frag_stop_idx,
            fragment_mz_library: fragment_mz_library_vec,
            fragment_mz: fragment_mz_vec,
            fragment_intensity: fragment_intensity_vec,
            fragment_cardinality: fragment_cardinality_vec,
            fragment_charge: fragment_charge_vec,
            fragment_loss_type: fragment_loss_type_vec,
            fragment_number: fragment_number_vec,
            fragment_position: fragment_position_vec,
            fragment_type: fragment_type_vec,
            idf,
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

/// Filters and sorts fragments based on intensity and fragment m/z.
///
/// This function performs the following operations:
/// 1. Optionally filters out fragments with zero intensity (if `non_zero` is true)
/// 2. Selects the top-k fragments by intensity using partial sorting
/// 3. Sorts the selected fragments by fragment_mz in ascending order
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
        top_k_fragments: usize,
    ) -> Option<Precursor> {
        // Use binary search since precursor_idx is now sorted
        match self.precursor_idx.binary_search(&precursor_idx) {
            Ok(array_index) => Some(self.get_precursor_filtered(
                array_index,
                non_zero,
                filter_y1_ions,
                top_k_fragments,
            )),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests;
