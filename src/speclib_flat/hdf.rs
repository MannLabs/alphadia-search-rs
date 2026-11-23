//! Reads a flat spectral library from the HDF file that alphabase `SpecLibFlat.save_hdf` writes.
//!
//! The file holds one dataset per column in the groups `library/precursor_df` and
//! `library/fragment_df`. alphabase compresses the datasets with LZF and shuffle.
//! HDF5 converts the stored numeric types to the types that `SpecLibFlat` uses.

use super::SpecLibFlat;
use hdf5::{File, Group, H5Type};
use std::path::Path;

const PRECURSOR_GROUP: &str = "library/precursor_df";
const FRAGMENT_GROUP: &str = "library/fragment_df";

/// Reads the column `name` of the dataframe group `group` and converts it to `T`.
fn read_column<T: H5Type>(group: &Group, name: &str) -> Result<Vec<T>, String> {
    let dataset = group
        .dataset(name)
        .map_err(|e| format!("Column '{name}' not found in '{}': {e}", group.name()))?;
    dataset
        .read_raw::<T>()
        .map_err(|e| format!("Could not read column '{name}' of '{}': {e}", group.name()))
}

/// Reads an index column. Negative values are an error.
fn read_index_column(group: &Group, name: &str) -> Result<Vec<usize>, String> {
    read_column::<i64>(group, name)?
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                format!(
                    "Column '{name}' of '{}' has the negative value {value}",
                    group.name()
                )
            })
        })
        .collect()
}

/// Returns `Err` if a column of `group` does not have `expected` values.
fn check_lengths(group: &str, expected: usize, lengths: &[usize]) -> Result<(), String> {
    match lengths.iter().find(|&&length| length != expected) {
        Some(length) => Err(format!(
            "The columns of '{group}' do not have the same length: {length} and {expected}"
        )),
        None => Ok(()),
    }
}

/// Reads the flat spectral library in `path`.
///
/// The library values are also used as the search values: `precursor_mz` is `mz_library`,
/// `precursor_rt` is `rt_library` and `fragment_mz` is the fragment `mz_library`.
pub fn read_flat_hdf(path: &Path) -> Result<SpecLibFlat, String> {
    let file = File::open(path).map_err(|e| format!("Could not open '{}': {e}", path.display()))?;
    let precursors = file.group(PRECURSOR_GROUP).map_err(|e| {
        format!(
            "Group '{PRECURSOR_GROUP}' not found in '{}': {e}",
            path.display()
        )
    })?;
    let fragments = file.group(FRAGMENT_GROUP).map_err(|e| {
        format!(
            "Group '{FRAGMENT_GROUP}' not found in '{}': {e}",
            path.display()
        )
    })?;

    let precursor_idx = read_index_column(&precursors, "precursor_idx")?;
    let precursor_mz_library = read_column::<f32>(&precursors, "mz_library")?;
    let precursor_rt_library = read_column::<f32>(&precursors, "rt_library")?;
    let precursor_naa = read_column::<u8>(&precursors, "nAA")?;
    let flat_frag_start_idx = read_index_column(&precursors, "flat_frag_start_idx")?;
    let flat_frag_stop_idx = read_index_column(&precursors, "flat_frag_stop_idx")?;

    let fragment_mz_library = read_column::<f32>(&fragments, "mz_library")?;
    let fragment_intensity = read_column::<f32>(&fragments, "intensity")?;
    let fragment_cardinality = read_column::<u8>(&fragments, "cardinality")?;
    let fragment_charge = read_column::<u8>(&fragments, "charge")?;
    let fragment_loss_type = read_column::<u8>(&fragments, "loss_type")?;
    let fragment_number = read_column::<u8>(&fragments, "number")?;
    let fragment_position = read_column::<u8>(&fragments, "position")?;
    let fragment_type = read_column::<u8>(&fragments, "type")?;

    let num_precursors = precursor_idx.len();
    check_lengths(
        PRECURSOR_GROUP,
        num_precursors,
        &[
            precursor_mz_library.len(),
            precursor_rt_library.len(),
            precursor_naa.len(),
            flat_frag_start_idx.len(),
            flat_frag_stop_idx.len(),
        ],
    )?;
    let num_fragments = fragment_mz_library.len();
    check_lengths(
        FRAGMENT_GROUP,
        num_fragments,
        &[
            fragment_intensity.len(),
            fragment_cardinality.len(),
            fragment_charge.len(),
            fragment_loss_type.len(),
            fragment_number.len(),
            fragment_position.len(),
            fragment_type.len(),
        ],
    )?;
    if let Some(i) = (0..num_precursors).find(|&i| flat_frag_start_idx[i] > flat_frag_stop_idx[i]) {
        return Err(format!(
            "flat_frag_start_idx {} is larger than flat_frag_stop_idx {}",
            flat_frag_start_idx[i], flat_frag_stop_idx[i]
        ));
    }
    if let Some(stop) = flat_frag_stop_idx
        .iter()
        .find(|&&stop| stop > num_fragments)
    {
        return Err(format!(
            "flat_frag_stop_idx {stop} is larger than the number of fragments {num_fragments}"
        ));
    }

    Ok(SpecLibFlat::from_vecs(
        precursor_idx,
        precursor_mz_library.clone(),
        precursor_mz_library,
        precursor_rt_library.clone(),
        precursor_rt_library,
        precursor_naa,
        flat_frag_start_idx,
        flat_frag_stop_idx,
        fragment_mz_library.clone(),
        fragment_mz_library,
        fragment_intensity,
        fragment_cardinality,
        fragment_charge,
        fragment_loss_type,
        fragment_number,
        fragment_position,
        fragment_type,
    ))
}
