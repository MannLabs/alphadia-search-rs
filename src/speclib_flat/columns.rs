//! Reads the columns of a dataframe group of an alphabase HDF file into numpy arrays.
//!
//! alphabase stores each column of a dataframe as one dataset of the group.
//! Numeric and boolean columns keep their stored type. String columns become numpy object arrays of `str`.

use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{File, H5Type, Selection};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::HashMap;

/// The values of one column, read without the GIL.
enum Column {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

/// Reads all values of `dataset`, or only the values at `rows`, in the order of `rows`.
fn read<T: H5Type>(dataset: &hdf5::Dataset, rows: Option<&[usize]>) -> Result<Vec<T>, String> {
    let result = match rows {
        None => dataset.read_raw::<T>(),
        // an empty point selection is an error in HDF5
        Some([]) => Ok(Vec::new()),
        Some(rows) => {
            let points = Array2::from_shape_vec((rows.len(), 1), rows.to_vec())
                .expect("one coordinate per row");
            dataset
                .read_slice_1d::<T, _>(Selection::Points(points))
                .map(|values| values.into_raw_vec_and_offset().0)
        }
    };
    result.map_err(|e| format!("Could not read '{}': {e}", dataset.name()))
}

fn read_column(dataset: &hdf5::Dataset, rows: Option<&[usize]>) -> Result<Column, String> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(|e| format!("Unknown type of '{}': {e}", dataset.name()))?;
    Ok(match descriptor {
        TypeDescriptor::Integer(IntSize::U1) => Column::I8(read(dataset, rows)?),
        TypeDescriptor::Integer(IntSize::U2) => Column::I16(read(dataset, rows)?),
        TypeDescriptor::Integer(IntSize::U4) => Column::I32(read(dataset, rows)?),
        TypeDescriptor::Integer(IntSize::U8) => Column::I64(read(dataset, rows)?),
        TypeDescriptor::Unsigned(IntSize::U1) => Column::U8(read(dataset, rows)?),
        TypeDescriptor::Unsigned(IntSize::U2) => Column::U16(read(dataset, rows)?),
        TypeDescriptor::Unsigned(IntSize::U4) => Column::U32(read(dataset, rows)?),
        TypeDescriptor::Unsigned(IntSize::U8) => Column::U64(read(dataset, rows)?),
        TypeDescriptor::Float(FloatSize::U4) => Column::F32(read(dataset, rows)?),
        TypeDescriptor::Float(FloatSize::U8) => Column::F64(read(dataset, rows)?),
        TypeDescriptor::Boolean => Column::Bool(read(dataset, rows)?),
        TypeDescriptor::VarLenUnicode => Column::Str(
            read::<VarLenUnicode>(dataset, rows)?
                .into_iter()
                .map(|s| s.as_str().to_owned())
                .collect(),
        ),
        TypeDescriptor::VarLenAscii => Column::Str(
            read::<VarLenAscii>(dataset, rows)?
                .into_iter()
                .map(|s| s.as_str().to_owned())
                .collect(),
        ),
        other => {
            return Err(format!(
                "The type {other:?} of '{}' is not supported",
                dataset.name()
            ))
        }
    })
}

/// Reads the columns `columns` (all columns if None) of the group `group` in the file `path`.
fn read_columns(
    path: &str,
    group: &str,
    columns: Option<Vec<String>>,
    rows: Option<Vec<usize>>,
) -> Result<Vec<(String, Column)>, String> {
    let file = File::open(path).map_err(|e| format!("Could not open '{path}': {e}"))?;
    let group = file
        .group(group)
        .map_err(|e| format!("Group '{group}' not found in '{path}': {e}"))?;
    let names = match columns {
        Some(names) => names,
        None => group
            .member_names()
            .map_err(|e| format!("Could not list '{}': {e}", group.name()))?,
    };
    names
        .into_iter()
        .map(|name| {
            let dataset = group
                .dataset(&name)
                .map_err(|e| format!("Column '{name}' not found in '{}': {e}", group.name()))?;
            Ok((name, read_column(&dataset, rows.as_deref())?))
        })
        .collect()
}

fn to_python<'py>(py: Python<'py>, column: Column) -> PyResult<Bound<'py, PyAny>> {
    Ok(match column {
        Column::I8(v) => v.into_pyarray(py).into_any(),
        Column::I16(v) => v.into_pyarray(py).into_any(),
        Column::I32(v) => v.into_pyarray(py).into_any(),
        Column::I64(v) => v.into_pyarray(py).into_any(),
        Column::U8(v) => v.into_pyarray(py).into_any(),
        Column::U16(v) => v.into_pyarray(py).into_any(),
        Column::U32(v) => v.into_pyarray(py).into_any(),
        Column::U64(v) => v.into_pyarray(py).into_any(),
        Column::F32(v) => v.into_pyarray(py).into_any(),
        Column::F64(v) => v.into_pyarray(py).into_any(),
        Column::Bool(v) => v.into_pyarray(py).into_any(),
        Column::Str(v) => {
            // Equal values share one `str` object: library string columns repeat many values.
            let mut interned: HashMap<&str, Py<PyAny>> = HashMap::new();
            let objects: Vec<Py<PyAny>> = v
                .iter()
                .map(|s| {
                    interned
                        .entry(s.as_str())
                        .or_insert_with(|| PyString::new(py, s).into_any().unbind())
                        .clone_ref(py)
                })
                .collect();
            PyArray1::from_vec(py, objects).into_any()
        }
    })
}

/// Reads the columns of the dataframe group `group` of the alphabase HDF file `path`.
///
/// Returns a dict from the column name to a numpy array. `columns` selects the columns, None reads all.
/// `rows` selects the rows by position, in the given order. None reads all rows.
/// Numeric and boolean columns keep the stored type. String columns are numpy object arrays of `str`.
#[pyfunction]
#[pyo3(signature = (path, group, columns=None, rows=None))]
pub fn read_hdf_columns<'py>(
    py: Python<'py>,
    path: String,
    group: String,
    columns: Option<Vec<String>>,
    rows: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let rows: Option<Vec<usize>> = rows.map(|r| r.as_array().iter().map(|&i| i as usize).collect());
    let read = py
        .detach(|| read_columns(&path, &group, columns, rows))
        .map_err(PyValueError::new_err)?;
    let dict = PyDict::new(py);
    for (name, column) in read {
        dict.set_item(name, to_python(py, column)?)?;
    }
    Ok(dict)
}
