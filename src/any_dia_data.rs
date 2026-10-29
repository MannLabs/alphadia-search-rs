//! Lets the Python entry points take either kind of DIA data.

use pyo3::prelude::*;

use crate::dia_data::DIAData;
use crate::tims_dia_data::TimsDIAData;

/// DIA data passed from Python: Thermo-style `DIAData` or timsTOF `TimsDIAData`.
#[derive(FromPyObject)]
pub enum AnyDIAData<'py> {
    Dia(PyRef<'py, DIAData>),
    Tims(PyRef<'py, TimsDIAData>),
}

/// Evaluates `$body` with `$data` bound to the concrete DIA data, so generic code is
/// monomorphized for each kind. `$body` is an expression, not a closure: `return` and `?`
/// in it leave the enclosing function.
macro_rules! with_dia_data {
    ($any:expr, |$data:ident| $body:expr) => {
        match $any {
            $crate::any_dia_data::AnyDIAData::Dia(ref data) => {
                let $data = &**data;
                $body
            }
            $crate::any_dia_data::AnyDIAData::Tims(ref data) => {
                let $data = &**data;
                $body
            }
        }
    };
}

pub(crate) use with_dia_data;
