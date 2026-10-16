//! Arguments and results typed for the stub where pyo3 cannot derive the type
//! from the Rust one. A string annotation in `#[pyo3(signature = ...)]` cannot
//! name what the stub does not import, such as `collections.abc.Callable`, and a
//! getter takes no signature at all.

use std::convert::Infallible;

use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{Borrowed, PyTypeInfo};

use crate::slot::BufferView;

const ANY: PyStaticExpr = PyStaticExpr::Attribute {
    value: &PyStaticExpr::Name { id: "typing" },
    attr: "Any",
};

/// The views of a message: `tuple[BufferView, ...]`.
pub struct Views<'py>(pub Bound<'py, PyTuple>);

impl<'py> IntoPyObject<'py> for Views<'py> {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = Infallible;
    const OUTPUT_TYPE: PyStaticExpr = PyStaticExpr::Subscript {
        value: &PyTuple::TYPE_HINT,
        slice: &PyStaticExpr::Tuple {
            elts: &[
                BufferView::TYPE_HINT,
                PyStaticExpr::Constant {
                    value: PyStaticConstant::Ellipsis,
                },
            ],
        },
    };

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// The `build` of `Runtime.capture`: `Callable[[list[object]], Any]`. Not
/// checked to be callable: calling it raises if it is not.
pub struct Build(pub Py<PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for Build {
    type Error = Infallible;
    const INPUT_TYPE: PyStaticExpr = PyStaticExpr::Subscript {
        value: &PyStaticExpr::Attribute {
            value: &PyStaticExpr::Name {
                id: "collections.abc",
            },
            attr: "Callable",
        },
        slice: &PyStaticExpr::Tuple {
            elts: &[
                PyStaticExpr::List {
                    elts: &[PyStaticExpr::Subscript {
                        value: &PyStaticExpr::Name { id: "list" },
                        slice: &PyStaticExpr::Name { id: "object" },
                    }],
                },
                ANY,
            ],
        },
    };

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self(obj.to_owned().unbind()))
    }
}
