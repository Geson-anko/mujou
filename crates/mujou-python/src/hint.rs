//! Arguments and results typed for the stub where pyo3 cannot derive the type
//! from the Rust one. A string annotation in `#[pyo3(signature = ...)]` cannot
//! name what the stub does not import, such as `collections.abc.Callable`.

use std::convert::Infallible;

use pyo3::Borrowed;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;

const ANY: PyStaticExpr = PyStaticExpr::Attribute {
    value: &PyStaticExpr::Name { id: "typing" },
    attr: "Any",
};

/// The values of one message: `Sequence[Any]` (tensors and Python scalars). Not
/// checked here: `borrow_values` rejects what it cannot read. The stub cannot
/// say `SupportsDLPack`, which only a hand-written stub could define.
pub struct Values<'py>(pub Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for Values<'py> {
    type Error = Infallible;
    const INPUT_TYPE: PyStaticExpr = PyStaticExpr::Subscript {
        value: &PyStaticExpr::Attribute {
            value: &PyStaticExpr::Name {
                id: "collections.abc",
            },
            attr: "Sequence",
        },
        slice: &ANY,
    };

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self(obj.to_owned()))
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
