//! Schemas: the named tensors a message is made of, and how they are printed.

use std::collections::HashSet;
use std::fmt;

use crate::dlpack::ffi::DLDataType;

/// One named tensor of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DLDataType,
}

impl FieldSpec {
    pub(crate) fn elem_bytes(&self) -> usize {
        usize::from(self.dtype.bits / 8) * usize::from(self.dtype.lanes)
    }

    /// The size of the field as compact row-major bytes. Cannot overflow for the
    /// fields of a [`Schema`], which checks it.
    pub(crate) fn nbytes(&self) -> usize {
        self.shape.iter().product::<i64>() as usize * self.elem_bytes()
    }

    /// `nbytes`, or `None` if it exceeds `isize::MAX`, the most one allocation
    /// (and a pointer offset within it) can span.
    fn checked_nbytes(&self) -> Option<usize> {
        let nbytes = self.shape.iter().try_fold(self.elem_bytes(), |acc, &n| {
            acc.checked_mul(usize::try_from(n).ok()?)
        })?;
        (nbytes <= isize::MAX as usize).then_some(nbytes)
    }
}

/// The fields of a message, fixed when the network is built: the fields of its
/// value, then the fields of the data it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// The value fields, then the carry fields.
    fields: Vec<FieldSpec>,
    /// Where the carry fields start in `fields`.
    carry_from: usize,
}

impl Schema {
    /// Checks that there is at least one value field (and any number of carry
    /// fields), the names are distinct among the value fields and among the
    /// carry fields, every extent is at least 1 (a 0-dimensional shape is fine),
    /// every dtype is a scalar of whole bytes, and every field fits in one
    /// allocation.
    pub fn new(values: Vec<FieldSpec>, carry: Vec<FieldSpec>) -> Result<Self, SchemaError> {
        if values.is_empty() {
            return Err(SchemaError("a schema needs at least one field".to_owned()));
        }
        check_fields(&values, "field")?;
        check_fields(&carry, "carry field")?;
        let carry_from = values.len();
        let mut fields = values;
        fields.extend(carry);
        Ok(Self { fields, carry_from })
    }

    /// Every field: the value fields, then the carry fields.
    pub fn fields(&self) -> &[FieldSpec] {
        &self.fields
    }

    /// The fields of the value.
    pub fn values(&self) -> &[FieldSpec] {
        &self.fields[..self.carry_from]
    }

    /// The fields of the carried data, possibly none.
    pub fn carry(&self) -> &[FieldSpec] {
        &self.fields[self.carry_from..]
    }
}

/// The checks of [`Schema::new`] on one list of fields, which `what` names.
fn check_fields(fields: &[FieldSpec], what: &str) -> Result<(), SchemaError> {
    let mut names = HashSet::new();
    for f in fields {
        let invalid = |reason: String| Err(SchemaError(format!("{what} '{}': {reason}", f.name)));
        if !names.insert(f.name.as_str()) {
            return Err(SchemaError(format!("duplicate {what} '{}'", f.name)));
        }
        if f.shape.iter().any(|&n| n < 1) {
            return invalid(format!(
                "every extent must be >= 1, got shape {}",
                format_shape(&f.shape)
            ));
        }
        let DLDataType { bits, lanes, .. } = f.dtype;
        if lanes != 1 || bits == 0 || bits % 8 != 0 {
            return invalid(format!("unsupported dtype {}", format_dtype(f.dtype)));
        }
        if f.checked_nbytes().is_none() {
            return invalid(format!(
                "shape {} of {} takes more than isize::MAX bytes",
                format_shape(&f.shape),
                format_dtype(f.dtype)
            ));
        }
    }
    Ok(())
}

/// Why [`Schema::new`] refused its fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError(String);

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid schema: {}", self.0)
    }
}

impl std::error::Error for SchemaError {}

/// Why a tensor cannot be the value of a field. It does not name the field,
/// which the caller knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldError {
    /// The tensor does not fit the field. `what` is `"shape"` or `"dtype"`.
    Mismatch {
        what: &'static str,
        expected: String,
        actual: String,
    },
    /// The tensor is not in host memory.
    UnsupportedDevice { device_type: i32 },
    /// The producer handed over something that is not a valid DLPack tensor.
    InvalidDLPack { reason: String },
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch {
                what,
                expected,
                actual,
            } => write!(f, "expected {what} {expected}, got {actual}"),
            Self::UnsupportedDevice { device_type } => write!(
                f,
                "expected a tensor in host memory (DLPack device type 1 or 3), \
                 got device type {device_type}; move it to the CPU first"
            ),
            Self::InvalidDLPack { reason } => write!(f, "invalid DLPack tensor: {reason}"),
        }
    }
}

impl std::error::Error for FieldError {}

/// `(2, 3)`, `(5,)` or `()`, as Python prints a shape.
pub(crate) fn format_shape(shape: &[i64]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(i64::to_string).collect();
            format!("({})", dims.join(", "))
        }
    }
}

/// The name torch and NumPy use for the scalar dtypes, so that error messages
/// read naturally; the raw code otherwise.
pub(crate) fn format_dtype(dtype: DLDataType) -> String {
    let DLDataType { code, bits, lanes } = dtype;
    match (code, lanes) {
        (0, 1) => format!("int{bits}"),
        (1, 1) => format!("uint{bits}"),
        (2, 1) => format!("float{bits}"),
        (4, 1) => format!("bfloat{bits}"),
        (6, 1) if bits == 8 => "bool".to_owned(),
        _ => format!("DLDataType(code={code}, bits={bits}, lanes={lanes})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I64: DLDataType = DLDataType {
        code: 0,
        bits: 64,
        lanes: 1,
    };

    fn field(name: &str, shape: &[i64]) -> FieldSpec {
        FieldSpec {
            name: name.to_owned(),
            shape: shape.to_vec(),
            dtype: I64,
        }
    }

    #[test]
    fn schema_rejects_invalid_fields() {
        let bool8 = DLDataType {
            code: 6,
            bits: 8,
            lanes: 1,
        };
        let values = |fields: Vec<FieldSpec>| Schema::new(fields, vec![]);
        assert!(values(vec![]).is_err());
        assert!(values(vec![field("a", &[]), field("a", &[2])]).is_err());
        assert!(values(vec![field("a", &[2, 0])]).is_err());
        // 2^61 int64 elements are 2^64 bytes, and 2^32 x 2^32 overflows usize.
        assert!(values(vec![field("a", &[1 << 61])]).is_err());
        assert!(values(vec![field("a", &[1 << 32, 1 << 32])]).is_err());
        let mut f = field("a", &[2]);
        f.dtype.lanes = 4;
        assert!(values(vec![f.clone()]).is_err());
        let ok = values(vec![FieldSpec {
            name: "m".to_owned(),
            shape: vec![3],
            dtype: bool8,
        }]);
        assert_eq!(ok.unwrap().fields()[0].nbytes(), 3);

        // The carry fields are checked alike, and named apart from the values.
        let carry = |fields: Vec<FieldSpec>| Schema::new(vec![field("x", &[])], fields);
        assert!(carry(vec![field("y", &[]), field("y", &[2])]).is_err());
        assert!(carry(vec![field("y", &[0])]).is_err());
        assert!(carry(vec![f]).is_err());
        let both = Schema::new(vec![field("", &[])], vec![field("", &[2])]).unwrap();
        assert_eq!((both.values().len(), both.carry().len()), (1, 1));
        assert!(Schema::new(vec![], vec![field("y", &[])]).is_err());
    }
}
