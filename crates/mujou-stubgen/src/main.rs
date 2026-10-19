//! Writes the type stub of the native extension module from the introspection
//! data that pyo3 puts into the built library (`experimental-inspect`): the
//! signatures and the doc comments of `crates/mujou-python`.
//!
//! Usage: `mujou-stubgen <library> <module> <output.pyi>`, where `<module>` is the
//! full name of the module (such as `mujou._core._native`). Run it through `just stub`.

use std::path::Path;
use std::{env, fs};

use anyhow::{Context, Result, anyhow, ensure};
use pyo3_introspection::model::{Argument, Arguments, Class, Expr, Function, Module};
use pyo3_introspection::{introspect_cdylib, module_stub_files};

/// The exceptions made with `create_exception!`, which pyo3 does not
/// introspect: `(name, base, docstring)`. Keep them as in
/// `crates/mujou-python/src/runtime.rs`; `tests/mujou/_core/test_native_stub.py` checks
/// that the stub matches the module.
const EXCEPTIONS: &[(&str, &str, &str)] = &[
    (
        "NetworkError",
        "RuntimeError",
        "The network stopped because a node raised. `__cause__` is the first exception.",
    ),
    (
        "PauseTimeoutError",
        "TimeoutError",
        "Some nodes did not pause in time, so the state was not taken. The network keeps running.",
    ),
];

fn add_exceptions(module: &mut Module) {
    for (name, base, doc) in EXCEPTIONS {
        module.classes.push(Class {
            name: (*name).to_owned(),
            bases: vec![Expr::Name {
                id: (*base).to_owned(),
            }],
            methods: Vec::new(),
            attributes: Vec::new(),
            decorators: Vec::new(),
            inner_classes: Vec::new(),
            docstring: Some((*doc).to_owned()),
        });
    }
    module.classes.sort_by(|a, b| a.name.cmp(&b.name));
    // The exceptions were the members it could not find.
    module.incomplete = false;
}

fn typing(name: &str) -> Expr {
    Expr::Attribute {
        value: Box::new(Expr::Name {
            id: "typing".to_owned(),
        }),
        attr: name.to_owned(),
    }
}

/// Fixes the methods where pyo3's types are too loose or missing:
///
/// - every `__new__` returns `typing.Self`: pyo3 writes the class itself, which
///   would make a type checker take `Sub()` for an instance of the base
/// - `*args` and `**kwargs` without an annotation take `typing.Any`, as a
///   signature can only annotate them with a string of an unimported name
/// - the overrides of `object`'s methods are marked with `typing.override`
///   (PEP 698), which pyo3 leaves out
/// - the ordering operators and `__sub__` take the class itself
/// - a class without a constructor gets `__new__(cls) -> typing.NoReturn`:
///   calling it raises `TypeError`, but a type checker would otherwise take
///   the constructor of `object` or of a base class
fn fix_methods(module: &mut Module) {
    for class in &mut module.classes {
        // pyo3's `__new__` returns the class, written as the stub refers to it.
        let own_type = class
            .methods
            .iter()
            .find(|m| m.name == "__new__")
            .and_then(|m| m.returns.clone());
        if own_type.is_none() {
            class.methods.push(no_constructor());
        }
        let object = Expr::Name {
            id: "object".to_owned(),
        };
        for method in &mut class.methods {
            if let Some(own_type) = &own_type {
                if method.name == "__new__" {
                    method.returns = Some(typing("Self"));
                }
                if SAME_CLASS_OPERATORS.contains(&method.name.as_str()) {
                    let arguments = &mut method.arguments;
                    for operand in arguments
                        .positional_only_arguments
                        .iter_mut()
                        .chain(arguments.arguments.iter_mut())
                        .filter(|a| a.annotation.as_ref() == Some(&object))
                    {
                        operand.annotation = Some(own_type.clone());
                    }
                }
            }
            let arguments = &mut method.arguments;
            for variadic in [&mut arguments.vararg, &mut arguments.kwarg]
                .into_iter()
                .flatten()
            {
                variadic.annotation.get_or_insert_with(|| typing("Any"));
            }
            if OBJECT_METHODS.contains(&method.name.as_str()) {
                method.decorators.push(typing("override"));
            }
        }
    }
}

/// The binary operators whose operand pyo3 types as `object`: it returns
/// `NotImplemented` for any operand it cannot extract, so in this module (where
/// every such operand is the class itself) only the class works.
const SAME_CLASS_OPERATORS: &[&str] = &["__lt__", "__le__", "__gt__", "__ge__", "__sub__"];

/// The methods of `object` that a class may override.
const OBJECT_METHODS: &[&str] = &[
    "__eq__",
    "__ne__",
    "__hash__",
    "__reduce__",
    "__repr__",
    "__str__",
];

fn no_constructor() -> Function {
    Function {
        name: "__new__".to_owned(),
        decorators: Vec::new(),
        arguments: Arguments {
            positional_only_arguments: vec![Argument {
                name: "cls".to_owned(),
                default_value: None,
                annotation: None,
            }],
            arguments: Vec::new(),
            vararg: None,
            keyword_only_arguments: Vec::new(),
            kwarg: None,
        },
        returns: Some(typing("NoReturn")),
        is_async: false,
        docstring: Some("No constructor: calling the class raises `TypeError`.".to_owned()),
    }
}

fn main() -> Result<()> {
    let [_, library, module, output] = <[String; 4]>::try_from(env::args().collect::<Vec<_>>())
        .map_err(|_| anyhow!("usage: mujou-stubgen <library> <module> <output.pyi>"))?;
    // The introspection data names the module by its last component.
    let name = module.rsplit('.').next().unwrap_or(&module);
    let mut introspected = introspect_cdylib(&library, name)
        .with_context(|| format!("failed to introspect {module} in {library}"))?;
    ensure!(
        introspected.modules.is_empty(),
        "{module} has submodules; write one stub per module"
    );
    // Before adding the exceptions, which take the constructor of their base.
    fix_methods(&mut introspected);
    add_exceptions(&mut introspected);
    let stubs = module_stub_files(&introspected);
    let stub = stubs
        .get(Path::new("__init__.pyi"))
        .ok_or_else(|| anyhow!("no stub was generated for {module}"))?;
    fs::write(&output, stub).with_context(|| format!("failed to write {output}"))?;
    Ok(())
}
