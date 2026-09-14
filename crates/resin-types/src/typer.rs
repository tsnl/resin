//! Concrete checking, explicit conversions, and shader signatures.
use crate::prelude::*;
use crate::{shader, types};
use std::sync::Arc;

//
// Primitive operation signatures
//

pub(super) fn lookup(name: &str, arity: usize) -> Result<BuiltinRule, TypeError> {
    let (rule, valid_arity) = match name {
        "+" | "-" => (BuiltinRule::Arithmetic, matches!(arity, 1 | 2)),
        "~" => (BuiltinRule::Arithmetic, arity == 1),
        "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" => (BuiltinRule::Arithmetic, arity == 2),
        "==" | "!=" | "<" | "<=" | ">" | ">=" => (BuiltinRule::Comparison, arity == 2),
        "!" => (BuiltinRule::Boolean, arity == 1),
        "&&" | "||" => (BuiltinRule::Boolean, arity == 2),
        "format_bytes" => (BuiltinRule::Format, arity == 3),
        "string_from_bytes" => (BuiltinRule::StringFromBytes, arity == 2),
        _ => {
            return Err(TypeError::new(TypeErrorKind::UnknownBuiltin {
                name: name.into(),
            }));
        }
    };
    if !valid_arity {
        return Err(TypeError::new(TypeErrorKind::InvalidBuiltinArgumentCount {
            name: name.into(),
            found: arity,
        }));
    }
    Ok(rule)
}

//
// Concrete type checking
//

pub(super) fn same(expected: &Ty, found: &Ty) -> Result<(), TypeError> {
    if expected == found {
        Ok(())
    } else {
        Err(TypeError::new(TypeErrorKind::TypeMismatch {
            expected: expected.clone(),
            found: found.clone(),
        }))
    }
}

pub(super) fn ascribe(context: &TyperContext, from: &Ty, to: &Ty) -> Result<Vec<Conv>, TypeError> {
    ascription(context.definitions(), from, to)?.ok_or_else(|| {
        TypeError::new(TypeErrorKind::TypeMismatch {
            expected: to.clone(),
            found: from.clone(),
        })
    })
}

pub(super) fn explicit_conversion(
    context: &TyperContext,
    from: &Ty,
    to: &Ty,
) -> Result<ExplicitConversion, TypeError> {
    if from != to {
        if from.widens_to(to) {
            return Ok(ExplicitConversion::Widen);
        }
        if from.is_numeric() && to.is_numeric() {
            return Ok(ExplicitConversion::NumericCast);
        }
        if from.pointer_cast(to) {
            return Ok(ExplicitConversion::PointerCast);
        }
    }
    let steps = context.ascribe(from, to)?;
    for step in &steps {
        if let Conv::Unwrap { definition } = step
            && context.definition(*definition)?.drop_hook().is_some()
        {
            return Err(TypeError::new(TypeErrorKind::UnwrapManaged {
                definition: *definition,
            }));
        }
    }
    Ok(ExplicitConversion::Ascribe(steps))
}

pub(super) fn as_bool(ty: &Ty) -> Result<(), TypeError> {
    if ty == &Ty::Bool {
        Ok(())
    } else {
        Err(TypeError::new(TypeErrorKind::ExpectedBoolean {
            found: ty.clone(),
        }))
    }
}

pub(super) fn as_record(context: &TyperContext, ty: &Ty) -> Result<Converted, TypeError> {
    let mut current = ty.clone();
    let mut steps = Vec::new();
    while let Some(pointee) = current.deref_target() {
        steps.push(Conv::Deref);
        current = pointee.clone();
    }
    if let Ty::Defined { definition } = current {
        steps.push(Conv::Unwrap { definition });
        current = context.definition_body(definition)?.clone();
    }
    if matches!(current, Ty::Record { .. } | Ty::Str) {
        Ok(Converted { ty: current, steps })
    } else {
        Err(TypeError::new(TypeErrorKind::ExpectedRecord {
            found: ty.clone(),
        }))
    }
}

pub(super) fn from_definitions(definitions: impl Into<TypeTable>) -> TyperContext {
    let definitions = definitions.into();
    TyperContext { definitions }
}

pub(super) fn into_definitions(context: TyperContext) -> Result<TypeTable, TypeError> {
    for index in 0..context.definitions.len() {
        if context.definitions[index].name().is_some() {
            context.definition_body(TypeId::from_index(index))?;
        }
    }
    Ok(context.definitions)
}

pub(super) fn create_type(
    context: &mut TyperContext,
    name: impl Into<Arc<str>>,
    body: Ty,
) -> Result<TypeId, TypeError> {
    let definition = context.reserve_type(name);
    if let Err(err) = context.define_type(definition, body) {
        context.definitions.pop_nominal();
        return Err(err);
    }
    Ok(definition)
}

pub(super) fn define_type(
    context: &mut TyperContext,
    definition: TypeId,
    body: Ty,
) -> Result<(), TypeError> {
    if context.definition(definition)?.body().is_some() {
        return Err(TypeError::new(TypeErrorKind::TypeAlreadyDefined {
            definition,
        }));
    }
    types::check_references(&context.definitions, &body)?;
    types::check_layout(&context.definitions, definition, &body)?;
    context.definitions.define(definition, body);
    Ok(())
}

pub(super) fn type_field(
    context: &TyperContext,
    base: &Ty,
    name: &str,
) -> Result<FieldAccess, TypeError> {
    let converted = context.as_record(base)?;
    let shape = converted.ty.view_record().unwrap_or(converted.ty);
    let Ty::Record { fields } = shape else {
        return Err(TypeError::new(TypeErrorKind::ExpectedRecord {
            found: base.clone(),
        }));
    };
    for (index, field) in fields.into_iter().enumerate() {
        if field.name.as_ref() == name {
            return Ok(FieldAccess {
                ty: field.ty,
                index,
                steps: converted.steps,
            });
        }
    }
    Err(TypeError::new(TypeErrorKind::UnknownField {
        name: Arc::from(name),
    }))
}

pub(super) fn type_builtin_call(
    context: &TyperContext,
    name: &str,
    args: &[Ty],
) -> Result<BuiltinCall, TypeError> {
    let rule = BuiltinRule::lookup(name, args.len())?;
    let result = match rule {
        BuiltinRule::Format => {
            byte_parameters(context, args)?;
            context.type_format(&args[2])?
        }
        BuiltinRule::StringFromBytes => {
            byte_parameters(context, args)?;
            Ty::StrongOwner
        }
        BuiltinRule::Boolean => {
            for arg in args {
                context.as_bool(arg)?;
            }
            Ty::Bool
        }
        BuiltinRule::Arithmetic | BuiltinRule::Comparison => {
            if rule == BuiltinRule::Arithmetic
                && args.iter().any(|ty| matches!(ty, Ty::Pointer { .. }))
            {
                return Err(TypeError::new(TypeErrorKind::PointerArithmetic));
            }
            for arg in &args[1..] {
                context.same(&args[0], arg)?;
            }
            if rule == BuiltinRule::Comparison {
                Ty::Bool
            } else {
                args[0].clone()
            }
        }
    };
    Ok(BuiltinCall {
        params: args.to_vec(),
        result,
    })
}

pub(super) fn builtin_instance(
    context: &TyperContext,
    name: &str,
    args: &[Ty],
) -> Result<BuiltinCall, TypeError> {
    let signature = type_builtin_call(context, name, args)?;
    let rule = BuiltinRule::lookup(name, args.len())?;
    if matches!(rule, BuiltinRule::Arithmetic | BuiltinRule::Comparison) {
        let operand = context.body(&args[0])?;
        let supported = match name {
            "==" | "!=" => {
                operand.is_numeric() || matches!(operand, Ty::Bool | Ty::Type | Ty::Pointer { .. })
            }
            "~" | "%" | "<<" | ">>" | "&" | "|" | "^" => operand.is_integer(),
            _ => operand.is_numeric(),
        };
        if !supported {
            return Err(TypeError::new(TypeErrorKind::UnsupportedBuiltin {
                name: name.into(),
                operand,
            }));
        }
    }
    Ok(signature)
}

impl TyperContext {
    pub(super) fn definition_body(&self, definition: TypeId) -> Result<&Ty, TypeError> {
        Ok(types::body(&self.definitions, definition)?)
    }
}

impl TyperContext {
    fn type_format(&self, values: &Ty) -> Result<Ty, TypeError> {
        let invalid = || {
            TypeError::new(TypeErrorKind::InvalidFormatArguments {
                found: values.clone(),
            })
        };
        let fields = match values {
            Ty::Unit => &[][..],
            Ty::Record { fields } => fields,
            _ => return Err(invalid()),
        };
        for (i, field) in fields.iter().enumerate() {
            if field.name.as_ref() != format!("_{i}") {
                return Err(invalid());
            }
            let ty = &field.ty;
            if !(ty.is_numeric()
                || matches!(ty, Ty::Bool | Ty::Unit | Ty::Pointer { .. })
                || *ty == Ty::Str
                || format_byte_record(ty))
            {
                return Err(TypeError::new(TypeErrorKind::UnformattableType {
                    found: ty.clone(),
                }));
            }
        }
        Ok(Ty::StrongOwner)
    }
}

//
// Explicit representation conversions
//

pub(super) fn ascription(
    table: &[TypeDef],
    from: &Ty,
    to: &Ty,
) -> Result<Option<Vec<Conv>>, TypeError> {
    let step = if from == to {
        return Ok(Some(Vec::new()));
    } else if from.view_record().as_ref() == Some(to) {
        Conv::ViewRecord
    } else if let Ty::Defined { definition } = to
        && from == types::body(table, *definition)?
    {
        Conv::Wrap {
            definition: *definition,
        }
    } else if let Ty::Defined { definition } = from
        && to == types::body(table, *definition)?
    {
        Conv::Unwrap {
            definition: *definition,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(vec![step]))
}

//
// Shader entry signatures
//

pub(super) fn validate_shader(
    typer: &TyperContext,
    parameters: &[Ty],
    result: &Ty,
    foreign: bool,
    stage: &str,
) -> Result<shader::Interface, String> {
    fn vector(typer: &TyperContext, ty: &Ty, names: &[&str]) -> bool {
        matches!(shader_shape(typer, ty), Ok(Ty::Record { fields }) if fields.len() == names.len()
                && fields.iter().zip(names).all(|(field, name)| field.name.as_ref() == *name && field.ty == Ty::Float32))
    }
    if foreign {
        return Err("foreign functions cannot be shader entries".into());
    }
    let (input, root) = match parameters {
        [input] => (input, false),
        [input, Ty::Pointer { .. }] => (input, true),
        _ => {
            return Err(format!(
                "invalid @{stage}_shader signature: expected one input and an optional Ptr<T> root"
            ));
        }
    };
    let input_shape = shader_shape(typer, input)?;
    let result = shader_shape(typer, result)?;
    let interface = match stage {
        "compute" if root && input_shape == Ty::UInt64 && result == Ty::Unit => {
            Some(shader::Interface::Compute {
                index: input.clone(),
            })
        }
        "vertex" if input_shape == Ty::Int32 => match &result {
            Ty::Record { fields }
                if fields.len() == 2
                    && fields[0].name.as_ref() == "position"
                    && fields[1].name.as_ref() == "color"
                    && vector(typer, &fields[0].ty, &["x", "y", "z", "w"])
                    && vector(typer, &fields[1].ty, &["r", "g", "b", "a"]) =>
            {
                Some(shader::Interface::Vertex {
                    index: input.clone(),
                    root,
                    position: fields[0].ty.clone(),
                    color: fields[1].ty.clone(),
                })
            }
            _ => None,
        },
        "fragment"
            if vector(typer, input, &["r", "g", "b", "a"])
                && vector(typer, &result, &["r", "g", "b", "a"]) =>
        {
            Some(shader::Interface::Fragment {
                color: input.clone(),
                root,
            })
        }
        _ => None,
    };
    if let Some(interface) = interface {
        Ok(interface)
    } else {
        Err(format!(
            "invalid @{stage}_shader signature: {}",
            match stage {
                "compute" => "expected (ulong, Ptr<T>) -> ()",
                "vertex" => "expected int or (int, Ptr<T>) returning a position/color record",
                "fragment" =>
                    "expected Color or (Color, Ptr<T>) returning Color with float32 r/g/b/a fields",
                _ => "unknown shader stage",
            }
        ))
    }
}

fn shader_shape(typer: &TyperContext, ty: &Ty) -> Result<Ty, String> {
    let mut ty = ty.clone();
    while matches!(ty, Ty::Defined { .. }) {
        ty = typer.body(&ty).map_err(|error| error.to_string())?;
    }
    Ok(ty)
}

pub(super) fn pipeline_root(
    typer: &TyperContext,
    stages: &[(&[Ty], &Ty, &str)],
) -> Result<Ty, String> {
    let root =
        match stages {
            [(parameter, result, "compute")] => {
                validate_shader(typer, parameter, result, false, "compute")?;
                Some(shader_root(parameter)?)
            }
            [
                (vertex_parameter, vertex_result, "vertex"),
                (fragment_parameter, fragment_result, "fragment"),
            ] => graphics_root(
                typer,
                vertex_parameter,
                vertex_result,
                fragment_parameter,
                fragment_result,
            )?,
            _ => return Err(
                "pipeline creation requires one compute shader or a vertex/fragment shader pair"
                    .into(),
            ),
        };
    Ok(root.unwrap_or(Ty::None))
}

fn shader_root(parameters: &[Ty]) -> Result<Ty, String> {
    let Some(Ty::Pointer { pointee }) = parameters.get(1) else {
        return Err("shader root parameter must be a pointer".into());
    };
    Ok(*pointee.clone())
}

fn graphics_root(
    typer: &TyperContext,
    vertex_parameter: &[Ty],
    vertex_result: &Ty,
    fragment_parameter: &[Ty],
    fragment_result: &Ty,
) -> Result<Option<Ty>, String> {
    let shader::Interface::Vertex {
        root: vertex_root,
        color: vertex_color,
        ..
    } = validate_shader(typer, vertex_parameter, vertex_result, false, "vertex")?
    else {
        unreachable!("validated vertex shader")
    };
    let shader::Interface::Fragment {
        root: fragment_root,
        color: fragment_color,
    } = validate_shader(
        typer,
        fragment_parameter,
        fragment_result,
        false,
        "fragment",
    )?
    else {
        unreachable!("validated fragment shader")
    };
    if vertex_color != fragment_color {
        return Err("vertex shader color and fragment shader input must have the same type".into());
    }
    let vertex_root = vertex_root
        .then(|| shader_root(vertex_parameter))
        .transpose()?;
    let fragment_root = fragment_root
        .then(|| shader_root(fragment_parameter))
        .transpose()?;
    match (vertex_root, fragment_root) {
        (Some(vertex), Some(fragment)) if vertex != fragment => {
            Err("vertex and fragment shaders must use the same root type".into())
        }
        (Some(root), _) | (_, Some(root)) => Ok(Some(root)),
        (None, None) => Ok(None),
    }
}

pub(super) fn shader_builtin_instance(
    context: &TyperContext,
    name: &str,
    arguments: &[Ty],
) -> Result<BuiltinCall, String> {
    if matches!(name, "format_bytes" | "string_from_bytes") {
        return Err(format!("{name} is only supported in host programs"));
    }
    let signature =
        builtin_instance(context, name, arguments).map_err(|error| error.to_string())?;
    let operand = context
        .body(&arguments[0])
        .map_err(|error| error.to_string())?;
    let supported = match (name, arguments.len()) {
        ("+" | "-" | "~" | "!", 1) => true,
        (
            "+" | "-" | "*" | "&" | "|" | "^" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||",
            2,
        ) => true,
        ("/", 2) => operand == Ty::Float32,
        _ => false,
    };
    if !supported || operand == Ty::Type {
        return Err(format!(
            "unsupported shader builtin {name:?} for {operand:?}"
        ));
    }
    shader_value_type(context.definitions(), &operand)?;
    Ok(signature)
}

pub(super) fn shader_value_type(definitions: &[TypeDef], ty: &Ty) -> Result<(), String> {
    let mut pending = vec![ty];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(ty) = pending.pop() {
        if !seen.insert(ty) {
            continue;
        }
        match ty {
            Ty::Unit | Ty::None | Ty::Bool | Ty::Int32 | Ty::UInt8 | Ty::UInt32
            | Ty::UInt64 | Ty::Int64 | Ty::Float32 | Ty::StrongOwner | Ty::WeakOwner => {},
            Ty::Str => return Err("shader string literals need device-backed storage; pass a Span<ubyte> in the shader root".into()),
            Ty::GpuPipelineContract | Ty::GpuView | Ty::GpuArguments => {
                return Err("shader cannot consume a managed GPU view or projected arguments".into());
            }
            Ty::Pointer { pointee: element } => {
                crate::layout::layout(definitions, element).map_err(|error| error.to_string())?;
                pending.push(element);
            }
            Ty::Array { element, length } => {
                if *length == 0 { return Err("shader arrays must not be empty".into()); }
                pending.push(element);
            }
            Ty::Record { fields } => pending.extend(fields.iter().map(|field| &field.ty)),
            Ty::Defined { definition } => pending.push(crate::definition_body(definitions, *definition).map_err(|error| error.to_string())?),
            Ty::Union { variants } => pending.extend(variants),
            Ty::Result { value, error } => pending.extend([value.as_ref(), error.as_ref()]),
            _ => return Err(format!("shader profile does not support type {ty:?}")),
        }
    }
    Ok(())
}

fn byte_parameters(context: &TyperContext, args: &[Ty]) -> Result<(), TypeError> {
    context.same(
        &Ty::Pointer {
            pointee: Box::new(Ty::UInt8),
        },
        &args[0],
    )?;
    context.same(&Ty::UInt64, &args[1])
}

// Formatting transports explicit structural byte views, never inferred nominal layouts.
fn format_byte_record(ty: &Ty) -> bool {
    let Ty::Record { fields } = ty else {
        return false;
    };
    matches!(fields.as_slice(), [data, length]
        if data.name.as_ref() == "data" && data.ty == Ty::Pointer { pointee: Box::new(Ty::UInt8) }
        && length.name.as_ref() == "length" && length.ty == Ty::UInt64)
}
