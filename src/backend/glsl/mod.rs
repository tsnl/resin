use crate::{
    backend::Error,
    ir::{self, Module, Ty},
};

mod function;
mod types;

use types::Types;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

impl std::str::FromStr for Stage {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        match name {
            "compute" => Ok(Self::Compute),
            "vertex" => Ok(Self::Vertex),
            "fragment" => Ok(Self::Fragment),
            _ => Err(Error("stage must be compute, vertex, or fragment".into())),
        }
    }
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
    pub fn entry(self) -> &'static str {
        match self {
            Self::Compute => "kernel",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

pub fn emit(module: &Module, entry: &str, stage: Stage) -> Result<String, Error> {
    let analysis = ir::verify::analyze(module)?;
    let candidates: Vec<_> = module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name.as_deref() == Some(entry))
        .collect();
    let [(index, function)] = candidates.as_slice() else {
        return Err(Error(format!(
            "expected one shader function named {entry:?}"
        )));
    };
    if !function.nonlocals.is_empty() {
        return Err(Error("shader entry cannot capture values".into()));
    }
    let param = &function.locals[function.param.index()].ty;
    let mut types = Types::new(module);
    for local in &function.locals {
        types.register(&local.ty)?;
    }
    types.register(&function.result)?;
    for ty in analysis[*index].inputs.iter().flatten() {
        if matches!(ty, Ty::Pointer { .. }) {
            return Err(Error(
                "shader profile cannot carry addresses across block edges".into(),
            ));
        }
        types.register(ty)?;
    }
    for ty in analysis[*index].results.iter().flatten().flatten() {
        if let Ty::Pointer { pointee } = ty {
            types.register(pointee)?;
        } else {
            types.register(ty)?;
        }
    }
    let wrapper = wrapper(&types, param, &function.result, stage)?;
    let mut out = "#version 460\n".to_string();
    if stage == Stage::Compute {
        out.push_str("#extension GL_EXT_buffer_reference : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n");
    }
    out.push_str(&types.declarations());
    out.push_str(&function::emit(&types, function, &analysis[*index])?);
    out.push_str(&wrapper);
    Ok(out)
}

fn wrapper(types: &Types<'_>, param: &Ty, result: &Ty, stage: Stage) -> Result<String, Error> {
    match stage {
        Stage::Compute => {
            if types.shape(param) != &Ty::UInt32 || types.shape(result) != &Ty::UInt32 {
                return Err(Error(
                    "compute entry must map uint to uint (one RGBA8 pixel per invocation)".into(),
                ));
            }
            let value = types.unwrap(
                result,
                format!("r_entry({})", types.wrap(param, "index".into())),
            );
            Ok(format!("
layout(local_size_x = 64) in;
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Root {{ uint count; uint64_t pixels; }};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Pixels {{ uint values[]; }};
layout(push_constant) uniform Push {{ uint64_t root; }};
void main() {{
    uint index = gl_GlobalInvocationID.x;
    Root data = Root(root);
    if (index >= data.count) return;
    Pixels(data.pixels).values[index] = {value};
}}
"))
        }
        Stage::Vertex => {
            if types.shape(param) != &Ty::Int32 {
                return Err(Error("vertex entry must take int (vertex index)".into()));
            }
            let Ty::Record { fields } = types.shape(result) else {
                return Err(Error("vertex must return a position/color record".into()));
            };
            if fields.len() != 2
                || fields[0].name.as_ref() != "position"
                || fields[1].name.as_ref() != "color"
            {
                return Err(Error(
                    "vertex result fields must be position, color in that order".into(),
                ));
            }
            vector(types, &fields[0].ty, &["x", "y", "z", "w"])?;
            vector(types, &fields[1].ty, &["r", "g", "b", "a"])?;
            let value = types.unwrap(result, "v".into());
            let position = vector_expr(types, &fields[0].ty, format!("({value}).f0"));
            let color = vector_expr(types, &fields[1].ty, format!("({value}).f1"));
            Ok(format!(
                "
layout(location = 0) out vec4 r_color;
void main() {{
    {} v = r_entry({});
    gl_Position = {position};
    r_color = {color};
}}
",
                types.name(result),
                types.wrap(param, "gl_VertexIndex".into())
            ))
        }
        Stage::Fragment => {
            vector(types, param, &["r", "g", "b", "a"])?;
            vector(types, result, &["r", "g", "b", "a"])?;
            let input = types.wrap(
                param,
                format!(
                    "{}(r_color.r, r_color.g, r_color.b, r_color.a)",
                    types.name(types.shape(param))
                ),
            );
            let output = vector_expr(types, result, "v".into());
            Ok(format!(
                "
layout(location = 0) in vec4 r_color;
layout(location = 0) out vec4 r_output;
void main() {{
    {} v = r_entry({input});
    r_output = {output};
}}
",
                types.name(result)
            ))
        }
    }
}

fn vector(types: &Types<'_>, ty: &Ty, names: &[&str]) -> Result<(), Error> {
    if let Ty::Record { fields } = types.shape(ty)
        && fields.len() == names.len()
        && fields
            .iter()
            .zip(names)
            .all(|(f, n)| f.name.as_ref() == *n && f.ty == Ty::Float32)
    {
        return Ok(());
    }
    Err(Error(format!(
        "shader interface requires float32 fields {names:?}"
    )))
}

fn vector_expr(types: &Types<'_>, ty: &Ty, expr: String) -> String {
    let expr = types.unwrap(ty, expr);
    format!("vec4(({expr}).f0, ({expr}).f1, ({expr}).f2, ({expr}).f3)")
}
