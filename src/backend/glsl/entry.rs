use crate::{backend::Error, ir::Ty};

use super::{Stage, types::Types};

pub(super) fn emit(
    types: &Types<'_>,
    param: &Ty,
    result: &Ty,
    stage: Stage,
) -> Result<String, Error> {
    match stage {
        Stage::Compute => {
            let (index, root) = parameters(types, param);
            if root && types.shape(index) == &Ty::UInt32 && result == &Ty::Unit {
                let arg = argument(
                    types,
                    param,
                    types.wrap(index, "gl_GlobalInvocationID.x".into()),
                );
                return Ok(format!(
                    "layout(local_size_x = 64) in;\n{}\nvoid main() {{ r_entry({arg}); }}\n",
                    root_declaration(root)
                ));
            }
            if types.shape(param) != &Ty::UInt32 || types.shape(result) != &Ty::UInt32 {
                return Err(Error(
                    "compute entry must be (uint, Ptr<T>) -> () or map uint to uint (one RGBA8 pixel per invocation)".into(),
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
            let (index, root) = parameters(types, param);
            if types.shape(index) != &Ty::Int32 {
                return Err(Error("vertex entry must take int or (int, Ptr<T>)".into()));
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
{}
layout(location = 0) out vec4 r_color;
void main() {{
    {} v = r_entry({});
    gl_Position = {position};
    r_color = {color};
}}
",
                root_declaration(root),
                types.name(result),
                argument(types, param, types.wrap(index, "gl_VertexIndex".into()))
            ))
        }
        Stage::Fragment => {
            let (color, root) = parameters(types, param);
            vector(types, color, &["r", "g", "b", "a"])?;
            vector(types, result, &["r", "g", "b", "a"])?;
            let input = types.wrap(
                color,
                format!(
                    "{}(r_color.r, r_color.g, r_color.b, r_color.a)",
                    types.name(types.shape(color))
                ),
            );
            let input = argument(types, param, input);
            let output = vector_expr(types, result, "v".into());
            Ok(format!(
                "
{}
layout(location = 0) in vec4 r_color;
layout(location = 0) out vec4 r_output;
void main() {{
    {} v = r_entry({input});
    r_output = {output};
}}
",
                root_declaration(root),
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

fn parameters<'a>(types: &'a Types<'_>, param: &'a Ty) -> (&'a Ty, bool) {
    if let Ty::Record { fields } = types.shape(param)
        && let [input, root] = fields.as_slice()
        && input.name.as_ref() == "_0"
        && root.name.as_ref() == "_1"
        && matches!(root.ty, Ty::Pointer { .. })
    {
        return (&input.ty, true);
    }
    (param, false)
}

fn argument(types: &Types<'_>, param: &Ty, input: String) -> String {
    if parameters(types, param).1 {
        types.wrap(
            param,
            format!("{}({input}, r_root)", types.name(types.shape(param))),
        )
    } else {
        input
    }
}

fn root_declaration(root: bool) -> &'static str {
    if root {
        "layout(push_constant) uniform Push { uint64_t r_root; };"
    } else {
        ""
    }
}
