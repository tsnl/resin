use crate::ir::{Ty, shader::Interface};

use super::types::Types;

pub(super) fn emit(types: &Types<'_>, param: &Ty, result: &Ty, interface: &Interface) -> String {
    match interface {
        Interface::Compute { index } => {
            let arg = argument(
                types,
                param,
                true,
                types.wrap(index, "gl_GlobalInvocationID.x".into()),
            );
            format!(
                "layout(local_size_x = 64) in;\n{}\nvoid main() {{ r_entry({arg}); }}\n",
                root_declaration(true)
            )
        }
        Interface::Vertex {
            index,
            root,
            position,
            color,
        } => {
            let value = types.unwrap(result, "v".into());
            let position = vector_expr(types, position, format!("({value}).f0"));
            let color = vector_expr(types, color, format!("({value}).f1"));
            format!(
                "
{}
layout(location = 0) out vec4 r_color;
void main() {{
    {} v = r_entry({});
    if (r_failed) return;
    gl_Position = {position};
    r_color = {color};
}}
",
                root_declaration(*root),
                types.name(result),
                argument(
                    types,
                    param,
                    *root,
                    types.wrap(index, "gl_VertexIndex".into())
                )
            )
        }
        Interface::Fragment { color, root } => {
            let input = types.wrap(
                color,
                format!(
                    "{}(r_color.r, r_color.g, r_color.b, r_color.a)",
                    types.name(types.shape(color))
                ),
            );
            let input = argument(types, param, *root, input);
            let output = vector_expr(types, result, "v".into());
            format!(
                "
{}
layout(location = 0) in vec4 r_color;
layout(location = 0) out vec4 r_output;
void main() {{
    {} v = r_entry({input});
    if (r_failed) return;
    r_output = {output};
}}
",
                root_declaration(*root),
                types.name(result)
            )
        }
    }
}

fn vector_expr(types: &Types<'_>, ty: &Ty, expr: String) -> String {
    let expr = types.unwrap(ty, expr);
    format!("vec4(({expr}).f0, ({expr}).f1, ({expr}).f2, ({expr}).f3)")
}

fn argument(types: &Types<'_>, param: &Ty, root: bool, input: String) -> String {
    if root {
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
