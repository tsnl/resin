use super::Types;
use resin_types::prelude::*;
use std::fmt::Write;

impl Types<'_> {
    pub fn copy(&self, ty: &Ty, value: &str) -> String {
        if ty.needs_drop(&self.module.types) {
            format!("r_copy{}({value})", self.id(ty))
        } else {
            value.into()
        }
    }

    pub fn drop_value(&self, ty: &Ty, value: &str, out: &mut String) {
        if ty.needs_drop(&self.module.types) {
            writeln!(out, "  r_drop{}(&({value}));", self.id(ty)).unwrap();
        }
    }

    pub fn lifecycle(&self) -> String {
        let mut out = String::new();
        for ty in self.table.types() {
            let ty = &ty;
            if matches!(ty, Ty::Foreign { .. }) {
                continue;
            }
            let id = self.id(ty);
            writeln!(
                out,
                "void r_drop{id}(void *raw);\n{} r_copy{id}({} value);",
                self.name(ty),
                self.name(ty)
            )
            .unwrap();
        }
        for ty in self.table.types() {
            let ty = &ty;
            if matches!(ty, Ty::Foreign { .. }) {
                continue;
            }
            let id = self.id(ty);
            writeln!(
                out,
                "{} r_copy{id}({} value) {{",
                self.name(ty),
                self.name(ty)
            )
            .unwrap();
            self.retain_fields(ty, "value", &mut out);
            out.push_str("  return value;\n}\n");
            writeln!(
                out,
                "void r_drop{id}(void *raw) {{ {} *p = raw; (void)p;",
                self.name(ty)
            )
            .unwrap();
            match ty {
                Ty::Arc { .. }
                | Ty::GpuArguments
                | Ty::GpuComputePipeline { .. }
                | Ty::GpuGraphicsPipeline { .. } => out.push_str("  resin_arc_release(*p);\n"),
                Ty::GpuPointer { .. } => out.push_str("  resin_arc_release(p->owner);\n"),
                Ty::GpuSpan { .. } => out.push_str("  resin_arc_release(p->data.owner);\n"),
                Ty::Weak { .. } => out.push_str("  resin_weak_release(*p);\n"),
                Ty::Defined { definition } => {
                    let def = &self.module.types[definition.index()];
                    if let Some(function) = def.drop_hook() {
                        writeln!(out, "  r_fn{}(p);", function.index()).unwrap();
                    }
                    self.drop_value(def.body().unwrap(), "p->value", &mut out);
                }
                Ty::Record { fields } => {
                    for (i, field) in fields.iter().enumerate().rev() {
                        self.drop_value(&field.ty, &format!("p->f{i}"), &mut out);
                    }
                }
                Ty::Array { element, length } if element.needs_drop(&self.module.types) => {
                    writeln!(
                        out,
                        "  for (size_t i = {length}; i > 0; --i) r_drop{}(&p->items[i - 1]);",
                        self.id(element)
                    )
                    .unwrap();
                }
                Ty::Union { .. } | Ty::Result { .. } => {
                    out.push_str("  switch (p->tag) {\n");
                    for (case, payload) in ty.payloads().unwrap() {
                        let tag = self.tag(&case);
                        writeln!(out, "  case {tag}: {{").unwrap();
                        self.drop_value(&payload, &format!("p->payload.v{tag}"), &mut out);
                        out.push_str("    break; }\n");
                    }
                    out.push_str("  default: break;\n  }\n");
                }
                _ => {}
            }
            out.push_str("}\n");
        }
        out
    }

    fn retain_fields(&self, ty: &Ty, value: &str, out: &mut String) {
        match ty {
            Ty::Arc { .. }
            | Ty::GpuArguments
            | Ty::GpuComputePipeline { .. }
            | Ty::GpuGraphicsPipeline { .. } => {
                writeln!(out, "  resin_arc_retain({value});").unwrap();
            }
            Ty::GpuPointer { .. } => {
                writeln!(out, "  resin_arc_retain(({value}).owner);").unwrap();
            }
            Ty::GpuSpan { .. } => {
                writeln!(out, "  resin_arc_retain(({value}).data.owner);").unwrap();
            }
            Ty::Weak { .. } => {
                writeln!(out, "  resin_weak_retain({value});").unwrap();
            }
            Ty::Defined { definition } => self.retain_fields(
                self.module.types[definition.index()].body().unwrap(),
                &format!("({value}).value"),
                out,
            ),
            Ty::Record { fields } => {
                for (i, field) in fields.iter().enumerate() {
                    self.retain_fields(&field.ty, &format!("({value}).f{i}"), out);
                }
            }
            Ty::Array { element, length } if element.needs_drop(&self.module.types) => {
                writeln!(out, "  for (size_t i = 0; i < {length}; ++i) ({value}).items[i] = r_copy{}(({value}).items[i]);", self.id(element)).unwrap();
            }
            Ty::Union { .. } | Ty::Result { .. } => {
                writeln!(out, "  switch (({value}).tag) {{").unwrap();
                for (case, payload) in ty.payloads().unwrap() {
                    let tag = self.tag(&case);
                    writeln!(out, "  case {tag}: {{").unwrap();
                    self.retain_fields(&payload, &format!("({value}).payload.v{tag}"), out);
                    out.push_str("    break; }\n");
                }
                out.push_str("  default: break;\n  }\n");
            }
            _ => {}
        }
    }
}
