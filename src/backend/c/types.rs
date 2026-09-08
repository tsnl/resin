use std::fmt::Write;

use crate::ir::{Case, Instr, Module, Ty, TypeTable, Value};

mod lifecycle;

pub(super) struct Types<'a> {
    pub table: &'a TypeTable,
    pub module: &'a Module,
    pub shaders: &'a [super::Shader],
    literals: Vec<&'a [u8]>,
}

impl<'a> Types<'a> {
    pub fn new(module: &'a Module, table: &'a TypeTable, shaders: &'a [super::Shader]) -> Self {
        let mut literals = Vec::new();
        for instruction in module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| {
                b.instrs
                    .iter()
                    .take_while(|instruction| !matches!(instruction, Instr::Eliminate { .. }))
            })
        {
            if let Instr::Push { value } = instruction {
                collect_literals(value, &mut literals);
            }
        }
        Self {
            literals,
            module,
            table,
            shaders,
        }
    }

    pub fn literal(&self, bytes: &[u8]) -> String {
        format!(
            "r_literal_{}",
            self.literals
                .iter()
                .position(|value| *value == bytes)
                .expect("collected literal")
        )
    }

    pub fn id(&self, ty: &Ty) -> usize {
        self.table.id(ty).expect("verified type").index()
    }

    pub fn tag(&self, case: &Case) -> u32 {
        case.tag(self.table)
    }

    pub fn name(&self, ty: &Ty) -> String {
        if let Ty::Foreign { name } = ty {
            return name.to_string();
        }
        format!("r_t{}", self.id(ty))
    }

    pub fn shape<'b>(&'b self, mut ty: &'b Ty) -> &'b Ty {
        while let Ty::Defined { definition } = ty {
            ty = self.module.types[definition.index()].body().unwrap();
        }
        ty
    }

    pub fn unwrap(&self, ty: &Ty, mut expr: String) -> String {
        let mut ty = ty;
        while let Ty::Defined { definition } = ty {
            expr = format!("({expr}).value");
            ty = self.module.types[definition.index()].body().unwrap();
        }
        expr
    }

    pub fn wrap(&self, ty: &Ty, expr: String) -> String {
        match ty {
            Ty::Defined { definition } => format!(
                "({}){{ .value = {} }}",
                self.name(ty),
                self.wrap(self.module.types[definition.index()].body().unwrap(), expr)
            ),
            _ => expr,
        }
    }

    pub fn declarations(&self) -> String {
        let mut out = String::new();
        for (index, literal) in self.literals.iter().enumerate() {
            let bytes = literal
                .iter()
                .map(u8::to_string)
                .chain(std::iter::once("0".into()))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "static uint8_t r_literal_{index}[] = {{ {bytes} }};").unwrap();
        }
        for ty in self.table.types() {
            let ty = &ty;
            let name = self.name(ty);
            match ty {
                Ty::Pointer { .. } => {}
                _ => {
                    if let Some(scalar) = scalar(ty) {
                        writeln!(out, "typedef {scalar} {name};").unwrap();
                    } else {
                        writeln!(out, "typedef struct {name} {name};").unwrap();
                    }
                }
            }
        }
        let mut emitted = vec![false; self.table.len()];
        for ty in self.table.types() {
            let ty = &ty;
            self.pointer(ty, &mut emitted, &mut out);
        }
        let mut emitted = vec![false; self.table.len()];
        for ty in self.table.types() {
            let ty = &ty;
            self.definition(ty, &mut emitted, &mut out);
        }
        for ty in self.table.types() {
            let ty = &ty;
            if let Ok(layout) = crate::backend::layout::layout(self.module, ty) {
                let name = self.name(ty);
                writeln!(out, "_Static_assert(sizeof({name}) == {} && _Alignof({name}) == {}, \"host/device layout mismatch\");", layout.size, layout.align).unwrap();
                if matches!(ty, Ty::Record { .. } | Ty::Span { .. }) {
                    for (i, offset) in layout.offsets.iter().enumerate() {
                        writeln!(out, "_Static_assert(offsetof({name}, f{i}) == {offset}, \"host/device field offset mismatch\");").unwrap();
                    }
                }
            }
        }
        out
    }

    fn pointer(&self, ty: &Ty, emitted: &mut [bool], out: &mut String) {
        if emitted[self.id(ty)] {
            return;
        }
        emitted[self.id(ty)] = true;
        if let Ty::Pointer { pointee } = ty {
            self.pointer(pointee, emitted, out);
            writeln!(out, "typedef {} *{};", self.name(pointee), self.name(ty)).unwrap();
        }
    }

    fn definition(&self, ty: &Ty, emitted: &mut [bool], out: &mut String) {
        if emitted[self.id(ty)] {
            return;
        }
        emitted[self.id(ty)] = true;
        let body = match ty {
            Ty::Union { .. } | Ty::Result { .. } => {
                let mut fields = String::new();
                for (case, payload) in ty.payloads().unwrap() {
                    let tag = self.tag(&case);
                    self.definition(&payload, emitted, out);
                    write!(fields, " {} v{tag};", self.name(&payload)).unwrap();
                }
                if fields.is_empty() {
                    "uint32_t tag;".into()
                } else {
                    format!("uint32_t tag; union {{ {fields} }} payload;")
                }
            }
            Ty::Defined { definition } => {
                let body = self.module.types[definition.index()].body().unwrap();
                self.definition(body, emitted, out);
                format!("{} value;", self.name(body))
            }
            Ty::Array { element, length } => {
                self.definition(element, emitted, out);
                // Byte arrays keep a NUL beyond their logical length for C string interop.
                let capacity = if element.as_ref() == &Ty::UInt8 {
                    format!("{length} + 1")
                } else {
                    (*length).max(1).to_string()
                };
                format!("{} items[{capacity}];", self.name(element))
            }
            Ty::Record { fields } => {
                if fields.is_empty() {
                    "uint8_t empty;".into()
                } else {
                    fields
                        .iter()
                        .enumerate()
                        .map(|(index, field)| {
                            self.definition(&field.ty, emitted, out);
                            format!("{} f{index};", self.name(&field.ty))
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            }
            Ty::Span { element } => format!("{} *f0; uint64_t f1;", self.name(element)),
            Ty::Function { param, result } => {
                format!("{} (*call)({});", self.name(result), self.name(param))
            }
            _ => return,
        };
        writeln!(out, "struct {} {{ {body} }};", self.name(ty)).unwrap();
    }
}

fn scalar(ty: &Ty) -> Option<&'static str> {
    Some(match ty {
        Ty::Arc { .. } | Ty::Weak { .. } => "ResinArc *",
        Ty::Unit | Ty::None => "uint8_t",
        Ty::Type => "size_t",
        Ty::Bool => "bool",
        Ty::Int8 => "int8_t",
        Ty::Int16 => "int16_t",
        Ty::Int32 => "int32_t",
        Ty::Int64 => "int64_t",
        Ty::UInt8 => "uint8_t",
        Ty::UInt16 => "uint16_t",
        Ty::UInt32 => "uint32_t",
        Ty::UInt64 => "uint64_t",
        Ty::Float32 => "float",
        Ty::Float64 => "double",
        _ => return None,
    })
}

fn collect_literals<'a>(value: &'a Value, literals: &mut Vec<&'a [u8]>) {
    match value {
        Value::Bytes { value } => {
            if !literals.contains(&value.as_ref()) {
                literals.push(value);
            }
        }
        Value::Array { value } => {
            for value in &value.elements {
                collect_literals(value, literals);
            }
        }
        Value::Record { value } => {
            for field in &value.fields {
                collect_literals(&field.value, literals);
            }
        }
        _ => {}
    }
}
