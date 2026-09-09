use resin_common::prelude::*;
use std::{collections::HashSet, fmt::Write};

use crate::Error;
use resin_lir::Module;

pub(super) struct Types<'a> {
    pub table: &'a TypeTable,
    pub module: &'a Module,
    records: Vec<Ty>,
    seen: HashSet<Ty>,
    buffers: Vec<Ty>,
}

impl<'a> Types<'a> {
    pub fn new(module: &'a Module, table: &'a TypeTable) -> Self {
        Self {
            module,
            table,
            records: Vec::new(),
            seen: HashSet::new(),
            buffers: Vec::new(),
        }
    }

    pub fn tag(&self, case: &Case) -> u32 {
        case.tag(self.table)
    }

    pub fn register(&mut self, ty: &Ty) -> Result<(), Error> {
        if !self.seen.insert(ty.clone()) {
            return Ok(());
        }
        match ty {
            Ty::Union { .. } | Ty::Result { .. } => {
                for (_, payload) in ty.payloads().unwrap() {
                    self.register(&payload)?;
                }
                self.records.push(ty.clone());
            }
            Ty::Arc { .. }
            | Ty::Weak { .. }
            | Ty::Unit
            | Ty::None
            | Ty::Bool
            | Ty::Int32
            | Ty::UInt8
            | Ty::UInt32
            | Ty::UInt64
            | Ty::Float32 => {}
            Ty::Span { element } => {
                self.buffer(element)?;
                self.records.push(ty.clone());
            }
            Ty::Array { element, length } => {
                if *length == 0 {
                    return Err(Error("shader arrays must not be empty".into()));
                }
                self.register(element)?;
                self.records.push(ty.clone());
            }
            Ty::Pointer { pointee } => {
                self.buffer(pointee)?;
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.register(&field.ty)?;
                }
                self.records.push(ty.clone());
            }
            Ty::Defined { definition } => {
                self.register(self.module.types[definition.index()].body().unwrap())?;
                self.records.push(ty.clone());
            }
            _ => {
                return Err(Error(format!(
                    "shader profile does not support type {ty:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn buffer(&mut self, ty: &Ty) -> Result<String, Error> {
        crate::layout::layout(self.module, ty)?;
        let index = self.table.id(ty).expect("verified buffer type").index();
        if self.buffers.contains(ty) {
            return Ok(format!("r_p{index}"));
        }
        self.buffers.push(ty.clone());
        self.register(ty)?;
        Ok(format!("r_p{index}"))
    }

    pub fn name(&self, ty: &Ty) -> String {
        match ty {
            Ty::UInt8 => "uint8_t".into(),
            Ty::Unit | Ty::None | Ty::UInt32 => "uint".into(),
            Ty::UInt64 | Ty::Pointer { .. } | Ty::Arc { .. } | Ty::Weak { .. } => "uint64_t".into(),
            Ty::Int32 => "int".into(),
            Ty::Bool => "bool".into(),
            Ty::Float32 => "float".into(),
            _ => format!("r_t{}", self.table.id(ty).expect("verified type").index()),
        }
    }

    pub fn shape<'b>(&'b self, mut ty: &'b Ty) -> &'b Ty {
        while let Ty::Defined { definition } = ty {
            ty = self.module.types[definition.index()].body().unwrap();
        }
        ty
    }

    pub fn unwrap<'b>(&'b self, mut ty: &'b Ty, mut value: String) -> String {
        while let Ty::Defined { definition } = ty {
            value = format!("({value}).value");
            ty = self.module.types[definition.index()].body().unwrap();
        }
        value
    }

    pub fn wrap(&self, ty: &Ty, value: String) -> String {
        match ty {
            Ty::Defined { definition } => format!(
                "{}({})",
                self.name(ty),
                self.wrap(self.module.types[definition.index()].body().unwrap(), value)
            ),
            _ => value,
        }
    }

    pub fn zero(&self, ty: &Ty) -> String {
        match ty {
            Ty::Union { .. } | Ty::Result { .. } => {
                let mut values = vec!["0u".into()];
                values.extend(ty.payloads().unwrap().iter().map(|(_, ty)| self.zero(ty)));
                format!("{}({})", self.name(ty), values.join(", "))
            }
            Ty::Span { .. } => format!("{}(uint64_t(0), uint64_t(0))", self.name(ty)),
            Ty::Array { element, length } => format!(
                "{}({}[{}]({}))",
                self.name(ty),
                self.name(element),
                length,
                vec![self.zero(element); *length].join(", ")
            ),
            Ty::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|f| self.zero(&f.ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{}({})",
                    self.name(ty),
                    if fields.is_empty() { "0u" } else { &fields }
                )
            }
            Ty::Defined { definition } => format!(
                "{}({})",
                self.name(ty),
                self.zero(self.module.types[definition.index()].body().unwrap())
            ),
            _ => format!("{}(0)", self.name(ty)),
        }
    }

    pub fn extensions(&self) -> &'static str {
        if self.seen.contains(&Ty::UInt8) {
            "#extension GL_EXT_shader_8bit_storage : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
        } else {
            ""
        }
    }

    pub fn declarations(&self) -> String {
        let mut out: String = self
            .records
            .iter()
            .map(|ty| {
                let fields = match ty {
                    Ty::Union { .. } | Ty::Result { .. } => {
                        let mut fields = String::from("uint tag;");
                        for (case, payload) in ty.payloads().unwrap() {
                            let tag = self.tag(&case);
                            write!(fields, " {} v{tag};", self.name(&payload)).unwrap();
                        }
                        fields
                    }
                    Ty::Defined { definition } => format!(
                        "{} value;",
                        self.name(self.module.types[definition.index()].body().unwrap())
                    ),
                    Ty::Span { .. } => "uint64_t f0; uint64_t f1;".into(),
                    Ty::Array { element, length } => {
                        format!("{} items[{length}];", self.name(element))
                    }
                    Ty::Record { fields } if fields.is_empty() => "uint empty;".into(),
                    Ty::Record { fields } => fields
                        .iter()
                        .enumerate()
                        .map(|(i, f)| format!("{} f{i};", self.name(&f.ty)))
                        .collect::<Vec<_>>()
                        .join(" "),
                    _ => unreachable!(),
                };
                format!("struct {} {{ {fields} }};\n", self.name(ty))
            })
            .collect();
        for ty in &self.buffers {
            let index = self.table.id(ty).expect("verified buffer type").index();
            let layout = crate::layout::layout(self.module, ty).unwrap();
            writeln!(out, "layout(buffer_reference, std430, buffer_reference_align = {}) buffer r_p{index} {{ {} value; }};", layout.align, self.name(ty)).unwrap();
        }
        out
    }
}
