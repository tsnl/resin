use crate::{
    backend::Error,
    ir::{Module, Ty},
};

pub(super) struct Types<'a> {
    pub module: &'a Module,
    records: Vec<Ty>,
}

impl<'a> Types<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self {
            module,
            records: Vec::new(),
        }
    }

    pub fn register(&mut self, ty: &Ty) -> Result<(), Error> {
        match ty {
            Ty::Unit | Ty::Bool | Ty::Int32 | Ty::UInt32 | Ty::Float32 => {}
            Ty::Record { fields } => {
                if self.records.contains(ty) {
                    return Ok(());
                }
                for field in fields {
                    self.register(&field.ty)?;
                }
                self.records.push(ty.clone());
            }
            Ty::Defined { definition } => {
                if self.records.contains(ty) {
                    return Ok(());
                }
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

    pub fn name(&self, ty: &Ty) -> String {
        match ty {
            Ty::Unit | Ty::UInt32 => "uint".into(),
            Ty::Int32 => "int".into(),
            Ty::Bool => "bool".into(),
            Ty::Float32 => "float".into(),
            _ => format!("r_t{}", self.records.iter().position(|t| t == ty).unwrap()),
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

    pub fn declarations(&self) -> String {
        self.records
            .iter()
            .map(|ty| {
                let fields = match ty {
                    Ty::Defined { definition } => format!(
                        "{} value;",
                        self.name(self.module.types[definition.index()].body().unwrap())
                    ),
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
            .collect()
    }
}
