//! AST → typed stack IR.

use std::{collections::HashMap, fmt, sync::Arc};

use crate::ast::{Ident, SourceFile, Span, Stmt, StmtKind, Term, TermKind, Type, TypeKind};

use super::scope::{Initialization, Scopes, ValueBinding, ValueBindingKind};
use super::{
    BasicBlock, BlockId, Conv, Function, FunctionId, Global, GlobalId, Instr, Local, LocalId,
    Module, NonLocal, NonLocalId, RecordField, Terminator, Ty, TypeError, TypeErrorKind, TypeId,
    TyperContext, Value, VerifyError, verify,
};

/// Lower a source file to a verified IR module.
/// `functions[0]` initializes globals and evaluates top-level expressions.
pub fn generate(file: &SourceFile) -> Result<Module, GenerateError> {
    Generator::new().generate_file(file)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateError {
    pub span: Span,
    pub kind: GenerateErrorKind,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateErrorKind {
    Type(TypeErrorKind),
    UnboundValue { name: Arc<str> },
    UnboundType { name: Arc<str> },
    UnknownTypeFormer { name: Arc<str> },
    EagerRecursion { name: Arc<str> },
    UninitializedValue { name: Arc<str> },
    DuplicateValue { name: Arc<str> },
    DuplicateType { name: Arc<str> },
    NeedsTypeAnnotation { name: Arc<str> },
    MissingField { name: Arc<str> },
    ExtraField { name: Arc<str> },
    NotAPlace,
    ExpectedType,
    InvalidLiteral { message: Arc<str> },
    InvalidIr(VerifyError),
}
impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compile error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}
impl std::error::Error for GenerateError {}

struct Generator {
    module: Module,
    typer: TyperContext,
    functions: Vec<FunctionBuilder>,
    scopes: Scopes,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Remote {
    owner_depth: usize,
    value: RemoteValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RemoteValue {
    Local(LocalId),
    CurrentClosure,
}

/// The stack holds either the expression's value or an address to that value.
enum Operand {
    Value(Ty),
    Place(Ty),
}

struct FunctionBuilder {
    name: Option<Arc<str>>,
    nonlocals: Vec<NonLocal>,
    param: LocalId,
    result: Option<Ty>,
    locals: Vec<Local>,
    entry: BlockId,
    blocks: Vec<BasicBlock>,
    current: BlockId,
    remote_to_nonlocal: HashMap<Remote, NonLocalId>,
    capture_order: Vec<Remote>,
    terminated: Vec<bool>,
}

impl Generator {
    fn new() -> Self {
        let mut module = Module::default();
        module.functions.push(Function {
            name: Some("init".into()),
            nonlocals: Vec::new(),
            param: LocalId::from_index(0),
            result: Ty::Unit,
            locals: vec![Local {
                name: None,
                ty: Ty::Unit,
            }],
            entry: BlockId::from_index(0),
            blocks: Vec::new(),
        });
        Self {
            module,
            typer: TyperContext::new(),
            functions: vec![FunctionBuilder::new(Some("init".into()))],
            scopes: Scopes::new(),
        }
    }

    fn generate_file(mut self, file: &SourceFile) -> Result<Module, GenerateError> {
        for stmt in &file.stmts {
            self.gen_stmt(stmt)?;
        }
        self.emit(Instr::Push { value: Value::Unit });
        self.terminate(Terminator::Return);
        let init = self.functions.pop().expect("module initializer").finish();
        self.module.functions[0] = init;
        self.module.types = self.typer.into_definitions().map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::Type(err.kind),
        })?;
        verify(&self.module).map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::InvalidIr(err),
        })?;
        Ok(self.module)
    }

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), GenerateError> {
        match &stmt.val {
            StmtKind::Define { name, init } => self.gen_define(name, init),
            StmtKind::DefineType { name, init } => self.gen_define_type(name, init),
            StmtKind::Declare { name, ann } => self.gen_declare(name, ann),
            StmtKind::Expr { term } => {
                self.gen_term(term, None)?;
                self.emit(Instr::Discard);
                Ok(())
            }
        }
    }

    fn gen_define(&mut self, name: &Ident, init: &Term) -> Result<(), GenerateError> {
        let peeked = self.peek_lambda_ty(init);
        let placeholder = peeked.clone().unwrap_or(Ty::Unit);
        let depth = self.current_depth();
        let binding = if depth == 0 {
            let global = self.alloc_global(placeholder, name.val.clone());
            ValueBinding {
                kind: ValueBindingKind::Global(global),
                ty: peeked.clone(),
                initialization: Initialization::Initializing,
                depth,
            }
        } else {
            let local = self.alloc_local(placeholder, Some(name.val.clone()));
            ValueBinding {
                kind: ValueBindingKind::Local(local),
                ty: peeked.clone(),
                initialization: Initialization::Initializing,
                depth,
            }
        };
        self.bind_value(name, binding.clone())?;
        self.emit_binding_address(&binding);
        let found = match &init.val {
            TermKind::Lambda { params, body } => {
                self.gen_lambda(params, body, peeked.as_ref(), Some(&name.val))?
            }
            _ => self.gen_term(init, peeked.as_ref())?,
        };
        let ty = if let Some(expected) = peeked {
            self.apply_ascription(init.span, &expected, found)?
        } else {
            found
        };
        self.complete_value(&name.val, ty)?;
        self.emit(Instr::Store);
        self.emit(Instr::Discard);
        Ok(())
    }

    fn gen_define_type(&mut self, name: &Ident, init: &Type) -> Result<(), GenerateError> {
        let definition = self.typer.reserve_type(name.val.clone());
        self.bind_type(name, definition)?;
        let body = self.eval_type(init)?;
        self.typer
            .define_type(definition, body)
            .map_err(|err| self.type_error(init.span, err))
    }

    fn gen_declare(&mut self, name: &Ident, ann: &Type) -> Result<(), GenerateError> {
        let ty = self.eval_type(ann)?;
        let depth = self.current_depth();
        let binding = if depth == 0 {
            let global = self.alloc_global(ty.clone(), name.val.clone());
            ValueBinding {
                kind: ValueBindingKind::Global(global),
                ty: Some(ty),
                initialization: Initialization::Uninitialized,
                depth,
            }
        } else {
            let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
            ValueBinding {
                kind: ValueBindingKind::Local(local),
                ty: Some(ty),
                initialization: Initialization::Uninitialized,
                depth,
            }
        };
        self.bind_value(name, binding)
    }

    fn gen_term(&mut self, term: &Term, expected: Option<&Ty>) -> Result<Ty, GenerateError> {
        let found = self.gen_term_inner(term, expected)?;
        if let Some(expected) = expected {
            self.typer
                .same(expected, &found)
                .map_err(|err| self.type_error(term.span, err))?;
        }
        Ok(found)
    }

    fn gen_term_inner(&mut self, term: &Term, expected: Option<&Ty>) -> Result<Ty, GenerateError> {
        match &term.val {
            TermKind::Var { name } => self.gen_var(name),
            TermKind::Num { value } => {
                let pushed = self.evaluate_number(term.span, value, expected)?;
                let ty = immediate_ty(&pushed);
                self.emit(Instr::Push { value: pushed });
                Ok(ty)
            }
            TermKind::Unit => {
                self.emit(Instr::Push { value: Value::Unit });
                Ok(Ty::Unit)
            }
            TermKind::Type { ty } => {
                let value = self.eval_type(ty)?;
                self.emit(Instr::Push {
                    value: Value::Type { ty: value },
                });
                Ok(Ty::Type)
            }
            TermKind::Lambda { params, body } => self.gen_lambda(params, body, expected, None),
            TermKind::If { cond, then, els } => self.gen_if(cond, then, els, expected),
            TermKind::Array { elems } => self.gen_array(term.span, elems, expected),
            TermKind::Record { fields } => self.gen_record(term.span, fields, expected),
            TermKind::Block { stmts, tail } => self.gen_block(stmts, tail, expected),
            TermKind::Call { func, arg } => self.gen_call(term.span, func, arg),
            TermKind::Builtin { name, args } => self.gen_builtin(term.span, name, args, expected),
            TermKind::Assign { place, value } => self.gen_assign(place, value),
            TermKind::Deref { pointer } => {
                let pointer_ty = self.gen_term(pointer, None)?;
                let converted = self
                    .typer
                    .as_pointer(&pointer_ty)
                    .map_err(|err| self.type_error(term.span, err))?;
                self.emit_value_conv(&converted.steps);
                let ty = self
                    .typer
                    .type_deref(&pointer_ty)
                    .map_err(|err| self.type_error(term.span, err))?;
                self.emit(Instr::Load);
                Ok(ty)
            }
            TermKind::Field { base, name } => self.gen_field_value(term.span, base, name),
        }
    }

    fn gen_var(&mut self, name: &Ident) -> Result<Ty, GenerateError> {
        let binding = self.resolve_value(name)?;
        let ty = self.binding_ty(name, &binding)?;
        if matches!(binding.kind, ValueBindingKind::CurrentClosure) {
            if binding.depth == self.current_depth() {
                self.emit(Instr::CurrentClosure);
            } else {
                let nonlocal = self.capture(
                    Remote {
                        owner_depth: binding.depth,
                        value: RemoteValue::CurrentClosure,
                    },
                    &ty,
                );
                self.emit(Instr::NonLocalAddress { nonlocal });
                self.emit(Instr::Load);
            }
            return Ok(ty);
        }
        self.emit_binding_address(&binding);
        self.emit(Instr::Load);
        Ok(ty)
    }

    fn gen_lambda(
        &mut self,
        params: &[(Ident, Type)],
        body: &Term,
        expected: Option<&Ty>,
        name: Option<&Arc<str>>,
    ) -> Result<Ty, GenerateError> {
        let expected_fn = expected.and_then(|ty| match ty {
            Ty::Function { param, result } => Some((param.as_ref(), result.as_ref())),
            _ => None,
        });

        let enclosing_scopes = self.scopes.clone();
        let recursive_binding = name
            .and_then(|name| self.scopes.lookup_value(name))
            .filter(|binding| matches!(binding.kind, ValueBindingKind::Local(_)))
            .cloned();
        self.functions.push(FunctionBuilder::new(name.cloned()));
        self.scopes.push();
        if let (Some(name), Some(binding)) = (name, recursive_binding) {
            self.scopes
                .define_value(
                    name.clone(),
                    ValueBinding {
                        kind: ValueBindingKind::CurrentClosure,
                        ty: binding.ty,
                        initialization: Initialization::Initialized,
                        depth: self.current_depth(),
                    },
                )
                .expect("fresh recursive-name scope");
        }
        // Parameters may shadow the recursive name.
        self.scopes.push();

        let mut param_tys = Vec::with_capacity(params.len());
        for (_, ann) in params {
            param_tys.push(self.eval_type(ann)?);
        }
        let param_ty = Ty::parameter(&param_tys);
        let parameter = LocalId::from_index(0);
        self.functions.last_mut().expect("lambda builder").locals[0] = Local {
            name: if params.len() == 1 {
                Some(params[0].0.val.clone())
            } else {
                None
            },
            ty: param_ty.clone(),
        };
        for (index, ((name, _), ty)) in params.iter().zip(&param_tys).enumerate() {
            let local = if params.len() == 1 {
                parameter
            } else {
                let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
                self.emit(Instr::LocalAddress { local });
                self.emit(Instr::LocalAddress { local: parameter });
                self.emit(Instr::AccessStatic { index });
                self.emit(Instr::Load);
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
                local
            };
            self.bind_value(
                name,
                ValueBinding {
                    kind: ValueBindingKind::Local(local),
                    ty: Some(ty.clone()),
                    initialization: Initialization::Initialized,
                    depth: self.current_depth(),
                },
            )?;
        }
        if let Some((expected_param, _)) = expected_fn {
            self.typer
                .same(expected_param, &param_ty)
                .map_err(|err| self.type_error(body.span, err))?;
        }

        let body_expected = self
            .peek_result_ty(body)
            .or_else(|| expected_fn.map(|(_, result)| result.clone()));
        let body_ty = self.gen_term(body, body_expected.as_ref())?;
        self.functions.last_mut().expect("lambda builder").result = Some(body_ty.clone());
        self.terminate(Terminator::Return);
        self.scopes.pop();
        self.scopes.pop();
        // Generating a delayed body cannot initialize its enclosing bindings.
        self.scopes = enclosing_scopes;
        self.finish_lambda()?;
        Ok(self.typer.type_lambda(&param_ty, &body_ty))
    }

    fn gen_if(
        &mut self,
        cond: &Term,
        then: &Term,
        els: &Term,
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        let cond_ty = self.gen_term(cond, None)?;
        let converted = self
            .typer
            .as_bool(&cond_ty)
            .map_err(|err| self.type_error(cond.span, err))?;
        self.emit_value_conv(&converted.steps);
        let then_block = self.new_block("then");
        let else_block = self.new_block("else");
        let join_block = self.new_block("join");
        self.terminate(Terminator::Branch {
            then: then_block,
            els: else_block,
        });

        let before = self.scopes.clone();
        self.switch(then_block);
        let then_ty = self.gen_term(then, expected)?;
        self.terminate(Terminator::Break { target: join_block });
        let after_then = self.scopes.clone();

        self.scopes = before;
        self.switch(else_block);
        let else_ty = self.gen_term(els, expected)?;
        self.terminate(Terminator::Break { target: join_block });
        self.scopes.intersect_initialization(&after_then);

        self.switch(join_block);
        self.typer
            .type_if(&Ty::Bool, &then_ty, &else_ty)
            .map_err(|err| self.type_error(cond.span, err))
    }

    fn gen_array(
        &mut self,
        span: Span,
        elems: &[Term],
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        let expected_element = expected.and_then(|ty| self.array_element(ty));
        let mut elem_tys = Vec::with_capacity(elems.len());
        for elem in elems {
            elem_tys.push(self.gen_term(elem, expected_element.as_ref())?);
        }
        let ty = if let Some(element) = expected_element {
            self.typer
                .type_array_of(&element, &elem_tys)
                .map_err(|err| self.type_error(span, err))?
        } else {
            self.typer
                .type_array(&elem_tys)
                .map_err(|err| self.type_error(span, err))?
        };
        let Ty::Array { element, length } = &ty else {
            unreachable!("array typing produces an array");
        };
        self.emit(Instr::MakeArray {
            elements: *length,
            element: element.as_ref().clone(),
        });
        Ok(ty)
    }

    fn gen_record(
        &mut self,
        span: Span,
        fields: &[(Ident, Term)],
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        if let Some(expected_fields) = expected.and_then(|ty| self.record_fields(ty)) {
            let mut source = HashMap::with_capacity(fields.len());
            for (name, value) in fields {
                if source.insert(name.val.clone(), (name, value)).is_some() {
                    return Err(GenerateError {
                        span: name.span,
                        kind: GenerateErrorKind::Type(TypeErrorKind::DuplicateField {
                            name: name.val.clone(),
                        }),
                    });
                }
            }
            for field in &expected_fields {
                if !source.contains_key(&field.name) {
                    return Err(GenerateError {
                        span,
                        kind: GenerateErrorKind::MissingField {
                            name: field.name.clone(),
                        },
                    });
                }
            }
            for (name, _) in fields {
                if !expected_fields.iter().any(|field| field.name == name.val) {
                    return Err(GenerateError {
                        span: name.span,
                        kind: GenerateErrorKind::ExtraField {
                            name: name.val.clone(),
                        },
                    });
                }
            }
            // Evaluate in source order; layout order must not reorder effects.
            let mut values = HashMap::with_capacity(fields.len());
            for (name, value) in fields {
                let field = expected_fields
                    .iter()
                    .find(|field| field.name == name.val)
                    .unwrap();
                let local = self.alloc_local(field.ty.clone(), None);
                self.emit(Instr::LocalAddress { local });
                self.gen_term(value, Some(&field.ty))?;
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
                values.insert(name.val.clone(), local);
            }
            let names = expected_fields
                .iter()
                .map(|field| {
                    self.emit(Instr::LocalAddress {
                        local: values[&field.name],
                    });
                    self.emit(Instr::Load);
                    field.name.clone()
                })
                .collect();
            self.emit(Instr::MakeRecord { fields: names });
            return Ok(Ty::Record {
                fields: expected_fields,
            });
        }

        let mut names = Vec::with_capacity(fields.len());
        let mut typed = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            let ty = self.gen_term(value, None)?;
            names.push(name.val.clone());
            typed.push(RecordField {
                name: name.val.clone(),
                ty,
            });
        }
        let ty = self
            .typer
            .type_record(&typed)
            .map_err(|err| self.type_error(span, err))?;
        self.emit(Instr::MakeRecord { fields: names });
        Ok(ty)
    }

    fn gen_block(
        &mut self,
        stmts: &[Stmt],
        tail: &Term,
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        self.scopes.push();
        for stmt in stmts {
            self.gen_stmt(stmt)?;
        }
        let ty = self.gen_term(tail, expected)?;
        self.scopes.pop();
        Ok(self.typer.type_block(&ty))
    }

    fn gen_call(&mut self, span: Span, func: &Term, arg: &Term) -> Result<Ty, GenerateError> {
        if let TermKind::Type { ty } = &func.val {
            return self.gen_ascription(span, ty, arg);
        }

        let callee_ty = self.gen_term(func, None)?;
        let converted = self
            .typer
            .as_function(&callee_ty)
            .map_err(|err| self.type_error(func.span, err))?;
        self.emit_value_conv(&converted.steps);
        let Ty::Function { param, .. } = &converted.ty else {
            return Err(self.type_error(
                func.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedFunction { found: callee_ty },
                },
            ));
        };
        let arg_ty = self.gen_term(arg, Some(param))?;
        let result = self
            .typer
            .type_call(&callee_ty, &arg_ty)
            .map_err(|err| self.type_error(span, err))?;
        self.emit(Instr::Call);
        Ok(result)
    }

    fn gen_ascription(&mut self, span: Span, ty: &Type, arg: &Term) -> Result<Ty, GenerateError> {
        let ascribed = self.eval_type(ty)?;
        let context = self
            .typer
            .body(&ascribed)
            .map_err(|err| self.type_error(span, err))?;
        let found = self.gen_term_inner(arg, Some(&context))?;
        self.apply_ascription(span, &ascribed, found)
    }

    fn gen_builtin(
        &mut self,
        span: Span,
        name: &str,
        args: &[Term],
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        if name == "&&" || name == "||" {
            return self.gen_short_circuit(span, name, args);
        }
        if matches!(name, "+" | "-")
            && let [
                Term {
                    val: TermKind::Num { value },
                    ..
                },
            ] = args
        {
            let text = if name == "-" {
                format!("-{value}")
            } else {
                value.to_string()
            };
            let value = self.evaluate_number(span, &text, expected)?;
            let ty = immediate_ty(&value);
            self.emit(Instr::Push { value });
            return Ok(ty);
        }
        let numeric_context = expected.filter(|ty| is_numeric(ty)).filter(|_| {
            matches!(
                name,
                "+" | "-" | "*" | "/" | "%" | "~" | "<<" | ">>" | "&" | "|" | "^"
            )
        });
        let mut arg_tys = Vec::with_capacity(args.len());
        for arg in args {
            arg_tys.push(self.gen_term(arg, numeric_context)?);
        }
        if name == "!" && arg_tys.len() == 1 {
            let converted = self
                .typer
                .as_bool(&arg_tys[0])
                .map_err(|err| self.type_error(span, err))?;
            self.emit_value_conv(&converted.steps);
            arg_tys[0] = converted.ty;
        }
        let call = self
            .typer
            .type_builtin_call(name, &arg_tys)
            .map_err(|err| self.type_error(span, err))?;
        self.emit(Instr::CallBuiltin {
            name: Arc::from(name),
            params: call.params,
            result: call.result.clone(),
        });
        Ok(call.result)
    }

    fn gen_short_circuit(
        &mut self,
        span: Span,
        name: &str,
        args: &[Term],
    ) -> Result<Ty, GenerateError> {
        if args.len() != 2 {
            return Err(self.type_error(
                span,
                TypeError {
                    kind: TypeErrorKind::InvalidBuiltinArgumentCount {
                        name: Arc::from(name),
                        found: args.len(),
                    },
                },
            ));
        }
        let left_ty = self.gen_term(&args[0], None)?;
        let converted = self
            .typer
            .as_bool(&left_ty)
            .map_err(|err| self.type_error(span, err))?;
        self.emit_value_conv(&converted.steps);
        let then_block = self.new_block("then");
        let else_block = self.new_block("else");
        let join_block = self.new_block("join");
        self.terminate(Terminator::Branch {
            then: then_block,
            els: else_block,
        });
        let before_right = self.scopes.clone();
        if name == "&&" {
            self.switch(then_block);
            self.gen_bool(&args[1])?;
            self.terminate(Terminator::Break { target: join_block });
            self.switch(else_block);
            self.emit(Instr::Push {
                value: Value::Bool { value: false },
            });
            self.terminate(Terminator::Break { target: join_block });
        } else {
            self.switch(then_block);
            self.emit(Instr::Push {
                value: Value::Bool { value: true },
            });
            self.terminate(Terminator::Break { target: join_block });
            self.switch(else_block);
            self.gen_bool(&args[1])?;
            self.terminate(Terminator::Break { target: join_block });
        }
        self.switch(join_block);
        self.scopes.intersect_initialization(&before_right);
        Ok(Ty::Bool)
    }

    fn gen_bool(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let converted = self
            .typer
            .as_bool(&ty)
            .map_err(|err| self.type_error(term.span, err))?;
        self.emit_value_conv(&converted.steps);
        Ok(Ty::Bool)
    }

    fn gen_assign(&mut self, place: &Term, value: &Term) -> Result<Ty, GenerateError> {
        let place_ty = self.gen_place(place)?;
        let converted = self
            .typer
            .as_pointer(&place_ty)
            .map_err(|err| self.type_error(place.span, err))?;
        self.emit_value_conv(&converted.steps);
        let Ty::Pointer { pointee } = converted.ty else {
            return Err(self.type_error(
                place.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedPointer { found: place_ty },
                },
            ));
        };
        let value_ty = self.gen_term(value, Some(&pointee))?;
        let ty = self
            .typer
            .type_assign(&pointee, &value_ty)
            .map_err(|err| self.type_error(place.span, err))?;
        self.emit(Instr::Store);
        if let TermKind::Var { name } = &place.val {
            self.scopes
                .lookup_value_mut(&name.val)
                .expect("assigned binding")
                .initialization = Initialization::Initialized;
        }
        Ok(ty)
    }

    fn gen_field_value(
        &mut self,
        span: Span,
        base: &Term,
        name: &Ident,
    ) -> Result<Ty, GenerateError> {
        self.check_place_initialized(base)?;
        let base = self.gen_operand(base)?;
        match self.gen_field_operand(span, base, name)? {
            Operand::Place(ty) => {
                self.emit(Instr::Load);
                Ok(ty)
            }
            Operand::Value(ty) => Ok(ty),
        }
    }

    fn gen_place(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        match self.gen_operand(term)? {
            Operand::Place(ty) => Ok(Ty::Pointer {
                pointee: Box::new(ty),
            }),
            Operand::Value(_) => Err(GenerateError {
                span: term.span,
                kind: GenerateErrorKind::NotAPlace,
            }),
        }
    }

    fn check_place_initialized(&self, term: &Term) -> Result<(), GenerateError> {
        match &term.val {
            TermKind::Var { name } => {
                self.resolve_value(name)?;
            }
            TermKind::Field { base, .. } => self.check_place_initialized(base)?,
            _ => {}
        }
        Ok(())
    }

    // Lower once, preserving an address when available. Speculatively generating
    // a place and then retrying as a value can evaluate side effects twice.
    fn gen_operand(&mut self, term: &Term) -> Result<Operand, GenerateError> {
        match &term.val {
            TermKind::Var { name } => {
                let binding = self.resolve_binding(name, false)?;
                if matches!(binding.kind, ValueBindingKind::CurrentClosure) {
                    return self.gen_var(name).map(Operand::Value);
                }
                let ty = self.binding_ty(name, &binding)?;
                self.emit_binding_address(&binding);
                Ok(Operand::Place(ty))
            }
            TermKind::Field { base, name } => {
                self.check_place_initialized(base)?;
                let base = self.gen_operand(base)?;
                self.gen_field_operand(term.span, base, name)
            }
            TermKind::Deref { pointer } => {
                let pointer_ty = self.gen_term(pointer, None)?;
                let converted = self
                    .typer
                    .as_pointer(&pointer_ty)
                    .map_err(|err| self.type_error(term.span, err))?;
                self.emit_value_conv(&converted.steps);
                let Ty::Pointer { pointee } = converted.ty else {
                    unreachable!("as_pointer returns a pointer")
                };
                Ok(Operand::Place(*pointee))
            }
            _ => self.gen_term(term, None).map(Operand::Value),
        }
    }

    fn gen_field_operand(
        &mut self,
        span: Span,
        base: Operand,
        name: &Ident,
    ) -> Result<Operand, GenerateError> {
        let (base_ty, mut is_place) = match base {
            Operand::Value(ty) => (ty, false),
            Operand::Place(ty) => (ty, true),
        };
        let access = self
            .typer
            .type_field(&base_ty, &name.val)
            .map_err(|err| self.type_error(span, err))?;
        if is_place {
            self.emit_place_conv(&access.steps);
        } else if let Some(last_deref) = access
            .steps
            .iter()
            .rposition(|step| matches!(step, Conv::Deref))
        {
            // Keep the last address so pointer-valued expressions have writable fields.
            self.emit_value_conv(&access.steps[..last_deref]);
            is_place = true;
        } else {
            self.emit_value_conv(&access.steps);
        }
        self.emit(Instr::AccessStatic {
            index: access.index,
        });
        Ok(if is_place {
            Operand::Place(access.ty)
        } else {
            Operand::Value(access.ty)
        })
    }

    fn evaluate_number(
        &self,
        span: Span,
        text: &str,
        expected: Option<&Ty>,
    ) -> Result<Value, GenerateError> {
        let ty = self.numeric_ty(span, text, expected)?;
        parse_number(text, &ty).map_err(|message| GenerateError {
            span,
            kind: GenerateErrorKind::InvalidLiteral {
                message: message.into(),
            },
        })
    }

    fn numeric_ty(
        &self,
        span: Span,
        text: &str,
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        if let Some(expected) = expected {
            let shape = self
                .typer
                .body(expected)
                .map_err(|err| self.type_error(span, err))?;
            if is_numeric(&shape) {
                return Ok(shape);
            }
        }
        Ok(self.typer.type_num(text))
    }

    fn eval_type(&mut self, ty: &Type) -> Result<Ty, GenerateError> {
        match &ty.val {
            TypeKind::Unit => Ok(Ty::Unit),
            TypeKind::Atom { name } => {
                if let Some(builtin) = builtin_ty(&name.val) {
                    return Ok(builtin);
                }
                self.resolve_type(name)
                    .map(|definition| Ty::Defined { definition })
            }
            TypeKind::App { head, arg } => {
                let arg = self.eval_term_as_type(arg)?;
                match head.val.as_ref() {
                    "Ptr" => Ok(Ty::Pointer {
                        pointee: Box::new(arg),
                    }),
                    "Span" => Ok(Ty::Span {
                        element: Box::new(arg),
                    }),
                    _ => Err(GenerateError {
                        span: head.span,
                        kind: GenerateErrorKind::UnknownTypeFormer {
                            name: head.val.clone(),
                        },
                    }),
                }
            }
            TypeKind::Func { from, to } => Ok(Ty::Function {
                param: Box::new(self.eval_type(from)?),
                result: Box::new(self.eval_type(to)?),
            }),
            TypeKind::Record { fields } => {
                let mut typed = Vec::with_capacity(fields.len());
                for (name, field_ty) in fields {
                    typed.push(RecordField {
                        name: name.val.clone(),
                        ty: self.eval_type(field_ty)?,
                    });
                }
                self.typer
                    .type_record(&typed)
                    .map_err(|err| self.type_error(ty.span, err))
            }
        }
    }

    fn eval_term_as_type(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        match &term.val {
            TermKind::Type { ty } => self.eval_type(ty),
            _ => Err(GenerateError {
                span: term.span,
                kind: GenerateErrorKind::ExpectedType,
            }),
        }
    }

    fn peek_lambda_ty(&mut self, term: &Term) -> Option<Ty> {
        let TermKind::Lambda { params, body } = &term.val else {
            return None;
        };
        let mut param_tys = Vec::with_capacity(params.len());
        for (_, ann) in params {
            param_tys.push(self.eval_type(ann).ok()?);
        }
        let result = self.peek_result_ty(body)?;
        Some(Ty::Function {
            param: Box::new(Ty::parameter(&param_tys)),
            result: Box::new(result),
        })
    }

    fn peek_result_ty(&mut self, term: &Term) -> Option<Ty> {
        let TermKind::Call { func, .. } = &term.val else {
            return None;
        };
        let TermKind::Type { ty } = &func.val else {
            return None;
        };
        self.eval_type(ty).ok()
    }

    fn apply_ascription(
        &mut self,
        span: Span,
        expected: &Ty,
        found: Ty,
    ) -> Result<Ty, GenerateError> {
        let steps = self
            .typer
            .ascribe(&found, expected)
            .map_err(|err| self.type_error(span, err))?;
        self.emit_value_conv(&steps);
        Ok(expected.clone())
    }

    fn emit_value_conv(&mut self, steps: &[Conv]) {
        for step in steps {
            match step {
                Conv::Unwrap { definition } => {
                    let ty = self
                        .typer
                        .body(&Ty::Defined {
                            definition: *definition,
                        })
                        .expect("validated type definition");
                    self.emit(Instr::Ascribe { ty });
                }
                Conv::Wrap { definition } => {
                    self.emit(Instr::Ascribe {
                        ty: Ty::Defined {
                            definition: *definition,
                        },
                    });
                }
                Conv::Deref => self.emit(Instr::Load),
            }
        }
    }

    fn emit_place_conv(&mut self, steps: &[Conv]) {
        for step in steps {
            if matches!(step, Conv::Deref) {
                self.emit(Instr::Load);
            }
        }
    }

    fn finish_lambda(&mut self) -> Result<(), GenerateError> {
        let child = self.functions.pop().expect("lambda builder");
        let captures = child.capture_order.clone();
        let n_captures = child.nonlocals.len();
        let function = child.finish();
        let id = FunctionId::from_index(self.module.functions.len());
        self.module.functions.push(function);
        for remote in captures {
            self.emit_capture_value(remote)?;
        }
        self.emit(Instr::MakeClosure {
            function: id,
            captures: n_captures,
        });
        Ok(())
    }

    fn emit_capture_value(&mut self, remote: Remote) -> Result<(), GenerateError> {
        let current = self.current_depth();
        if remote.owner_depth == current {
            match remote.value {
                RemoteValue::Local(local) => self.emit(Instr::LocalAddress { local }),
                RemoteValue::CurrentClosure => {
                    self.emit(Instr::CurrentClosure);
                    return Ok(());
                }
            }
        } else {
            let nonlocal = self.functions[current]
                .remote_to_nonlocal
                .get(&remote)
                .copied()
                .expect("enclosing function captures the same remote");
            self.emit(Instr::NonLocalAddress { nonlocal });
        }
        self.emit(Instr::Load);
        Ok(())
    }

    fn emit_binding_address(&mut self, binding: &ValueBinding) {
        match binding.kind {
            ValueBindingKind::Global(global) => self.emit(Instr::GlobalAddress { global }),
            ValueBindingKind::Local(local) if binding.depth == self.current_depth() => {
                self.emit(Instr::LocalAddress { local });
            }
            ValueBindingKind::Local(local) => {
                let ty = binding.ty.clone().expect("captured bindings are typed");
                let nonlocal = self.capture(
                    Remote {
                        owner_depth: binding.depth,
                        value: RemoteValue::Local(local),
                    },
                    &ty,
                );
                self.emit(Instr::NonLocalAddress { nonlocal });
            }
            ValueBindingKind::CurrentClosure => {
                unreachable!("recursive names are values, not places")
            }
        }
    }

    fn capture(&mut self, remote: Remote, ty: &Ty) -> NonLocalId {
        let current = self.current_depth();
        for depth in remote.owner_depth + 1..=current {
            if self.functions[depth]
                .remote_to_nonlocal
                .contains_key(&remote)
            {
                continue;
            }
            let nonlocal = NonLocalId::from_index(self.functions[depth].nonlocals.len());
            let owner = &self.functions[remote.owner_depth];
            let name = match remote.value {
                RemoteValue::Local(local) => owner.locals[local.index()].name.clone(),
                RemoteValue::CurrentClosure => owner.name.clone(),
            };
            self.functions[depth].nonlocals.push(NonLocal {
                name,
                ty: ty.clone(),
            });
            self.functions[depth]
                .remote_to_nonlocal
                .insert(remote, nonlocal);
            self.functions[depth].capture_order.push(remote);
        }
        self.functions[current].remote_to_nonlocal[&remote]
    }

    fn resolve_value(&self, name: &Ident) -> Result<ValueBinding, GenerateError> {
        self.resolve_binding(name, true)
    }

    fn resolve_binding(&self, name: &Ident, read: bool) -> Result<ValueBinding, GenerateError> {
        let binding = self
            .scopes
            .lookup_value(&name.val)
            .cloned()
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })?;
        if binding.initialization == Initialization::Initializing
            && binding.depth == self.current_depth()
        {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::EagerRecursion {
                    name: name.val.clone(),
                },
            });
        }
        let captures_local = binding.depth != self.current_depth()
            && matches!(binding.kind, ValueBindingKind::Local(_));
        if binding.initialization != Initialization::Initialized
            && (captures_local || read && binding.depth == self.current_depth())
        {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UninitializedValue {
                    name: name.val.clone(),
                },
            });
        }
        Ok(binding)
    }

    fn resolve_type(&self, name: &Ident) -> Result<TypeId, GenerateError> {
        self.scopes
            .lookup_type(&name.val)
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundType {
                    name: name.val.clone(),
                },
            })
    }

    fn bind_value(&mut self, name: &Ident, binding: ValueBinding) -> Result<(), GenerateError> {
        self.scopes
            .define_value(name.val.clone(), binding)
            .map_err(|dup| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: dup },
            })
    }

    fn bind_type(&mut self, name: &Ident, definition: TypeId) -> Result<(), GenerateError> {
        self.scopes
            .define_type(name.val.clone(), definition)
            .map_err(|dup| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateType { name: dup },
            })
    }

    fn complete_value(&mut self, name: &Arc<str>, ty: Ty) -> Result<(), GenerateError> {
        let (kind, depth) = {
            let binding = self.scopes.lookup_value_mut(name).expect("defined binding");
            binding.ty = Some(ty.clone());
            binding.initialization = Initialization::Initialized;
            (binding.kind, binding.depth)
        };
        match kind {
            ValueBindingKind::Global(global) => {
                self.module.globals[global.index()].ty = ty;
            }
            ValueBindingKind::Local(local) => {
                self.functions[depth].locals[local.index()].ty = ty;
            }
            ValueBindingKind::CurrentClosure => {
                unreachable!("cannot define a recursive-name binding")
            }
        }
        Ok(())
    }

    fn binding_ty(&self, name: &Ident, binding: &ValueBinding) -> Result<Ty, GenerateError> {
        binding.ty.clone().ok_or_else(|| GenerateError {
            span: name.span,
            kind: GenerateErrorKind::NeedsTypeAnnotation {
                name: name.val.clone(),
            },
        })
    }

    fn array_element(&self, ty: &Ty) -> Option<Ty> {
        match self.typer.body(ty).ok()? {
            Ty::Array { element, .. } => Some(*element),
            _ => None,
        }
    }

    fn record_fields(&self, ty: &Ty) -> Option<Vec<RecordField>> {
        match self.typer.body(ty).ok()? {
            Ty::Record { fields } => Some(fields),
            _ => None,
        }
    }

    fn alloc_local(&mut self, ty: Ty, name: Option<Arc<str>>) -> LocalId {
        let function = self.functions.last_mut().expect("function");
        let id = LocalId::from_index(function.locals.len());
        function.locals.push(Local { name, ty });
        id
    }

    fn alloc_global(&mut self, ty: Ty, name: Arc<str>) -> GlobalId {
        let id = GlobalId::from_index(self.module.globals.len());
        self.module.globals.push(Global { name, ty });
        id
    }

    fn emit(&mut self, instr: Instr) {
        let function = self.functions.last_mut().expect("function");
        let current = function.current.index();
        debug_assert!(!function.terminated[current]);
        function.blocks[current].instrs.push(instr);
    }

    fn terminate(&mut self, terminator: Terminator) {
        let function = self.functions.last_mut().expect("function");
        let current = function.current.index();
        debug_assert!(!function.terminated[current]);
        function.blocks[current].terminator = terminator;
        function.terminated[current] = true;
    }

    fn new_block(&mut self, hint: &str) -> BlockId {
        self.functions.last_mut().expect("function").new_block(hint)
    }

    fn switch(&mut self, block: BlockId) {
        self.functions.last_mut().expect("function").current = block;
    }

    fn current_depth(&self) -> usize {
        self.functions.len() - 1
    }

    fn type_error(&self, span: Span, err: TypeError) -> GenerateError {
        GenerateError {
            span,
            kind: GenerateErrorKind::Type(err.kind),
        }
    }
}

impl FunctionBuilder {
    fn new(name: Option<Arc<str>>) -> Self {
        Self {
            name,
            nonlocals: Vec::new(),
            param: LocalId::from_index(0),
            result: None,
            locals: vec![Local {
                name: None,
                ty: Ty::Unit,
            }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: Some("entry".into()),
                instrs: Vec::new(),
                terminator: Terminator::Return,
            }],
            current: BlockId::from_index(0),
            remote_to_nonlocal: HashMap::new(),
            capture_order: Vec::new(),
            terminated: vec![false],
        }
    }

    fn new_block(&mut self, hint: &str) -> BlockId {
        let id = BlockId::from_index(self.blocks.len());
        self.blocks.push(BasicBlock {
            name: Some(self.unique_block_name(hint)),
            instrs: Vec::new(),
            terminator: Terminator::Return,
        });
        self.terminated.push(false);
        id
    }

    fn unique_block_name(&self, hint: &str) -> Arc<str> {
        if !self
            .blocks
            .iter()
            .any(|block| block.name.as_deref() == Some(hint))
        {
            return hint.into();
        }
        let mut suffix = 1;
        loop {
            let candidate = format!("{hint}.{suffix}");
            if !self
                .blocks
                .iter()
                .any(|block| block.name.as_deref() == Some(candidate.as_str()))
            {
                return candidate.into();
            }
            suffix += 1;
        }
    }

    fn finish(self) -> Function {
        Function {
            name: self.name,
            nonlocals: self.nonlocals,
            param: self.param,
            result: self.result.unwrap_or(Ty::Unit),
            locals: self.locals,
            entry: self.entry,
            blocks: self.blocks,
        }
    }
}

fn builtin_ty(name: &str) -> Option<Ty> {
    Some(match name {
        "sbyte" => Ty::Int8,
        "short" => Ty::Int16,
        "int" => Ty::Int32,
        "long" => Ty::Int64,
        "ubyte" => Ty::UInt8,
        "ushort" => Ty::UInt16,
        "uint" => Ty::UInt32,
        "ulong" => Ty::UInt64,
        "float32" => Ty::Float32,
        "float64" => Ty::Float64,
        _ => return None,
    })
}

fn is_numeric(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::Int8
            | Ty::Int16
            | Ty::Int32
            | Ty::Int64
            | Ty::UInt8
            | Ty::UInt16
            | Ty::UInt32
            | Ty::UInt64
            | Ty::Float32
            | Ty::Float64
    )
}

fn immediate_ty(value: &Value) -> Ty {
    match value {
        Value::Type { .. } => Ty::Type,
        Value::Unit => Ty::Unit,
        Value::Bool { .. } => Ty::Bool,
        Value::Int8 { .. } => Ty::Int8,
        Value::Int16 { .. } => Ty::Int16,
        Value::Int32 { .. } => Ty::Int32,
        Value::Int64 { .. } => Ty::Int64,
        Value::UInt8 { .. } => Ty::UInt8,
        Value::UInt16 { .. } => Ty::UInt16,
        Value::UInt32 { .. } => Ty::UInt32,
        Value::UInt64 { .. } => Ty::UInt64,
        Value::Float32 { .. } => Ty::Float32,
        Value::Float64 { .. } => Ty::Float64,
        Value::Array { value } => Ty::Array {
            element: Box::new(value.element_ty.clone()),
            length: value.elements.len(),
        },
        Value::Record { value } => Ty::Record {
            fields: value
                .fields
                .iter()
                .map(|field| RecordField {
                    name: field.name.clone(),
                    ty: immediate_ty(&field.value),
                })
                .collect(),
        },
        Value::StaticAddress { .. } | Value::DynamicAddress { .. } | Value::Closure { .. } => {
            unreachable!("evaluate() is limited to literals")
        }
    }
}

fn parse_number(text: &str, ty: &Ty) -> Result<Value, String> {
    let compact: String = text.chars().filter(|c| *c != '_').collect();
    match ty {
        Ty::Float32 => Ok(Value::Float32 {
            value: parse_float(&compact)? as f32,
        }),
        Ty::Float64 => Ok(Value::Float64 {
            value: parse_float(&compact)?,
        }),
        Ty::Int8 => Ok(Value::Int8 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int16 => Ok(Value::Int16 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int32 => Ok(Value::Int32 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int64 => Ok(Value::Int64 {
            value: parse_signed(&compact)?,
        }),
        Ty::UInt8 => Ok(Value::UInt8 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt16 => Ok(Value::UInt16 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt32 => Ok(Value::UInt32 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt64 => Ok(Value::UInt64 {
            value: parse_unsigned(&compact)?,
        }),
        other => Err(format!("cannot use numeric literal as {other:?}")),
    }
}

fn parse_float(text: &str) -> Result<f64, String> {
    if is_hex_literal(text) {
        return Err("hexadecimal float literals are not supported".into());
    }
    text.parse()
        .map_err(|err| format!("invalid float literal: {err}"))
}

fn parse_signed<T: TryFrom<i128>>(text: &str) -> Result<T, String>
where
    T::Error: fmt::Display,
{
    let (negative, magnitude) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let value = if let Some(hex) = magnitude
        .strip_prefix("0x")
        .or_else(|| magnitude.strip_prefix("0X"))
    {
        i128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        magnitude
            .parse::<i128>()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    let value = if negative { -value } else { value };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn parse_unsigned<T: TryFrom<u128>>(text: &str) -> Result<T, String>
where
    T::Error: fmt::Display,
{
    let value = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        text.parse()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn is_hex_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    value.len() >= 2
        && value.as_bytes()[0] == b'0'
        && (value.as_bytes()[1] == b'x' || value.as_bytes()[1] == b'X')
}
