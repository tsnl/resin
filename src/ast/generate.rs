//! Tree-sitter parse tree → AST generation.

use std::{fmt, sync::Arc};

use tree_sitter::Node;

use super::*;

pub struct AstGen<'a> {
    src: &'a str,
    recovering: bool,
    source_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstError {
    pub span: Span,
    pub kind: AstErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstErrorKind {
    Unexpected { found: Arc<str> },
    Missing { expected: Arc<str> },
}

impl fmt::Display for AstError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}

impl std::error::Error for AstError {}

impl<'a> AstGen<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            recovering: false,
            source_len: src.len(),
        }
    }

    /// Editor-only AST generation. Strict parsing remains the default.
    pub(crate) fn recovering(src: &'a str, source_len: usize) -> Self {
        Self {
            src,
            recovering: true,
            source_len,
        }
    }

    fn hole(&self, node: Node) -> Term {
        Spanned::new(
            TermKind::Hole {
                children: Vec::new(),
            },
            self.span(node),
        )
    }

    // Tree-sitter sometimes places an unfinished postfix/operator next to the
    // recognized expression instead of inside it. Preserve that receiver/operand.
    fn trailing_errors(&self, term: Term, parent: Node) -> Term {
        if !self.recovering {
            return term;
        }
        let errors: Vec<_> = parent
            .children(&mut parent.walk())
            .filter(|n| n.is_error() && n.start_byte() >= term.span.end)
            .collect();
        if errors.is_empty() {
            return term;
        }
        let span = Span {
            start: term.span.start,
            end: errors.last().unwrap().end_byte().min(self.source_len),
        };
        let val = if errors.len() == 1 && self.text(errors[0]).trim() == "." {
            TermKind::FieldHole {
                base: Box::new(term),
            }
        } else {
            TermKind::Hole {
                children: vec![term],
            }
        };
        Spanned::new(val, span)
    }

    pub fn gen_source_file(&self, node: Node) -> Result<SourceFile, AstError> {
        if !self.recovering
            && let Some(error) = first_error_node(node)
        {
            return Err(self.parse_error(error));
        }
        if !self.recovering {
            assert_eq!(node.kind(), "source_file");
        }
        let mut stmts = Vec::new();
        let mut cursor = node.walk();
        for child in node.children_by_field_name("stmt", &mut cursor) {
            stmts.push(self.gen_stmt(child));
        }
        if self.recovering {
            for child in node
                .named_children(&mut node.walk())
                .filter(|n| n.is_error())
            {
                stmts.push(Spanned::new(
                    StmtKind::Expr {
                        term: self.hole(child),
                    },
                    self.span(child),
                ));
            }
            if node.is_error() && stmts.is_empty() {
                stmts.push(Spanned::new(
                    StmtKind::Expr {
                        term: self.hole(node),
                    },
                    self.span(node),
                ));
            }
        }
        let exports = node
            .child_by_field_name("exports")
            .map_or_else(Vec::new, |clause| {
                clause
                    .children_by_field_name("name", &mut clause.walk())
                    .filter(|node| !node.has_error())
                    .map(|node| self.ident(node))
                    .collect()
            });
        let imports = node
            .child_by_field_name("imports")
            .map_or_else(Vec::new, |clause| {
                clause
                    .children_by_field_name("path", &mut clause.walk())
                    .filter(|node| !node.has_error())
                    .map(|node| {
                        Spanned::new(decode_string(self.text(node)).into(), self.span(node))
                    })
                    .collect()
            });
        Ok(SourceFile {
            exports,
            imports,
            stmts,
        })
    }

    fn parse_error(&self, node: Node) -> AstError {
        let span = self.span(node);
        if node.is_missing() {
            AstError {
                span,
                kind: AstErrorKind::Missing {
                    expected: Arc::from(node.kind()),
                },
            }
        } else {
            AstError {
                span,
                kind: AstErrorKind::Unexpected {
                    found: Arc::from(self.text(node)),
                },
            }
        }
    }

    fn gen_stmt(&self, node: Node) -> Stmt {
        if node.kind() == "impl_definition" {
            let Some(receiver) = node.child_by_field_name("receiver") else {
                assert!(self.recovering, "impl requires a receiver node");
                return Spanned::new(
                    StmtKind::Expr {
                        term: self.hole(node),
                    },
                    self.span(node),
                );
            };
            let receiver = self.ident(receiver);
            let methods = node
                .children_by_field_name("method", &mut node.walk())
                .map(|method| {
                    let mut stmt = self.gen_function(method);
                    if let StmtKind::Function {
                        receiver: target,
                        name,
                        ..
                    } = &mut stmt.val
                    {
                        name.val = format!("{}.{}", receiver.val, name.val).into();
                        *target = Some(receiver.clone());
                    }
                    stmt
                })
                .collect();
            return Spanned::new(StmtKind::Impl { receiver, methods }, self.span(node));
        }
        if node.kind() == "struct_definition" {
            let fields = node
                .children_by_field_name("fields", &mut node.walk())
                .map(|field| self.gen_declare(field))
                .collect();
            return Spanned::new(
                StmtKind::Struct {
                    name: self.ident(node.child_by_field_name("name").unwrap_or(node)),
                    body: Spanned::new(TypeKind::Record { fields }, self.span(node)),
                },
                self.span(node),
            );
        }
        if let Some(definition) = node.child_by_field_name("struct") {
            return self.gen_stmt(definition);
        }
        if node.kind() == "type_definition" {
            return self.gen_type_define(
                node.child_by_field_name("definition").unwrap_or(node),
                self.span(node),
            );
        }
        if matches!(node.kind(), "function_definition" | "foreign_function") {
            return self.gen_function(node);
        }
        if node.kind() == "foreign_type" {
            return Spanned::new(
                StmtKind::ForeignType {
                    name: self.ident(node.child_by_field_name("name").unwrap_or(node)),
                },
                self.span(node),
            );
        }
        if self.recovering && node.kind() != "statement" {
            return Spanned::new(
                StmtKind::Expr {
                    term: self.hole(node),
                },
                self.span(node),
            );
        }
        assert_eq!(node.kind(), "statement");
        if let Some(body) = node.child_by_field_name("defer") {
            return Spanned::new(
                StmtKind::Defer {
                    body: Arc::new(self.trailing_errors(self.gen_term(body), node)),
                },
                self.span(node),
            );
        }
        if let Some(define) = node.child_by_field_name("define") {
            let mut stmt = self.gen_define(define, self.span(node));
            if let StmtKind::Define { init, .. } = &mut stmt.val {
                *init = self.trailing_errors(init.clone(), node);
            }
            return stmt;
        }
        if let Some(declare) = node.child_by_field_name("declare") {
            let name = self.ident(declare.child_by_field_name("name").unwrap_or(node));
            let ann = self.gen_type(declare.child_by_field_name("ann").unwrap_or(node));
            return Spanned::new(StmtKind::Declare { name, ann }, self.span(node));
        }
        let expr = self.trailing_errors(
            self.gen_term(node.child_by_field_name("expr").unwrap_or(node)),
            node,
        );
        Spanned::new(StmtKind::Expr { term: expr }, self.span(node))
    }

    fn gen_define(&self, node: Node, span: Span) -> Stmt {
        if let Some(term_def) = node.child_by_field_name("term") {
            return self.gen_term_define(term_def, span);
        }
        let type_def = node.child_by_field_name("type").unwrap_or(node);
        self.gen_type_define(type_def, span)
    }

    fn gen_term_define(&self, node: Node, span: Span) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap_or(node));
        let init = self.gen_term(node.child_by_field_name("init").unwrap_or(node));
        Spanned::new(StmtKind::Define { name, init }, span)
    }

    fn gen_type_define(&self, node: Node, span: Span) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap_or(node));
        let init = self.gen_type(node.child_by_field_name("init").unwrap_or(node));
        Spanned::new(StmtKind::DefineType { name, init }, span)
    }

    fn gen_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "term" || node.is_missing() || node.is_error()) {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "term");
        self.gen_assignment_term(node.child(0).unwrap_or(node))
    }

    fn gen_assignment_term(&self, node: Node) -> Term {
        if self.recovering
            && (node.kind() != "assignment_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "assignment_term");
        if let Some(value) = node.child_by_field_name("value") {
            let place = self.gen_binary_term(node.child_by_field_name("place").unwrap_or(node));
            let value = self.gen_assignment_term(value);
            return Spanned::new(
                TermKind::Assign {
                    place: Box::new(place),
                    value: Box::new(value),
                },
                self.span(node),
            );
        }
        self.gen_binary_term(node.child(0).unwrap_or(node))
    }

    fn gen_binary_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "binary_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "binary_term");
        if let Some(op_node) = node.child_by_field_name("operator") {
            let op_text = self.text(op_node);
            let left = node
                .child_by_field_name("left")
                .map(|n| self.gen_binary_term(n))
                .unwrap_or_else(|| self.hole(node));
            let right = node
                .child_by_field_name("right")
                .map(|n| self.gen_binary_term(n))
                .unwrap_or_else(|| self.hole(node));
            return self.call_var(op_text, vec![left, right], self.span(node));
        }
        self.gen_unary_term(node.child(0).unwrap_or(node))
    }

    fn gen_unary_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "unary_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "unary_term");
        if let Some(op_node) = node.child_by_field_name("operator") {
            let op_text = self.text(op_node);
            let operand = node
                .child_by_field_name("operand")
                .map(|n| self.gen_unary_term(n))
                .unwrap_or_else(|| self.hole(node));
            if op_text == "&" {
                return Spanned::new(
                    TermKind::Address {
                        place: Box::new(operand),
                    },
                    self.span(node),
                );
            }
            return self.call_var(op_text, vec![operand], self.span(node));
        }
        self.gen_postfix_term(node.child(0).unwrap_or(node))
    }

    fn gen_postfix_term(&self, node: Node) -> Term {
        if self.recovering
            && (node.kind() != "postfix_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "postfix_term");
        let mut base = self.gen_primary_term(node.child_by_field_name("prefix").unwrap_or(node));
        let mut cursor = node.walk();
        for child in node.children_by_field_name("suffix", &mut cursor) {
            match child.kind() {
                "unwrap_suffix" => {
                    let span = Span {
                        start: base.span.start,
                        end: child.end_byte(),
                    };
                    base = Spanned::new(
                        TermKind::Unwrap {
                            value: Box::new(base),
                        },
                        span,
                    );
                }
                "try_suffix" => {
                    let span = Span {
                        start: base.span.start,
                        end: child.end_byte(),
                    };
                    base = Spanned::new(
                        TermKind::Try {
                            value: Box::new(base),
                        },
                        span,
                    );
                }
                "method_call" => {
                    let name = self.ident(child.child_by_field_name("name").unwrap());
                    let args = child.child_by_field_name("args").unwrap();
                    let arg = match args.kind() {
                        "paren_term" => self.gen_paren_term(args),
                        "tuple_term" => self.gen_tuple_term(args),
                        "unit_term" => Spanned::new(TermKind::Unit, self.span(args)),
                        _ => unreachable!("method arguments"),
                    };
                    let span = Span {
                        start: base.span.start,
                        end: child.end_byte(),
                    };
                    base = Spanned::new(
                        TermKind::MethodCall {
                            receiver: Box::new(base),
                            name,
                            arg: Box::new(arg),
                        },
                        span,
                    );
                }
                "field_access" => {
                    let field = child.child_by_field_name("name");
                    if self.recovering && field.is_none_or(|n| n.is_missing()) {
                        let span = Span {
                            start: base.span.start,
                            end: child.end_byte().min(self.source_len),
                        };
                        base = Spanned::new(
                            TermKind::FieldHole {
                                base: Box::new(base),
                            },
                            span,
                        );
                        continue;
                    }
                    let name = self.ident(field.unwrap());
                    let span = Span {
                        start: base.span.start,
                        end: child.end_byte(),
                    };
                    base = Spanned::new(
                        TermKind::Field {
                            base: Box::new(base),
                            name,
                        },
                        span,
                    );
                }
                "closed_term" => {
                    let arg = self.gen_closed_term(child);
                    base = Spanned::new(
                        TermKind::Call {
                            func: Box::new(base),
                            arg: Box::new(arg),
                        },
                        self.span(node),
                    );
                }
                "pointer_deref" => {
                    let span = Span {
                        start: base.span.start,
                        end: child.end_byte(),
                    };
                    base = Spanned::new(
                        TermKind::Deref {
                            pointer: Box::new(base),
                        },
                        span,
                    );
                }
                _ => unreachable!("unexpected suffix: {}", child.kind()),
            }
        }
        base
    }

    fn gen_primary_term(&self, node: Node) -> Term {
        if self.recovering
            && (node.kind() != "primary_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "primary_term");
        let child = node.child(0).unwrap_or(node);
        if self.recovering
            && child.has_error()
            && matches!(child.kind(), "lid" | "number" | "string")
        {
            return self.hole(node);
        }
        let span = self.span(node);
        match child.kind() {
            "closed_term" => self.gen_closed_term(child),
            "lid" => Spanned::new(
                TermKind::Var {
                    name: self.ident(child),
                },
                span,
            ),
            "unary_type" if self.text(child) == "None" => Spanned::new(TermKind::None, span),
            "number" => Spanned::new(
                TermKind::Num {
                    value: self.text(child).into(),
                },
                span,
            ),
            "if_term" => self.gen_if_term(child),
            "match_term" => {
                let value = self.gen_term(child.child_by_field_name("value").unwrap_or(child));
                let arms = child
                    .children_by_field_name("arms", &mut child.walk())
                    .map(|arm| {
                        let variant = arm.child_by_field_name("variant").unwrap_or(arm);
                        let variant = match variant.kind() {
                            "None" => MatchVariant::Type(Spanned::new(
                                TypeKind::Atom {
                                    name: Ident::new("None".into(), self.span(variant)),
                                },
                                self.span(variant),
                            )),
                            "ok" => MatchVariant::Ok,
                            "err" => MatchVariant::Err,
                            _ => MatchVariant::Type(self.gen_type(variant)),
                        };
                        MatchArm {
                            variant,
                            name: arm.child_by_field_name("name").map(|node| self.ident(node)),
                            body: self.gen_body(arm.child_by_field_name("body").unwrap_or(arm)),
                        }
                    })
                    .collect();
                Spanned::new(
                    TermKind::Match {
                        value: Box::new(value),
                        arms,
                    },
                    span,
                )
            }
            "while_term" => Spanned::new(
                TermKind::While {
                    cond: Box::new(
                        self.gen_term(child.child_by_field_name("cond").unwrap_or(node)),
                    ),
                    body: Box::new(
                        self.gen_body(child.child_by_field_name("body").unwrap_or(node)),
                    ),
                },
                span,
            ),
            "string" => Spanned::new(
                TermKind::String {
                    value: decode_string(self.text(child)).into(),
                },
                span,
            ),
            "unary_type" => {
                let ty = self.gen_unary_type(child);
                Spanned::new(TermKind::Type { ty }, span)
            }
            _ if self.recovering => self.hole(node),
            _ => unreachable!("unexpected primary: {}", child.kind()),
        }
    }

    fn gen_closed_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "closed_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "closed_term");
        let child = node.child(0).unwrap_or(node);
        match child.kind() {
            "paren_term" => self.gen_paren_term(child),
            "tuple_term" => self.gen_tuple_term(child),
            "array_term" => self.gen_array_term(child),
            "record_term" => self.gen_record_term(child),
            "chain_term" => self.gen_chain_term(child),
            "unit_term" if self.recovering && child.has_error() => self.hole(child),
            "unit_term" => Spanned::new(TermKind::Unit, self.span(node)),
            _ if self.recovering => self.hole(node),
            _ => unreachable!("unexpected closed: {}", child.kind()),
        }
    }

    fn gen_function(&self, node: Node) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap_or(node));
        let result = node
            .child_by_field_name("result")
            .map(|result| self.gen_type(result))
            .unwrap_or_else(|| Spanned::new(TypeKind::Unit, name.span));
        let mut params = Vec::new();
        let mut cursor = node.walk();
        for p in node.children_by_field_name("params", &mut cursor) {
            let (name, ann) = self.gen_declare(p);
            params.push((name, ann));
        }
        if let Some(header) = node.child_by_field_name("header") {
            return Spanned::new(
                StmtKind::ForeignFunction {
                    header: decode_string(self.text(header)).into(),
                    name,
                    params,
                    result,
                },
                self.span(node),
            );
        }
        let body = node
            .child_by_field_name("body")
            .map(|n| self.gen_body(n))
            .unwrap_or_else(|| self.hole(node));
        Spanned::new(
            StmtKind::Function {
                receiver: None,
                decorators: node
                    .children_by_field_name("decorator", &mut node.walk())
                    .filter_map(|n| n.child_by_field_name("name"))
                    .map(|n| self.ident(n))
                    .collect(),
                name,
                params,
                result,
                body,
            },
            self.span(node),
        )
    }

    fn gen_body(&self, node: Node) -> Term {
        self.gen_block(node, "stmt")
    }

    fn gen_block(&self, node: Node, statement_field: &str) -> Term {
        let mut cursor = node.walk();
        let stmts = node
            .children_by_field_name(statement_field, &mut cursor)
            .map(|stmt| self.gen_stmt(stmt))
            .collect();
        let tail = node
            .child_by_field_name("tail")
            .map(|term| self.gen_term(term))
            .unwrap_or_else(|| Spanned::new(TermKind::Unit, self.span(node)));
        let tail = self.trailing_errors(tail, node);
        Spanned::new(
            TermKind::Block {
                stmts,
                tail: Box::new(tail),
            },
            self.span(node),
        )
    }

    fn gen_paren_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "paren_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "paren_term");
        self.gen_term(node.child_by_field_name("inner").unwrap_or(node))
    }

    fn gen_tuple_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "tuple_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "tuple_term");
        let elems = self.field_children_terms(node, "elems");
        let span = self.span(node);
        let fields = elems
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                let name = Spanned::new(format!("_{i}").into(), span);
                (name, val)
            })
            .collect();
        Spanned::new(TermKind::Record { fields }, span)
    }

    fn gen_array_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "array_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "array_term");
        let elems = self.field_children_terms(node, "elems");
        Spanned::new(TermKind::Array { elems }, self.span(node))
    }

    fn gen_record_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "record_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "record_term");
        let mut fields = Vec::new();
        let mut cursor = node.walk();
        for f in node.children_by_field_name("fields", &mut cursor) {
            assert_eq!(f.kind(), "term_define");
            let name = self.ident(f.child_by_field_name("name").unwrap_or(node));
            let init = self.gen_term(f.child_by_field_name("init").unwrap_or(node));
            fields.push((name, init));
        }
        Spanned::new(TermKind::Record { fields }, self.span(node))
    }

    fn gen_chain_term(&self, node: Node) -> Term {
        if self.recovering && (node.kind() != "chain_term" || node.is_missing() || node.is_error())
        {
            return self.hole(node);
        }
        assert_eq!(node.kind(), "chain_term");
        self.gen_block(node, "prefix")
    }

    fn gen_if_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "if_term");
        let cond = self.gen_term(node.child_by_field_name("cond").unwrap_or(node));
        let then = self.gen_closed_term(node.child_by_field_name("then").unwrap_or(node));
        let els = node.child_by_field_name("else").map_or_else(
            || {
                Spanned::new(
                    TermKind::Unit,
                    Span {
                        start: node.end_byte(),
                        end: node.end_byte(),
                    },
                )
            },
            |node| self.gen_closed_term(node),
        );
        Spanned::new(
            TermKind::If {
                cond: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
            },
            self.span(node),
        )
    }

    fn gen_type(&self, node: Node) -> Type {
        if self.recovering && (node.kind() != "type" || node.is_missing() || node.is_error()) {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "type");
        self.gen_infix_type(node.child(0).unwrap_or(node))
    }

    fn gen_infix_type(&self, node: Node) -> Type {
        if self.recovering && (node.kind() != "infix_type" || node.is_missing() || node.is_error())
        {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "infix_type");
        if let Some(ret) = node.child_by_field_name("ret_ty") {
            let from = self.gen_closed_type(node.child_by_field_name("param_ty").unwrap_or(node));
            let to = self.gen_infix_type(ret);
            return Spanned::new(
                TypeKind::Func {
                    from: Box::new(from),
                    to: Box::new(to),
                },
                self.span(node),
            );
        }
        self.gen_union_type(node.child(0).unwrap_or(node))
    }

    fn gen_union_type(&self, node: Node) -> Type {
        if let Some(left) = node.child_by_field_name("left") {
            return Spanned::new(
                TypeKind::Union {
                    left: Box::new(self.gen_union_type(left)),
                    right: Box::new(
                        self.gen_unary_type(node.child_by_field_name("right").unwrap_or(node)),
                    ),
                },
                self.span(node),
            );
        }
        self.gen_unary_type(node.child(0).unwrap_or(node))
    }

    fn gen_unary_type(&self, node: Node) -> Type {
        if self.recovering && (node.kind() != "unary_type" || node.is_missing() || node.is_error())
        {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "unary_type");
        if let Some(value) = node.child_by_field_name("value") {
            return Spanned::new(
                TypeKind::Result {
                    value: Box::new(self.gen_type(value)),
                    error: Box::new(
                        self.gen_type(node.child_by_field_name("error").unwrap_or(node)),
                    ),
                },
                self.span(node),
            );
        }
        if let Some(former) = node.child_by_field_name("former") {
            let head = self.ident(former);
            let arg = self.gen_type(node.child_by_field_name("arg").unwrap_or(node));
            return Spanned::new(
                TypeKind::App {
                    head,
                    arg: Box::new(arg),
                },
                self.span(node),
            );
        }
        self.gen_primary_type(node.child(0).unwrap_or(node))
    }

    fn gen_primary_type(&self, node: Node) -> Type {
        if self.recovering
            && (node.kind() != "primary_type" || node.is_missing() || node.is_error())
        {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "primary_type");
        let child = node.child(0).unwrap_or(node);
        if self.recovering && child.has_error() {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        match child.kind() {
            "inferred_type" => Spanned::new(TypeKind::Infer, self.span(child)),
            "uid" => Spanned::new(
                TypeKind::Atom {
                    name: self.ident(child),
                },
                self.span(node),
            ),
            "builtin_type" => Spanned::new(
                TypeKind::Atom {
                    name: self.ident(child),
                },
                self.span(node),
            ),
            "closed_type" => self.gen_closed_type(child),
            _ if self.recovering => Spanned::new(TypeKind::Hole, self.span(node)),
            _ => unreachable!("unexpected primary_type: {}", child.kind()),
        }
    }

    fn gen_closed_type(&self, node: Node) -> Type {
        if self.recovering && (node.kind() != "closed_type" || node.is_missing() || node.is_error())
        {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "closed_type");
        let child = node.child(0).unwrap_or(node);
        match child.kind() {
            "paren_type" => self.gen_type(child.child_by_field_name("inner").unwrap_or(node)),
            "unit_type" => Spanned::new(TypeKind::Unit, self.span(child)),
            "tuple_type" => {
                let mut cursor = child.walk();
                let fields = child
                    .children_by_field_name("elems", &mut cursor)
                    .enumerate()
                    .map(|(i, elem)| {
                        (
                            Spanned::new(format!("_{i}").into(), self.span(elem)),
                            self.gen_type(elem),
                        )
                    })
                    .collect();
                Spanned::new(TypeKind::Record { fields }, self.span(child))
            }
            "record_type" => self.gen_record_type(child),
            _ if self.recovering => Spanned::new(TypeKind::Hole, self.span(node)),
            _ => unreachable!("unexpected closed_type: {}", child.kind()),
        }
    }

    fn gen_record_type(&self, node: Node) -> Type {
        if self.recovering && (node.kind() != "record_type" || node.is_missing() || node.is_error())
        {
            return Spanned::new(TypeKind::Hole, self.span(node));
        }
        assert_eq!(node.kind(), "record_type");
        let mut fields = Vec::new();
        let mut cursor = node.walk();
        for f in node.children_by_field_name("field", &mut cursor) {
            let (name, ann) = self.gen_declare(f);
            fields.push((name, ann));
        }
        Spanned::new(TypeKind::Record { fields }, self.span(node))
    }

    fn gen_declare(&self, node: Node) -> (Ident, Type) {
        assert_eq!(node.kind(), "declare");
        let name = self.ident(node.child_by_field_name("name").unwrap_or(node));
        let ann = self.gen_type(node.child_by_field_name("ann").unwrap_or(node));
        (name, ann)
    }

    fn field_children_terms(&self, node: Node, field: &str) -> Vec<Term> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children_by_field_name(field, &mut cursor) {
            out.push(self.gen_term(child));
        }
        out
    }

    fn call_var(&self, op_text: &str, args: Vec<Term>, span: Span) -> Term {
        Spanned::new(
            TermKind::Builtin {
                name: op_text.into(),
                args,
            },
            span,
        )
    }

    fn ident(&self, node: Node) -> Ident {
        let text = if self.recovering
            && (node.is_missing()
                || !matches!(node.kind(), "lid" | "uid" | "builtin_type" | "Ptr" | "Span"))
        {
            ""
        } else {
            self.text(node)
        };
        Spanned::new(text.into(), self.span(node))
    }

    fn text(&self, node: Node) -> &str {
        &self.src[node.byte_range()]
    }

    fn span(&self, node: Node) -> Span {
        Span {
            start: node.start_byte().min(self.source_len),
            end: node.end_byte().min(self.source_len),
        }
    }
}

fn decode_string(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text[1..text.len() - 1].chars();
    while let Some(ch) = chars.next() {
        result.push(if ch == '\\' {
            match chars.next().expect("validated string escape") {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '0' => '\0',
                '"' => '"',
                '\\' => '\\',
                _ => unreachable!("grammar rejects unknown escapes"),
            }
        } else {
            ch
        });
    }
    result
}

fn first_error_node(node: Node) -> Option<Node> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(error) = first_error_node(child) {
            return Some(error);
        }
    }
    if node.is_error() || node.is_missing() {
        Some(node)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse_err(src: &str) -> AstError {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .expect("failed to load Resin grammar");
        let tree = parser.parse(src, None).expect("parser returned no tree");
        AstGen::new(src)
            .gen_source_file(tree.root_node())
            .expect_err("expected a parse error")
    }

    #[test]
    fn unexpected_syntax_is_a_parse_error() {
        let err = parse_err("def main() -> () = { var x = ; };");
        assert!(matches!(
            err.kind,
            AstErrorKind::Unexpected { .. } | AstErrorKind::Missing { .. }
        ));
        assert!(err.span.start <= err.span.end);
    }
}
