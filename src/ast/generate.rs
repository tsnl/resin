//! Tree-sitter parse tree → AST generation.

use std::{fmt, sync::Arc};

use tree_sitter::Node;

use super::*;

pub struct AstGen<'a> {
    src: &'a str,
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
        Self { src }
    }

    pub fn gen_source_file(&self, node: Node) -> Result<SourceFile, AstError> {
        if let Some(error) = first_error_node(node) {
            return Err(self.parse_error(error));
        }
        assert_eq!(node.kind(), "source_file");
        let mut stmts = Vec::new();
        let mut cursor = node.walk();
        for child in node.children_by_field_name("stmt", &mut cursor) {
            stmts.push(self.gen_stmt(child));
        }
        let exports = node
            .child_by_field_name("exports")
            .map_or_else(Vec::new, |clause| {
                clause
                    .children_by_field_name("name", &mut clause.walk())
                    .map(|node| self.ident(node))
                    .collect()
            });
        let imports = node
            .child_by_field_name("imports")
            .map_or_else(Vec::new, |clause| {
                clause
                    .children_by_field_name("path", &mut clause.walk())
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
        if matches!(node.kind(), "function_definition" | "foreign_function") {
            return self.gen_function(node);
        }
        if node.kind() == "foreign_type" {
            return Spanned::new(
                StmtKind::ForeignType {
                    name: self.ident(node.child_by_field_name("name").unwrap()),
                },
                self.span(node),
            );
        }
        assert_eq!(node.kind(), "statement");
        if let Some(define) = node.child_by_field_name("define") {
            return self.gen_define(define, self.span(node));
        }
        if let Some(declare) = node.child_by_field_name("declare") {
            let name = self.ident(declare.child_by_field_name("name").unwrap());
            let ann = self.gen_type(declare.child_by_field_name("ann").unwrap());
            return Spanned::new(StmtKind::Declare { name, ann }, self.span(node));
        }
        let expr = self.gen_term(node.child_by_field_name("expr").unwrap());
        Spanned::new(StmtKind::Expr { term: expr }, self.span(node))
    }

    fn gen_define(&self, node: Node, span: Span) -> Stmt {
        if let Some(term_def) = node.child_by_field_name("term") {
            return self.gen_term_define(term_def, span);
        }
        let type_def = node.child_by_field_name("type").unwrap();
        self.gen_type_define(type_def, span)
    }

    fn gen_term_define(&self, node: Node, span: Span) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let init = self.gen_term(node.child_by_field_name("init").unwrap());
        Spanned::new(StmtKind::Define { name, init }, span)
    }

    fn gen_type_define(&self, node: Node, span: Span) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let init = self.gen_type(node.child_by_field_name("init").unwrap());
        Spanned::new(StmtKind::DefineType { name, init }, span)
    }

    fn gen_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "term");
        self.gen_assignment_term(node.child(0).unwrap())
    }

    fn gen_assignment_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "assignment_term");
        if let Some(value) = node.child_by_field_name("value") {
            let place = self.gen_binary_term(node.child_by_field_name("place").unwrap());
            let value = self.gen_assignment_term(value);
            return Spanned::new(
                TermKind::Assign {
                    place: Box::new(place),
                    value: Box::new(value),
                },
                self.span(node),
            );
        }
        self.gen_binary_term(node.child(0).unwrap())
    }

    fn gen_binary_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "binary_term");
        if let Some(op_node) = node.child_by_field_name("operator") {
            let op_text = self.text(op_node);
            let left = self.gen_binary_term(node.child_by_field_name("left").unwrap());
            let right = self.gen_binary_term(node.child_by_field_name("right").unwrap());
            return self.call_var(op_text, vec![left, right], self.span(node));
        }
        self.gen_unary_term(node.child(0).unwrap())
    }

    fn gen_unary_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "unary_term");
        if let Some(op_node) = node.child_by_field_name("operator") {
            let op_text = self.text(op_node);
            let operand = self.gen_unary_term(node.child_by_field_name("operand").unwrap());
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
        self.gen_postfix_term(node.child(0).unwrap())
    }

    fn gen_postfix_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "postfix_term");
        let mut base = self.gen_primary_term(node.child_by_field_name("prefix").unwrap());
        let mut cursor = node.walk();
        for child in node.children_by_field_name("suffix", &mut cursor) {
            match child.kind() {
                "field_access" => {
                    let name = self.ident(child.child_by_field_name("name").unwrap());
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
        assert_eq!(node.kind(), "primary_term");
        let child = node.child(0).unwrap();
        let span = self.span(node);
        match child.kind() {
            "closed_term" => self.gen_closed_term(child),
            "lid" => Spanned::new(
                TermKind::Var {
                    name: self.ident(child),
                },
                span,
            ),
            "number" => Spanned::new(
                TermKind::Num {
                    value: self.text(child).into(),
                },
                span,
            ),
            "if_term" => self.gen_if_term(child),
            "while_term" => Spanned::new(
                TermKind::While {
                    cond: Box::new(self.gen_term(child.child_by_field_name("cond").unwrap())),
                    body: Box::new(self.gen_body(child.child_by_field_name("body").unwrap())),
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
            _ => unreachable!("unexpected primary: {}", child.kind()),
        }
    }

    fn gen_closed_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "closed_term");
        let child = node.child(0).unwrap();
        match child.kind() {
            "paren_term" => self.gen_paren_term(child),
            "tuple_term" => self.gen_tuple_term(child),
            "array_term" => self.gen_array_term(child),
            "record_term" => self.gen_record_term(child),
            "chain_term" => self.gen_chain_term(child),
            "unit_term" => Spanned::new(TermKind::Unit, self.span(node)),
            _ => unreachable!("unexpected closed: {}", child.kind()),
        }
    }

    fn gen_function(&self, node: Node) -> Stmt {
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let result = self.gen_type(node.child_by_field_name("result").unwrap());
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
        let body = self.gen_body(node.child_by_field_name("body").unwrap());
        Spanned::new(
            StmtKind::Function {
                name,
                params,
                result,
                body,
            },
            self.span(node),
        )
    }

    fn gen_body(&self, node: Node) -> Term {
        let mut cursor = node.walk();
        let stmts = node
            .children_by_field_name("stmt", &mut cursor)
            .map(|stmt| self.gen_stmt(stmt))
            .collect();
        let tail = node
            .child_by_field_name("tail")
            .map(|term| self.gen_term(term))
            .unwrap_or_else(|| Spanned::new(TermKind::Unit, self.span(node)));
        Spanned::new(
            TermKind::Block {
                stmts,
                tail: Box::new(tail),
            },
            self.span(node),
        )
    }

    fn gen_paren_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "paren_term");
        self.gen_term(node.child_by_field_name("inner").unwrap())
    }

    fn gen_tuple_term(&self, node: Node) -> Term {
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
        assert_eq!(node.kind(), "array_term");
        let elems = self.field_children_terms(node, "elems");
        Spanned::new(TermKind::Array { elems }, self.span(node))
    }

    fn gen_record_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "record_term");
        let mut fields = Vec::new();
        let mut cursor = node.walk();
        for f in node.children_by_field_name("fields", &mut cursor) {
            assert_eq!(f.kind(), "term_define");
            let name = self.ident(f.child_by_field_name("name").unwrap());
            let init = self.gen_term(f.child_by_field_name("init").unwrap());
            fields.push((name, init));
        }
        Spanned::new(TermKind::Record { fields }, self.span(node))
    }

    fn gen_chain_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "chain_term");
        let mut stmts = Vec::new();
        let mut cursor = node.walk();
        for s in node.children_by_field_name("prefix", &mut cursor) {
            stmts.push(self.gen_stmt(s));
        }
        let tail = self.gen_term(node.child_by_field_name("tail").unwrap());
        Spanned::new(
            TermKind::Block {
                stmts,
                tail: Box::new(tail),
            },
            self.span(node),
        )
    }

    fn gen_if_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "if_term");
        let cond = self.gen_term(node.child_by_field_name("cond").unwrap());
        let then = self.gen_closed_term(node.child_by_field_name("then").unwrap());
        let els = self.gen_closed_term(node.child_by_field_name("else").unwrap());
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
        assert_eq!(node.kind(), "type");
        self.gen_infix_type(node.child(0).unwrap())
    }

    fn gen_infix_type(&self, node: Node) -> Type {
        assert_eq!(node.kind(), "infix_type");
        if let Some(ret) = node.child_by_field_name("ret_ty") {
            let from = self.gen_closed_type(node.child_by_field_name("param_ty").unwrap());
            let to = self.gen_infix_type(ret);
            return Spanned::new(
                TypeKind::Func {
                    from: Box::new(from),
                    to: Box::new(to),
                },
                self.span(node),
            );
        }
        self.gen_unary_type(node.child(0).unwrap())
    }

    fn gen_unary_type(&self, node: Node) -> Type {
        assert_eq!(node.kind(), "unary_type");
        if let Some(former) = node.child_by_field_name("former") {
            let head = self.ident(former);
            let arg = self.gen_type(node.child_by_field_name("arg").unwrap());
            return Spanned::new(
                TypeKind::App {
                    head,
                    arg: Box::new(arg),
                },
                self.span(node),
            );
        }
        self.gen_primary_type(node.child(0).unwrap())
    }

    fn gen_primary_type(&self, node: Node) -> Type {
        assert_eq!(node.kind(), "primary_type");
        let child = node.child(0).unwrap();
        match child.kind() {
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
            _ => unreachable!("unexpected primary_type: {}", child.kind()),
        }
    }

    fn gen_closed_type(&self, node: Node) -> Type {
        assert_eq!(node.kind(), "closed_type");
        let child = node.child(0).unwrap();
        match child.kind() {
            "paren_type" => self.gen_type(child.child_by_field_name("inner").unwrap()),
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
            _ => unreachable!("unexpected closed_type: {}", child.kind()),
        }
    }

    fn gen_record_type(&self, node: Node) -> Type {
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
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let ann = self.gen_type(node.child_by_field_name("ann").unwrap());
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
        Spanned::new(self.text(node).into(), self.span(node))
    }

    fn text(&self, node: Node) -> &str {
        &self.src[node.byte_range()]
    }

    fn span(&self, node: Node) -> Span {
        Span {
            start: node.start_byte(),
            end: node.end_byte(),
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
        let err = parse_err("x = ;");
        assert!(matches!(
            err.kind,
            AstErrorKind::Unexpected { .. } | AstErrorKind::Missing { .. }
        ));
        assert!(err.span.start <= err.span.end);
    }
}
