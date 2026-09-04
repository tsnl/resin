//! Tree-sitter parse tree → AST generation.

use tree_sitter::Node;

use crate::ast::*;

pub struct AstGen<'a> {
    src: &'a str,
}

impl<'a> AstGen<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src }
    }

    //
    // Source file
    //

    pub fn gen_source_file(&self, node: Node) -> SourceFile {
        assert_eq!(node.kind(), "source_file");
        let mut stmts = Vec::new();
        let mut cursor = node.walk();
        for child in node.children_by_field_name("stmt", &mut cursor) {
            stmts.push(self.gen_stmt(child));
        }
        SourceFile { stmts }
    }

    //
    // Statement
    //

    fn gen_stmt(&self, node: Node) -> Stmt {
        assert_eq!(node.kind(), "statement");
        if let Some(define) = node.child_by_field_name("define") {
            return self.gen_define(define, self.span(node));
        }
        let declare = node.child_by_field_name("declare").unwrap();
        let name = self.ident(declare.child_by_field_name("name").unwrap());
        let ann = self.gen_type(declare.child_by_field_name("ann").unwrap());
        Spanned::new(StmtKind::Declare { name, ann }, self.span(node))
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
        let ann = self.gen_type(node.child_by_field_name("init").unwrap());
        Spanned::new(StmtKind::Declare { name, ann }, span)
    }

    //
    // Term
    //

    fn gen_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "term");
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
                    let name = self.text(child.child_by_field_name("name").unwrap());
                    let field_op = format!(".{name}");
                    base = self.call_var(&field_op, vec![base], self.span(child));
                }
                "closed_term" => {
                    let arg = self.gen_closed_term(child);
                    base = Spanned::new(
                        TermKind::Call {
                            func: Box::new(base),
                            args: vec![arg],
                        },
                        self.span(node),
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
            "lid" => Spanned::new(TermKind::Var(self.ident(child)), span),
            "number" => Spanned::new(TermKind::Num(self.text(child).into()), span),
            "if_term" => self.gen_if_term(child),
            "unary_type" => {
                let ty = self.gen_unary_type(child);
                Spanned::new(TermKind::Type(ty), span)
            }
            _ => unreachable!("unexpected primary: {}", child.kind()),
        }
    }

    fn gen_closed_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "closed_term");
        let child = node.child(0).unwrap();
        match child.kind() {
            "lambda_term" => self.gen_lambda_term(child),
            "paren_term" => self.gen_paren_term(child),
            "tuple_term" => self.gen_tuple_term(child),
            "array_term" => self.gen_array_term(child),
            "record_term" => self.gen_record_term(child),
            "chain_term" => self.gen_chain_term(child),
            "unit_term" => Spanned::new(TermKind::Unit, self.span(node)),
            _ => unreachable!("unexpected closed: {}", child.kind()),
        }
    }

    fn gen_lambda_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "lambda_term");
        let mut params = Vec::new();
        let mut cursor = node.walk();
        for p in node.children_by_field_name("params", &mut cursor) {
            let (name, ann) = self.gen_declare(p);
            params.push((name, ann));
        }
        let body = self.gen_postfix_term(node.child_by_field_name("body").unwrap());
        Spanned::new(
            TermKind::Lambda {
                params,
                body: Box::new(body),
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
        Spanned::new(TermKind::Record(fields), span)
    }

    fn gen_array_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "array_term");
        let elems = self.field_children_terms(node, "elems");
        Spanned::new(TermKind::Array(elems), self.span(node))
    }

    fn gen_record_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "record_term");
        let mut fields = Vec::new();
        let mut cursor = node.walk();
        for f in node.children_by_field_name("fields", &mut cursor) {
            assert_eq!(f.kind(), "define");
            let term_def = f.child_by_field_name("term").unwrap();
            let name = self.ident(term_def.child_by_field_name("name").unwrap());
            let init = self.gen_term(term_def.child_by_field_name("init").unwrap());
            fields.push((name, init));
        }
        Spanned::new(TermKind::Record(fields), self.span(node))
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

    //
    // Type
    //

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
            let arg_node = node.child_by_field_name("arg").unwrap();
            let arg = match arg_node.kind() {
                "closed_term" => self.gen_closed_term(arg_node),
                "closed_type" => {
                    let ty = self.gen_closed_type(arg_node);
                    Spanned::new(TermKind::Type(ty), self.span(arg_node))
                }
                _ => unreachable!("unexpected type arg: {}", arg_node.kind()),
            };
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
            "uid" => Spanned::new(TypeKind::Atom(self.ident(child)), self.span(node)),
            "builtin_type" => Spanned::new(TypeKind::Atom(self.ident(child)), self.span(node)),
            "closed_type" => self.gen_closed_type(child),
            _ => unreachable!("unexpected primary_type: {}", child.kind()),
        }
    }

    fn gen_closed_type(&self, node: Node) -> Type {
        assert_eq!(node.kind(), "closed_type");
        let child = node.child(0).unwrap();
        match child.kind() {
            "paren_type" => self.gen_type(child.child_by_field_name("inner").unwrap()),
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
        Spanned::new(TypeKind::Record(fields), self.span(node))
    }

    //
    // Helpers
    //

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
        let func = Spanned::new(TermKind::Var(Spanned::new(op_text.into(), span)), span);
        Spanned::new(
            TermKind::Call {
                func: Box::new(func),
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
