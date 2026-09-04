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

    pub fn gen_source_file(&self, node: Node) -> SourceFile {
        assert_eq!(node.kind(), "source_file");
        let mut stmts = Vec::new();
        let mut cursor = node.walk();
        for child in node.children_by_field_name("stmt", &mut cursor) {
            stmts.push(self.gen_stmt(child));
        }
        SourceFile { stmts }
    }

    fn gen_stmt(&self, node: Node) -> Stmt {
        assert_eq!(node.kind(), "statement");
        let define = node.child_by_field_name("define").unwrap();
        self.gen_define(define)
    }

    fn gen_define(&self, node: Node) -> Stmt {
        assert_eq!(node.kind(), "define");
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let init = self.gen_term(node.child_by_field_name("init").unwrap());
        Spanned::new(StmtKind { name, init }, self.span(node))
    }

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
            "ident" => Spanned::new(TermKind::Var(self.ident(child)), span),
            "number" => Spanned::new(TermKind::Num(self.text(child).into()), span),
            "if_term" => self.gen_if_term(child),
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
            params.push((name, Box::new(ann)));
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
        // Parentheses are grouping only. Unwrap to the inner term.
        self.gen_term(node.child_by_field_name("inner").unwrap())
    }

    fn gen_tuple_term(&self, node: Node) -> Term {
        assert_eq!(node.kind(), "tuple_term");
        let elems = self.field_children_terms(node, "elems");
        Spanned::new(TermKind::Tuple(elems), self.span(node))
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
            let name = self.ident(f.child_by_field_name("name").unwrap());
            let init = self.gen_term(f.child_by_field_name("init").unwrap());
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

    // -- helpers ----------------------------------------------------

    fn gen_declare(&self, node: Node) -> (Ident, Term) {
        assert_eq!(node.kind(), "declare");
        let name = self.ident(node.child_by_field_name("name").unwrap());
        let ann = self.gen_term(node.child_by_field_name("ann").unwrap());
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

    /// `Call(Var(op_text), args)` with the given span.
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
