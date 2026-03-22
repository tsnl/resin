use std::fmt;
use std::sync::Arc;

use hashbrown::HashMap;

use crate::ast::{self, Expr, Pattern, Stmt, Type};
use crate::token::{Token, TokenKind};
use crate::vocab::{self, LiteralBool};
use crate::{Span, Symbol, fb};

#[derive(Clone, Default)]
pub struct ParseError {
    pub expects: HashMap<&'static str, Vec<Expect>>,
}
#[derive(Clone)]
pub struct Expect {
    pub what: &'static str,
    pub span: Span,
}

impl ParseError {
    pub fn new(rule: &'static str, what: &'static str, span: Span) -> Self {
        let mut expects = HashMap::new();
        expects.insert(rule, vec![Expect { what, span }]);
        Self { expects }
    }
    pub fn merge(mut self, other: Self) -> Self {
        for (rule, other_exps) in other.expects {
            let entry = self.expects.entry(rule).or_default();
            for exp in other_exps {
                if !entry.iter().any(|e| e.what == exp.what) {
                    entry.push(exp);
                }
            }
        }
        self
    }
}

impl ParseError {
    fn into_outbox(self) -> fb::Outbox {
        let mut messages = vec![];
        for (rule, exps) in self.expects {
            let whats: Vec<_> = exps.iter().map(|e| e.what).collect();
            let title = format!("[ParseError] In `{rule}`: expected {}", join_or(&whats));
            let span = exps.first().map(|e| e.span.clone());
            messages.push(fb::Message {
                title,
                span,
                notes: vec![],
            });
        }
        fb::Outbox { messages }
    }
}

fn join_or(items: &[&str]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].to_string(),
        2 => format!("{} or {}", items[0], items[1]),
        _ => {
            let (last, rest) = items.split_last().unwrap();
            format!("{}, or {}", rest.join(", "), last)
        }
    }
}

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rules: Vec<_> = self.expects.keys().copied().collect();
        writeln!(f, "Parse error in rule(s): {}", rules.join(", "))?;
        for (rule, exps) in &self.expects {
            for exp in exps {
                writeln!(f, "  in `{rule}`: expected {} at {}", exp.what, exp.span)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

pub type PResult<T> = Result<(T, TokenStream), ParseError>;
type Variant = (Symbol, Option<Expr>);

#[derive(Clone)]
pub struct TokenStream {
    tokens: Arc<[Token]>,
    pos: usize,
}

impl TokenStream {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens: tokens.into(),
            pos: 0,
        }
    }
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    pub fn peek_kind(&self) -> Option<&TokenKind> {
        self.peek().map(|t| &t.kind)
    }
    pub fn advance(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            pos: self.pos + 1,
        }
    }
    pub fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }
    pub fn span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .or(self.tokens.last())
            .map(|t| t.span.clone())
            .expect("TokenStream::span on empty list")
    }
}

fn expect_if<T>(
    extract: impl Fn(&Token) -> Option<T> + 'static,
    rule: &'static str,
    what: &'static str,
) -> impl Fn(TokenStream) -> PResult<T> {
    move |ts: TokenStream| match ts.peek().and_then(&extract) {
        Some(v) => Ok((v, ts.advance())),
        None => Err(ParseError::new(rule, what, ts.span())),
    }
}

fn tok(expected: TokenKind, rule: &'static str) -> impl Fn(TokenStream) -> PResult<Span> {
    let what = expected.name();
    expect_if(
        move |t| {
            if t.kind == expected {
                Some(t.span.clone())
            } else {
                None
            }
        },
        rule,
        what,
    )
}

fn expect_name(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(Symbol, Span)> {
    expect_if(
        |t| match &t.kind {
            TokenKind::Lid(s) | TokenKind::Uid(s) | TokenKind::Hole(s) => {
                Some((s.clone(), t.span.clone()))
            }
            _ => None,
        },
        rule,
        "<name>",
    )
}

fn expect_name_or_builtin(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(Symbol, Span)> {
    expect_if(
        |t| match &t.kind {
            TokenKind::Lid(s) | TokenKind::Uid(s) | TokenKind::Hole(s) => {
                Some((s.clone(), t.span.clone()))
            }
            _ => t
                .kind
                .builtin_name()
                .map(|name| (Symbol::from(name), t.span.clone())),
        },
        rule,
        "<name>",
    )
}

fn expect_lid(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(Symbol, Span)> {
    expect_if(
        |t| match &t.kind {
            TokenKind::Lid(s) => Some((s.clone(), t.span.clone())),
            _ => None,
        },
        rule,
        "<Lid>",
    )
}

fn expect_uid(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(Symbol, Span)> {
    expect_if(
        |t| match &t.kind {
            TokenKind::Uid(s) => Some((s.clone(), t.span.clone())),
            _ => None,
        },
        rule,
        "<Uid>",
    )
}

fn expect_hole(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(Symbol, Span)> {
    expect_if(
        |t| match &t.kind {
            TokenKind::Hole(s) => Some((s.clone(), t.span.clone())),
            _ => None,
        },
        rule,
        "<Hole>",
    )
}

fn expect_literal(rule: &'static str) -> impl Fn(TokenStream) -> PResult<(vocab::Literal, Span)> {
    expect_if(
        move |t| match &t.kind {
            TokenKind::LiteralNumber(n) => {
                Some((vocab::Literal::Number((**n).clone()), t.span.clone()))
            }
            TokenKind::LiteralString(s) => {
                Some((vocab::Literal::String((**s).clone()), t.span.clone()))
            }
            TokenKind::KwTrue => Some((
                vocab::Literal::Bool(LiteralBool { value: true }),
                t.span.clone(),
            )),
            TokenKind::KwFalse => Some((
                vocab::Literal::Bool(LiteralBool { value: false }),
                t.span.clone(),
            )),
            _ => None,
        },
        rule,
        "<literal>",
    )
}

fn many0<T>(
    mut ts: TokenStream,
    parser: impl Fn(TokenStream) -> PResult<T>,
) -> (Vec<T>, TokenStream) {
    let mut results = vec![];
    while let Ok((item, new_ts)) = parser(ts.clone()) {
        results.push(item);
        ts = new_ts;
    }
    (results, ts)
}

fn opt<T>(
    ts: TokenStream,
    parser: impl FnOnce(TokenStream) -> PResult<T>,
) -> (Option<T>, TokenStream) {
    match parser(ts.clone()) {
        Ok((item, ts)) => (Some(item), ts),
        Err(_) => (None, ts),
    }
}

fn sep_by<T, S>(
    ts: TokenStream,
    item: impl Fn(TokenStream) -> PResult<T>,
    sep: impl Fn(TokenStream) -> PResult<S>,
) -> (Vec<T>, TokenStream) {
    let mut results = vec![];
    let mut ts = ts;
    match item(ts.clone()) {
        Ok((first, new_ts)) => {
            results.push(first);
            ts = new_ts;
        }
        Err(_) => return (results, ts),
    }
    while let Ok((_, after_sep)) = sep(ts.clone()) {
        match item(after_sep.clone()) {
            Ok((next, new_ts)) => {
                results.push(next);
                ts = new_ts;
            }
            Err(_) => {
                ts = after_sep;
                break;
            }
        }
    }
    (results, ts)
}

fn skip_eols(mut ts: TokenStream) -> TokenStream {
    while ts.peek_kind() == Some(&TokenKind::Eol) {
        ts.pos += 1;
    }
    ts
}

macro_rules! alt {
    ($ts:expr, [$($parser:expr),+ $(,)?]) => {{
        let ts = $ts;
        let mut _err = ParseError::default();
        $(
            match $parser(ts.clone()) {
                ok @ Ok(_) => return ok,
                Err(e) => _err = _err.merge(e),
            }
        )+
        Err(_err)
    }};
    (local $ts:expr, [$($parser:expr),+ $(,)?]) => {{
        let ts = $ts;
        let mut _err = ParseError::default();
        let mut _result: Option<_> = None;
        $(
            if _result.is_none() {
                match $parser(ts.clone()) {
                    Ok(v) => _result = Some(Ok(v)),
                    Err(e) => _err = _err.merge(e),
                }
            }
        )+
        _result.unwrap_or(Err(_err))
    }};
}

fn indented_block<T>(
    ts: TokenStream,
    rule: &'static str,
    item: impl Fn(TokenStream) -> PResult<T>,
) -> PResult<(Vec<T>, Span)> {
    let (start, ts) = tok(TokenKind::Indent, rule)(ts)?;
    let (first, ts) = item(ts)?;
    let (rest, ts) = many0(ts, |ts| {
        let (_, ts) = tok(TokenKind::Eol, rule)(ts)?;
        item(ts)
    });
    let (_, ts) = opt(ts, tok(TokenKind::Eol, ""));
    let (end, ts) = tok(TokenKind::Dedent, rule)(ts)?;
    let mut items = vec![first];
    items.extend(rest);
    Ok(((items, span_from(&start, &end)), ts))
}

pub fn parse(ts: TokenStream) -> fb::Result<ast::File> {
    parse_file(ts)
        .map(|(file, _)| file)
        .map_err(|e| e.into_outbox())
}

fn parse_file(ts: TokenStream) -> PResult<ast::File> {
    let mut stmts = vec![];
    let mut ts = skip_eols(ts);
    while !ts.at_end() {
        let (stmt, new_ts) = parse_top_def(ts)?;
        stmts.push(stmt);
        ts = skip_eols(new_ts);
    }
    Ok((ast::File { stmts }, ts))
}

fn parse_top_def(ts: TokenStream) -> PResult<Stmt> {
    alt!(ts, [parse_type_sig, parse_uid_def, parse_lid_def])
}

fn parse_type_sig(ts: TokenStream) -> PResult<Stmt> {
    let start = ts.span();
    let ((name, _), ts) = expect_name("type_sig")(ts)?;
    let (_, ts) = tok(TokenKind::DblColon, "type_sig")(ts)?;
    let (sig, ts) = parse_type(ts)?;
    let span = span_from(&start, sig.span());
    Ok((Stmt::new_type_sig(name, sig, span), ts))
}

fn parse_lid_def(ts: TokenStream) -> PResult<Stmt> {
    let start = ts.span();
    let ((name, _), ts) = expect_lid("def")(ts)?;
    let (args, ts) = many0(ts, expect_name("param"));
    let (_, ts) = tok(TokenKind::Eq, "def")(ts)?;
    let (body, ts) = parse_expr(ts)?;
    let span = span_from(&start, body.span());
    Ok((Stmt::new_def(name, args, body, span), ts))
}

fn parse_uid_def(ts: TokenStream) -> PResult<Stmt> {
    enum Body {
        Enum(Vec<Variant>, Span),
        Struct(Expr, Span),
        Expr(Expr),
    }

    let start = ts.span();
    let ((name, _), ts) = expect_uid("def")(ts)?;
    let (args, ts) = many0(ts, expect_name("param"));
    let (_, ts) = tok(TokenKind::Eq, "def")(ts)?;

    let (body, ts) = alt!(local ts, [
        |ts| parse_enum_body(ts).map(|((v, s), ts)| (Body::Enum(v, s), ts)),
        |ts| parse_struct_body(ts).map(|((e, s), ts)| (Body::Struct(e, s), ts)),
        |ts| parse_expr(ts).map(|(e, ts)| (Body::Expr(e), ts)),
    ])?;

    let (body, end_span) = match body {
        Body::Enum(variants, end) => {
            let span = span_from(&start, &end);
            (Expr::new_ctor(Type::Enum { variants }, span.clone()), end)
        }
        Body::Struct(expr, end) => (expr, end),
        Body::Expr(expr) => {
            let s = expr.span().clone();
            (expr, s)
        }
    };
    let span = span_from(&start, &end_span);
    Ok((Stmt::new_def(name, args, body, span), ts))
}

fn parse_enum_body(ts: TokenStream) -> PResult<(Vec<Variant>, Span)> {
    let ts = skip_eols(ts);
    let (_, ts) = tok(TokenKind::Pipe, "enum")(ts)?;
    let ((first, first_span), ts) = parse_variant(ts)?;
    let (rest, ts) = many0(ts, |ts| {
        let ts = skip_eols(ts);
        let (_, ts) = tok(TokenKind::Pipe, "enum")(ts)?;
        parse_variant(ts)
    });
    let mut variants = vec![first];
    let mut last_span = first_span;
    for (v, s) in rest {
        variants.push(v);
        last_span = s;
    }
    Ok(((variants, last_span), ts))
}

fn parse_variant(ts: TokenStream) -> PResult<(Variant, Span)> {
    let ((name, name_span), ts) = expect_uid("variant")(ts)?;
    let (payload, ts) = opt(ts, parse_struct_body);
    match payload {
        Some((ty, end)) => Ok((((name, Some(ty)), span_from(&name_span, &end)), ts)),
        None => Ok((((name, None), name_span), ts)),
    }
}

fn parse_struct_body(ts: TokenStream) -> PResult<(Expr, Span)> {
    let ts = skip_eols(ts);
    let (start, ts) = tok(TokenKind::LCurly, "struct")(ts)?;
    let (_, ts) = opt(ts, tok(TokenKind::Comma, ""));
    let (fields, ts) = sep_by(ts, parse_field, tok(TokenKind::Comma, "struct"));
    let (end, ts) = tok(TokenKind::RCurly, "struct")(ts)?;
    let span = span_from(&start, &end);
    Ok((
        (Expr::new_ctor(Type::Record { fields }, span.clone()), span),
        ts,
    ))
}

fn parse_field(ts: TokenStream) -> PResult<(Symbol, Expr)> {
    let ((name, _), ts) = expect_lid("field")(ts)?;
    let (_, ts) = tok(TokenKind::Colon, "field")(ts)?;
    let (ty, ts) = parse_type(ts)?;
    Ok(((name, ty), ts))
}

fn parse_type(ts: TokenStream) -> PResult<Expr> {
    let (base, ts) = parse_type_base(ts)?;
    if let Ok((_, ts)) = tok(TokenKind::ThinRtArrow, "type")(ts.clone()) {
        let (ret, ts) = parse_type(ts)?;
        let span = span_from(base.span(), ret.span());
        return Ok((
            Expr::new_ctor(
                Type::Apply {
                    name: Symbol::from("->"),
                    args: vec![base, ret],
                },
                span,
            ),
            ts,
        ));
    }
    Ok((base, ts))
}

fn parse_type_base(ts: TokenStream) -> PResult<Expr> {
    alt!(ts, [
        |ts| parse_struct_body(ts).map(|((e, _), ts)| (e, ts)),
        parse_array_type,
        parse_named_type,
    ])
}

fn parse_named_type(ts: TokenStream) -> PResult<Expr> {
    let start = ts.span();
    let ((name, _), ts) = expect_name_or_builtin("type")(ts)?;
    let (args, ts) = many0(ts, parse_type_arg);
    let end = args.last().map(|a| a.span()).unwrap_or(&start);
    let span = span_from(&start, end);
    let ty = if args.is_empty() {
        Type::Name { name }
    } else {
        Type::Apply { name, args }
    };
    Ok((Expr::new_ctor(ty, span), ts))
}

fn parse_type_arg(ts: TokenStream) -> PResult<Expr> {
    alt!(ts, [parse_array_type, parse_atom])
}

fn parse_array_type(ts: TokenStream) -> PResult<Expr> {
    let (start, ts) = tok(TokenKind::LSqBrk, "array_type")(ts)?;
    let (dim, ts) = parse_expr(ts)?;
    let (_, ts) = tok(TokenKind::RSqBrk, "array_type")(ts)?;
    let (elem, ts) = parse_type_arg(ts)?;
    let span = span_from(&start, elem.span());
    Ok((
        Expr::new_ctor(
            Type::Apply {
                name: Symbol::from("[]"),
                args: vec![dim, elem],
            },
            span,
        ),
        ts,
    ))
}

fn parse_expr(ts: TokenStream) -> PResult<Expr> {
    alt!(ts, [parse_if, parse_match, parse_chain, parse_binop_expr])
}

fn parse_chain(ts: TokenStream) -> PResult<Expr> {
    let start = ts.span();
    let (_, ts) = tok(TokenKind::Eol, "chain")(ts)?;
    let ((stmts, end), ts) = indented_block(ts, "chain", parse_stmt)?;
    Ok((Expr::new_chain(stmts, span_from(&start, &end)), ts))
}

fn parse_stmt(ts: TokenStream) -> PResult<Stmt> {
    alt!(ts, [parse_let_stmt, parse_discard_stmt])
}

fn parse_let_stmt(ts: TokenStream) -> PResult<Stmt> {
    let start = ts.span();
    let ((name, name_span), ts) = expect_lid("let")(ts)?;
    let (_, ts) = tok(TokenKind::Eq, "let")(ts)?;
    let (init, ts) = parse_expr(ts)?;
    let span = span_from(&start, init.span());
    Ok((
        Stmt::new_let(Pattern::new_name(name, name_span), init, span),
        ts,
    ))
}

fn parse_discard_stmt(ts: TokenStream) -> PResult<Stmt> {
    let start = ts.span();
    let (val, ts) = parse_expr(ts)?;
    let span = span_from(&start, val.span());
    Ok((Stmt::new_discard(val, span), ts))
}

fn parse_if(ts: TokenStream) -> PResult<Expr> {
    let (start, ts) = tok(TokenKind::KwIf, "if")(ts)?;
    let (cond, ts) = parse_binop_expr(ts)?;
    let (_, ts) = tok(TokenKind::KwThen, "if")(ts)?;
    let (then_branch, ts) = parse_expr(ts)?;
    let (elifs, ts) = many0(ts, |ts| {
        let ts = skip_eols(ts);
        let (_, ts) = tok(TokenKind::KwElif, "if")(ts)?;
        let (cond, ts) = parse_binop_expr(ts)?;
        let (_, ts) = tok(TokenKind::KwThen, "if")(ts)?;
        let (body, ts) = parse_expr(ts)?;
        Ok(((cond, body), ts))
    });
    let (else_branch, ts) = opt(ts, |ts| {
        let ts = skip_eols(ts);
        let (_, ts) = tok(TokenKind::KwElse, "if")(ts)?;
        parse_expr(ts)
    });
    let mut cond_branches = vec![(cond, then_branch)];
    cond_branches.extend(elifs);
    let last_span = match &else_branch {
        Some(e) => e.span().clone(),
        None => cond_branches.last().unwrap().1.span().clone(),
    };
    Ok((
        Expr::new_if(cond_branches, else_branch, span_from(&start, &last_span)),
        ts,
    ))
}

fn parse_match(ts: TokenStream) -> PResult<Expr> {
    let (start, ts) = tok(TokenKind::KwMatch, "match")(ts)?;
    let (scrutinee, ts) = parse_atom(ts)?;
    let (_, ts) = tok(TokenKind::Eol, "match")(ts)?;
    let ((arms, end), ts) = indented_block(ts, "match", |ts| {
        let (pat, ts) = parse_pattern(ts)?;
        let (_, ts) = tok(TokenKind::ThickRtArrow, "match")(ts)?;
        let (body, ts) = parse_expr(ts)?;
        Ok(((pat, body), ts))
    })?;
    Ok((
        Expr::new_match(scrutinee, arms, span_from(&start, &end)),
        ts,
    ))
}

fn parse_pattern(ts: TokenStream) -> PResult<Pattern> {
    alt!(ts, [
        parse_pattern_constructor,
        |ts| {
            let ((s, sp), ts) = expect_lid("pattern")(ts)?;
            Ok((Pattern::new_name(s, sp), ts))
        },
        |ts| {
            let ((_, sp), ts) = expect_hole("pattern")(ts)?;
            Ok((Pattern::new_hole(sp), ts))
        },
        |ts| {
            let ((v, sp), ts) = expect_literal("pattern")(ts)?;
            Ok((Pattern::new_literal(v, sp), ts))
        },
    ])
}

fn parse_pattern_constructor(ts: TokenStream) -> PResult<Pattern> {
    let ((sym, span), ts) = expect_uid("pattern")(ts)?;
    let (arg, ts) = opt(ts, parse_pattern);
    let full_span = match &arg {
        Some(p) => span_from(&span, p.span()),
        None => span.clone(),
    };
    Ok((Pattern::new_constructor(sym, arg, full_span), ts))
}

fn parse_binop_expr(ts: TokenStream) -> PResult<Expr> {
    parse_prec(ts, 0)
}

fn parse_prec(ts: TokenStream, min_prec: u8) -> PResult<Expr> {
    let (mut lhs, ts) = if min_prec >= 5 {
        parse_apply_expr(ts)?
    } else {
        parse_prec(ts, min_prec + 1)?
    };
    let binop_tok = expect_if(
        move |t| {
            let (prec, kind) = binop_info(&t.kind)?;
            if prec < min_prec {
                return None;
            }
            Some((prec, kind, t.span.clone()))
        },
        "binop",
        "operator",
    );
    let (ops, ts) = many0(ts, |ts| {
        let ((prec, kind, op_span), ts) = binop_tok(ts)?;
        let (rhs, ts) = parse_prec(ts, prec + 1)?;
        Ok(((kind, op_span, rhs), ts))
    });
    for (kind, op_span, rhs) in ops {
        let span = span_from(lhs.span(), rhs.span());
        lhs = Expr::new_apply(
            Expr::new_name(kind.to_symbol(), op_span),
            vec![lhs, rhs],
            span,
        );
    }
    Ok((lhs, ts))
}

#[derive(Clone, Copy)]
enum BinOpKind {
    LogOr,
    LogAnd,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    IntDiv,
    Rem,
}

impl BinOpKind {
    fn to_symbol(self) -> Symbol {
        Symbol::from(match self {
            Self::LogOr => "||(_,_)",
            Self::LogAnd => "&&(_,_)",
            Self::Eq => "==(_,_)",
            Self::Ne => "!=(_,_)",
            Self::Lt => "<(_,_)",
            Self::Gt => ">(_,_)",
            Self::Le => "<=(_,_)",
            Self::Ge => ">=(_,_)",
            Self::Add => "+(_,_)",
            Self::Sub => "-(_,_)",
            Self::Mul => "*(_,_)",
            Self::Div => "/(_,_)",
            Self::IntDiv => "//(_,_)",
            Self::Rem => "%(_,_)",
        })
    }
}

fn binop_info(tk: &TokenKind) -> Option<(u8, BinOpKind)> {
    Some(match tk {
        TokenKind::DblPipe => (0, BinOpKind::LogOr),
        TokenKind::DblAmpersand => (1, BinOpKind::LogAnd),
        TokenKind::DblEq => (2, BinOpKind::Eq),
        TokenKind::NotEq => (2, BinOpKind::Ne),
        TokenKind::LessThan => (2, BinOpKind::Lt),
        TokenKind::GreaterThan => (2, BinOpKind::Gt),
        TokenKind::LessThanEq => (2, BinOpKind::Le),
        TokenKind::GreaterThanEq => (2, BinOpKind::Ge),
        TokenKind::Plus => (3, BinOpKind::Add),
        TokenKind::Minus => (3, BinOpKind::Sub),
        TokenKind::Asterisk => (4, BinOpKind::Mul),
        TokenKind::FSlash => (4, BinOpKind::Div),
        TokenKind::DblFSlash => (4, BinOpKind::IntDiv),
        TokenKind::Percent => (4, BinOpKind::Rem),
        _ => return None,
    })
}

fn parse_apply_expr(ts: TokenStream) -> PResult<Expr> {
    let (head, ts) = parse_postfix_expr(ts)?;
    let (args, ts) = many0(ts, parse_postfix_expr);
    if args.is_empty() {
        Ok((head, ts))
    } else {
        let span = span_from(head.span(), args.last().unwrap().span());
        Ok((Expr::new_apply(head, args, span), ts))
    }
}

fn parse_postfix_expr(ts: TokenStream) -> PResult<Expr> {
    let (mut expr, ts) = parse_atom(ts)?;
    let (dots, ts) = many0(ts, |ts| {
        let (_, ts) = tok(TokenKind::Dot, "dot")(ts)?;
        expect_lid("dot_access")(ts)
    });
    for (field_name, field_span) in dots {
        let span = span_from(expr.span(), &field_span);
        expr = Expr::new_dot(expr, field_name, span);
    }
    Ok((expr, ts))
}

fn parse_atom(ts: TokenStream) -> PResult<Expr> {
    alt!(ts, [
        |ts| {
            let ((v, s), ts) = expect_literal("expr")(ts)?;
            Ok((Expr::new_literal(v, s), ts))
        },
        |ts| {
            let ((s, sp), ts) = expect_name_or_builtin("expr")(ts)?;
            Ok((Expr::new_name(s, sp), ts))
        },
        parse_paren,
        parse_array_type,
    ])
}

fn parse_paren(ts: TokenStream) -> PResult<Expr> {
    let (start, ts) = tok(TokenKind::LParen, "paren")(ts)?;
    let (first, ts) = parse_expr(ts)?;
    let (rest, ts) = many0(ts, |ts| {
        let (_, ts) = tok(TokenKind::Comma, "paren")(ts)?;
        parse_expr(ts)
    });
    let (end, ts) = tok(TokenKind::RParen, "paren")(ts)?;
    if rest.is_empty() {
        Ok((first, ts))
    } else {
        let mut elements = vec![first];
        elements.extend(rest);
        Ok((Expr::new_tuple(elements, span_from(&start, &end)), ts))
    }
}

fn span_from(start: &Span, end: &Span) -> Span {
    Span::new(
        &start.source,
        start.beg_offset as usize,
        end.end_offset as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Lexer, Source};

    fn parse_source(src: &str) -> ast::File {
        let lexer = Lexer::new(Config::default());
        let source = Source::new("test", src, lexer.config());
        let tokens = lexer
            .lex(source)
            .unwrap_or_else(|e| panic!("Lex error: {e:?}"));
        let (file, _) = parse_file(TokenStream::new(tokens))
            .unwrap_or_else(|e| panic!("Parse error: {e:?}"));
        file
    }

    #[test]
    fn test_simple_def() {
        let f = parse_source("relu x =\n  max 0 x\n");
        assert!(
            matches!(&f.stmts[0], Stmt::Def(d) if d.name.text() == "relu" && d.args.len() == 1)
        );
    }
    #[test]
    fn test_multi_arg_def() {
        let f = parse_source("linear model input =\n  model\n");
        assert!(
            matches!(&f.stmts[0], Stmt::Def(d) if d.name.text() == "linear" && d.args.len() == 2)
        );
    }
    #[test]
    fn test_struct_curly() {
        let f = parse_source("Linear T o i =\n{ w: T\n, b: T }\n");
        match &f.stmts[0] {
            Stmt::Def(d) => {
                assert_eq!(d.name.text(), "Linear");
                assert_eq!(d.args.len(), 3);
                assert!(
                    matches!(&d.body, Expr::Ctor(c) if matches!(&c.ty, Type::Record { fields } if fields.len() == 2))
                );
            }
            other => panic!("Expected Def, got {other:?}"),
        }
    }
    #[test]
    fn test_type_sig() {
        let f = parse_source("linear :: T -> T\n");
        assert!(matches!(&f.stmts[0], Stmt::TypeSig(s) if s.name.text() == "linear"));
    }
    #[test]
    fn test_enum_def() {
        let f = parse_source("Option T =\n| Some\n| None\n");
        match &f.stmts[0] {
            Stmt::Def(d) => {
                assert_eq!(d.name.text(), "Option");
                assert_eq!(d.args.len(), 1);
                match &d.body {
                    Expr::Ctor(c) => match &c.ty {
                        Type::Enum { variants } => {
                            assert_eq!(variants.len(), 2);
                            assert_eq!(variants[0].0.text(), "Some");
                            assert_eq!(variants[1].0.text(), "None");
                        }
                        other => panic!("Expected Enum type, got {other:?}"),
                    },
                    other => panic!("Expected Ctor, got {other:?}"),
                }
            }
            other => panic!("Expected Def, got {other:?}"),
        }
    }
    #[test]
    fn test_if_else() {
        parse_source("f x =\n  if x then\n    1\n  else\n    0\n");
    }
    #[test]
    fn test_let_chain() {
        let f = parse_source("f x =\n  y = x\n  y\n");
        assert!(matches!(&f.stmts[0], Stmt::Def(d) if matches!(&d.body, Expr::Chain(_))));
    }
    #[test]
    fn test_binop() {
        parse_source("f x =\n  x + 1\n");
    }
    #[test]
    fn test_dot_access() {
        parse_source("f x =\n  x.y\n");
    }
    #[test]
    fn test_paren_expr() {
        parse_source("f x =\n  g (x + 1)\n");
    }
    #[test]
    fn test_multiple_defs() {
        assert_eq!(
            parse_source("f x =\n  x\n\ng y =\n  y\n").stmts.len(),
            2
        );
    }
    #[test]
    fn test_array_type() {
        let f = parse_source("Foo T n =\n{ x: [n]T }\n");
        assert!(
            matches!(&f.stmts[0], Stmt::Def(d) if matches!(&d.body, Expr::Ctor(c) if matches!(&c.ty, Type::Record { .. })))
        );
    }
    #[test]
    fn test_builtin_type() {
        let f = parse_source("Wrapper =\n{ x: u32 }\n");
        match &f.stmts[0] {
            Stmt::Def(d) => match &d.body {
                Expr::Ctor(c) => match &c.ty {
                    Type::Record { fields } => match &fields[0].1 {
                        Expr::Ctor(ft) => {
                            assert!(matches!(&ft.ty, Type::Name { name } if name.text() == "u32"))
                        }
                        other => panic!("Expected Ctor, got {other:?}"),
                    },
                    other => panic!("Expected Record, got {other:?}"),
                },
                other => panic!("Expected Ctor, got {other:?}"),
            },
            other => panic!("Expected Def, got {other:?}"),
        }
    }
}
