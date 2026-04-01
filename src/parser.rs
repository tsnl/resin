use lalrpop_util::{ParseError, lalrpop_mod, lexer::Token};
use num::{BigInt, BigRational, FromPrimitive, Num};

use crate::{Source, Span, ast, feedback as fb};

pub fn parse_file(source: &Source) -> fb::Result<Vec<ast::Def>> {
    parser::FileParser::new()
        .parse(source, source.text())
        .map_err(|err| parse_error(source, err))
}

fn parse_error(source: &Source, err: ParseError<usize, Token<'_>, fb::Error>) -> fb::Error {
    match err {
        ParseError::InvalidToken { location } => fb::Error::new_at(
            fb::ErrorKind::Syntax,
            "invalid token",
            Span::new(source, location, location),
        ),
        ParseError::UnrecognizedEof { location, expected } => {
            let expected = format_expected_tokens(&expected);
            let message = if expected.is_empty() {
                String::from("unexpected end of file")
            } else {
                format!("unexpected end of file; expected {expected}")
            };
            fb::Error::new_at(
                fb::ErrorKind::Syntax,
                message,
                Span::new(source, location, location),
            )
        }
        ParseError::UnrecognizedToken {
            token: (ll, token, rl),
            expected,
        } => {
            let expected = format_expected_tokens(&expected);
            let message = if expected.is_empty() {
                format!("unexpected token {}", format_token_text(token.1))
            } else {
                format!(
                    "unexpected token {}; expected {expected}",
                    format_token_text(token.1)
                )
            };
            fb::Error::new_at(fb::ErrorKind::Syntax, message, Span::new(source, ll, rl))
        }
        ParseError::ExtraToken {
            token: (ll, token, rl),
        } => fb::Error::new_at(
            fb::ErrorKind::Syntax,
            format!("unexpected extra token {}", format_token_text(token.1)),
            Span::new(source, ll, rl),
        ),
        ParseError::User { error } => error,
    }
}

fn format_expected_tokens(expected: &[String]) -> String {
    let mut expected = expected
        .iter()
        .map(|expected| describe_expected_token(expected))
        .collect::<Vec<_>>();
    expected.sort();
    expected.dedup();

    match expected.len() {
        0 => String::new(),
        1 => expected.pop().unwrap(),
        _ => {
            let last = expected.pop().unwrap();
            format!("{} or {last}", expected.join(", "))
        }
    }
}

fn describe_expected_token(expected: &str) -> String {
    if expected.starts_with('"') && expected.ends_with('"') {
        format_token_text(&expected[1..expected.len() - 1])
    } else {
        let regex = expected
            .strip_prefix("r#\"")
            .and_then(|expected| expected.strip_suffix("\"#"))
            .unwrap_or(expected);
        if regex.contains("[a-zA-Z_][a-zA-Z0-9_]*") {
            String::from("identifier")
        } else if regex.starts_with("[0-9]+(") {
            String::from("number")
        } else if regex.contains("([^") && regex.contains("\\\\.") {
            String::from("string literal")
        } else {
            expected.to_string()
        }
    }
}

fn format_token_text(text: &str) -> String {
    format!("`{}`", text.escape_default())
}

fn parse_string_literal(source: &Source, raw: &str, ll: usize, rl: usize) -> fb::Result<String> {
    let mut unescaped = String::new();
    let body = &raw[1..raw.len() - 1];
    let string_span = Span::new(source, ll, rl);

    let mut chars = body.char_indices();
    while let Some((index, c)) = chars.next() {
        if c != '\\' {
            unescaped.push(c);
            continue;
        }

        let Some((_, escaped)) = chars.next() else {
            return Err(fb::Error::new_at(
                fb::ErrorKind::Internal,
                "unterminated escape sequence in string literal",
                string_span,
            ));
        };

        match escaped {
            'b' => unescaped.push('\u{0008}'),
            'f' => unescaped.push('\u{000C}'),
            'n' => unescaped.push('\n'),
            'r' => unescaped.push('\r'),
            't' => unescaped.push('\t'),
            '\'' => unescaped.push('\''),
            '"' => unescaped.push('"'),
            '\\' => unescaped.push('\\'),
            _ => {
                let escape_start = ll + 1 + index;
                let escape_end = escape_start + 1 + escaped.len_utf8();
                let escape_span = Span::new(source, escape_start, escape_end);
                return Err(fb::Error::new_at(
                    fb::ErrorKind::Syntax,
                    format!("invalid escape sequence `\\{escaped}` in string literal"),
                    escape_span,
                )
                .with_note("in this string literal", string_span));
            }
        }
    }

    Ok(unescaped)
}

fn parse_number_literal(raw: &str, radix: u32) -> (BigRational, bool) {
    let raw = raw.trim_start_matches("0x");

    enum State {
        S1, // before '.'
        S2, // after '.', before 'e' or 'E'
        S3, // after 'e' or 'E'
    }

    // Parsing as three parts: S1, S2, S3:
    //  |1234.4567e8901|
    //  |-S1-|-S2-|-S3-|
    //  The transitions between these states are:
    //  {S1     }  ->  S2: '.'
    //  {S1 | S2}  ->  S3: 'e' or 'E'
    // Note that we do not need to validate the input string as the lexer regex pattern already
    // ensures that the input string is valid.
    // Note that we recognize scientific notation as integers if there is no '.' in the literal.
    let mut state = State::S1;
    let mut i1_s = String::default();
    let mut i2_s = String::default();
    let mut i3_s = String::default();
    let mut force_float = false;
    for c in raw.chars() {
        if c.is_ascii_digit() || c == '+' || c == '-' {
            match state {
                State::S1 => i1_s.push(c),
                State::S2 => i2_s.push(c),
                State::S3 => i3_s.push(c),
            }
        } else {
            match (state, c) {
                (State::S1, '.') => {
                    force_float = true;
                    state = State::S2;
                }
                (State::S1 | State::S2, 'e' | 'E') => {
                    state = State::S3;
                }
                _ => {
                    panic!("Invalid character in number literal");
                }
            }
        }
    }
    debug_assert!(!i1_s.is_empty());

    let n1: BigRational = BigInt::from_str_radix(&i1_s, radix).unwrap().into();
    let n2: BigRational = match i2_s {
        s if s.is_empty() => BigRational::from_i32(0).unwrap(),
        s => {
            let num: BigRational = BigInt::from_str_radix(&s, radix).unwrap().into();
            let den = BigRational::from_i32(10)
                .unwrap()
                .pow(s.len().try_into().unwrap());
            num / den
        }
    };
    let exp = match i3_s {
        s if s.is_empty() => 0,
        s => i32::from_str_radix(&s, radix).unwrap(),
    };

    let value = (n1 + n2) * BigRational::from_i32(10).unwrap().pow(exp);
    (value, force_float)
}

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    parser
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Literal, Source, Span, Symbol, ast, feedback::ErrorKind};

    #[test]
    fn test_parse_val() {
        let source = Source::new("test_parse_val", "a: ()", &Default::default());
        let ast = parser::ValParser::new()
            .parse(&source, source.text())
            .expect("Failed to parse val");
        assert_eq!(ast.name(), &Symbol::from("a"));
        assert_eq!(
            ast.expr(),
            &ast::Expr::new_literal(Literal::Unit, Span::new(&source, 3, 5))
        );
    }

    #[test]
    fn test_parse_file_reports_syntax_error() {
        let source = Source::new(
            "test_parse_file_reports_syntax_error",
            "def a() = (;",
            &Default::default(),
        );
        let error = parse_file(&source).expect_err("Expected parsing to fail");
        assert_eq!(error.kind, ErrorKind::Syntax);
        assert_eq!(
            error.to_string(),
            "syntax error: unexpected token `;`; expected `!`, `(`, `)`, `+`, `-`, `.`, `[`, `false`, `if`, `true`, `{`, identifier, number or string literal at test_parse_file_reports_syntax_error:1:12-13"
        );
    }

    #[test]
    fn test_parse_file_reports_invalid_string_escape() {
        let source = Source::new(
            "test_parse_file_reports_invalid_string_escape",
            r#"def a() = "\q";"#,
            &Default::default(),
        );
        let error = parse_file(&source).expect_err("Expected parsing to fail");
        assert_eq!(error.kind, ErrorKind::Syntax);
        assert_eq!(
            error.to_string(),
            "syntax error: invalid escape sequence `\\q` in string literal at test_parse_file_reports_invalid_string_escape:1:12-14\n  note: in this string literal at test_parse_file_reports_invalid_string_escape:1:11-15"
        );
    }
}
