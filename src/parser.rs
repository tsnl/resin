use lalrpop_util::lalrpop_mod;
use num::{BigInt, BigRational, FromPrimitive, Num};

use crate::{Source, ast, feedback as fb};

pub fn parse_file(source: &Source) -> fb::Result<Vec<ast::Def>> {
    parser::FileParser::new()
        .parse(source, source.text())
        .map_err(|err| match err {
            lalrpop_util::ParseError::InvalidToken { location } => {
                let (line, column) = source.line_column(location as u32);
                todo!(
                    "Report parsing error: {err:?}: {}:{line}:{column}",
                    source.name()
                )
            }
            _ => todo!("Report parsing error: {err:?}"),
        })
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
    use crate::{Literal, Source, Span, Symbol, ast};

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
}
