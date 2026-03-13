use lalrpop_util::{ParseError, lalrpop_mod};

use crate::{
    Source, Span, Symbol, ast, fb,
    lexer::{Token, TokenStream},
    vocab::{self, BuiltinOperatorNames, BuiltinType},
};

pub fn parse_file(
    builtin_op_names: &BuiltinOperatorNames,
    token_stream: TokenStream,
) -> fb::Result<ast::File> {
    let source = token_stream.source().clone();
    match parser::FileParser::new().parse(builtin_op_names, &source, token_stream) {
        Ok(file) => Ok(file),
        Err(err) => match err {
            ParseError::InvalidToken { location } => {
                syntax_error("Invalid token", None, loc(&source, location, location))
            }
            ParseError::UnrecognizedEof { location, expected } => syntax_error(
                "Unexpected end of file",
                Some(expected),
                loc(&source, location, location),
            ),
            ParseError::UnrecognizedToken {
                token: (c0, token, c1),
                expected,
            } => syntax_error(
                format!("Unrecognized token: {:?}", token),
                Some(expected),
                loc(&source, c0, c1),
            ),
            ParseError::ExtraToken {
                token: (c0, token, c1),
            } => syntax_error(format!("Extra token: {:?}", token), None, loc(&source, c0, c1)),
            ParseError::User { error } => Err(error),
        },
    }
}

fn syntax_error<T>(title: impl Into<String>, expected: Option<Vec<String>>, loc: fb::Loc) -> fb::Result<T> {
    Err(fb::Error::SyntaxError(fb::SyntaxError {
        title: title.into(),
        expected,
        loc,
    }))
}

pub fn loc(source: &Source, l: usize, r: usize) -> fb::Loc {
    fb::Loc::File(Span::new(source, l, r))
}

impl Iterator for TokenStream {
    type Item = fb::Result<(usize, Token, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Err(report) => Some(Err(report)),
            Ok(None) => None,
            Ok(Some((token, span))) => {
                let start = span.beg_offset as usize;
                let end = span.end_offset as usize;
                Some(Ok((start, token, end)))
            }
        }
    }
}

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
	#[rustfmt::skip]
	parser
);
