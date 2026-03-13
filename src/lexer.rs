use hashbrown::HashMap;
use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::Arc;

use crate::vocab::{LiteralNumber, LiteralString};
use crate::{Config, Source, Span, Symbol, fb};

//
// Lexer
//

pub struct Lexer {
    config: Config,
    keyword_map: HashMap<Symbol, Token>,
}
impl Lexer {
    pub fn new(config: Config) -> Arc<Self> {
        let keyword_map = HashMap::from([
            (Symbol::from("def"), Token::KwDef),
            (Symbol::from("let"), Token::KwLet),
            (Symbol::from("if"), Token::KwIf),
            (Symbol::from("then"), Token::KwThen),
            (Symbol::from("elif"), Token::KwElif),
            (Symbol::from("else"), Token::KwElse),
            (Symbol::from("match"), Token::KwMatch),
            (Symbol::from("u8"), Token::KwU8),
            (Symbol::from("u16"), Token::KwU16),
            (Symbol::from("u32"), Token::KwU32),
            (Symbol::from("u64"), Token::KwU64),
            (Symbol::from("i8"), Token::KwI8),
            (Symbol::from("i16"), Token::KwI16),
            (Symbol::from("i32"), Token::KwI32),
            (Symbol::from("i64"), Token::KwI64),
            (Symbol::from("f32"), Token::KwF32),
            (Symbol::from("f64"), Token::KwF64),
            (Symbol::from("bool"), Token::KwBool),
            (Symbol::from("string"), Token::KwString),
            (Symbol::from("symbol"), Token::KwSymbol),
            (Symbol::from("unit"), Token::KwUnit),
            (Symbol::from("Fn"), Token::KwFnUid),
            (Symbol::from("true"), Token::KwTrue),
            (Symbol::from("false"), Token::KwFalse),
        ]);
        Arc::new(Self {
            config,
            keyword_map,
        })
    }
    pub fn lex(self: &Arc<Self>, source: Source) -> TokenStream {
        TokenStream::new(source, self.clone())
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    fn keyword_map(&self) -> &HashMap<Symbol, Token> {
        &self.keyword_map
    }
}

pub struct TokenStream {
    input_stream: ByteStream,
    indent_stack: Vec<usize>,
    bookend_stack: Vec<Bookend>, // (), [], {}
    lookahead_buffer: VecDeque<(Token, Span)>,
    parent_lexer: Arc<Lexer>,
}
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bookend {
    Paren,
    SqBrk,
    Curly,
}
impl TokenStream {
    const INDENT_CAPACITY: usize = 16;
    const BOOKEND_CAPACITY: usize = 32;

    pub fn new(source: Source, manager: Arc<Lexer>) -> Self {
        Self {
            input_stream: ByteStream::new(source),
            indent_stack: {
                let mut indent_stack = Vec::with_capacity(Self::INDENT_CAPACITY);
                indent_stack.push(0);
                indent_stack
            },
            bookend_stack: Vec::with_capacity(Self::BOOKEND_CAPACITY),
            lookahead_buffer: VecDeque::with_capacity(Self::INDENT_CAPACITY),
            parent_lexer: manager,
        }
    }
    pub fn into_token_vec(mut self) -> fb::Result<Vec<(Token, Span)>> {
        let mut tokens = vec![];
        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }
    pub fn next_token(&mut self) -> fb::Result<Option<(Token, Span)>> {
        self.try_growing_lookahead_buffer_to_n(1)?;
        Ok(self.lookahead_buffer.pop_front())
    }
    pub fn peek_loc(&mut self) -> fb::Result<fb::Loc> {
        self.peek_loc_at(0)
    }
    pub fn peek_loc_at(&mut self, i: usize) -> fb::Result<fb::Loc> {
        Ok(self
            .peek_at(i)?
            .map(|(_, span)| fb::Loc::File(span.clone()))
            .unwrap_or_else(|| {
                let offset = self.cursor();
                fb::Loc::File(Span::new(self.input_stream.source(), offset, offset))
            }))
    }
    pub fn peek(&mut self) -> fb::Result<Option<(&Token, &Span)>> {
        self.peek_at(0)
    }
    pub fn peek_at(&mut self, i: usize) -> fb::Result<Option<(&Token, &Span)>> {
        self.try_growing_lookahead_buffer_to_n(i + 1)?;
        Ok(self
            .lookahead_buffer
            .get(i)
            .map(|(token, span)| (token, span)))
    }
    fn try_growing_lookahead_buffer_to_n(&mut self, n: usize) -> fb::Result<()> {
        while self.lookahead_buffer.len() < n {
            let initial_len = self.lookahead_buffer.len();
            self.replenish_lookahead_buffer_once()?;
            let new_len = self.lookahead_buffer.len();
            if new_len == initial_len {
                // Fixed point at EOF => terminate growth
                break;
            }
        }
        Ok(())
    }
    fn replenish_lookahead_buffer_once(&mut self) -> fb::Result<()> {
        // Try skipping whitespace and comments:
        self.skip();

        // If at EOF, early out:
        if self.input_stream.peek() == 0 {
            return Ok(());
        }

        // Try replenishing each token type:
        if self.try_lex_punct()? {
            return Ok(());
        }
        if self.try_lex_identifier()? {
            return Ok(());
        }
        if self.try_lex_number()? {
            return Ok(());
        }
        if self.try_lex_string_literal()? {
            return Ok(());
        }
        if self.bookend_stack.is_empty() && self.try_lex_eol()? {
            return Ok(());
        }

        // If all replenishment attempts failed, error:
        syntax_err(
            format!(
                "Unexpected byte in source file: {:#?} (0x{:02x})",
                str::from_utf8(&[self.input_stream.peek()]).unwrap_or("(?)"),
                self.input_stream.peek()
            ),
            self.loc(self.cursor()),
        )
    }
    fn skip(&mut self) {
        loop {
            if self.try_skip_whitespace() {
                continue;
            }
            if self.try_skip_line_comment() {
                continue;
            }
            break;
        }
    }
    fn try_skip_whitespace(&mut self) -> bool {
        self.input_stream.match_byte_if(|b| {
            if b.is_ascii_whitespace() && b != b'\n' {
                return true;
            }
            if !self.bookend_stack.is_empty() && b == b'\n' {
                return true;
            }
            false
        }) != 0
    }
    fn try_skip_line_comment(&mut self) -> bool {
        if self.input_stream.match_bstr(b"#") {
            while self.input_stream.match_byte_if(|b| b != b'\n') != 0 {}
            true
        } else {
            false
        }
    }
    fn try_lex_identifier(&mut self) -> fb::Result<bool> {
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<(Token, Span)>> {
            let lt_cursor = this.cursor();

            // Scan the identifier's characters:
            let b0 = this
                .input_stream
                .match_byte_if(|b| b.is_ascii_alphabetic() || b == b'_');
            if b0 == 0 {
                return Ok(None);
            }
            let mut bs = vec![b0];
            loop {
                let b1 = this
                    .input_stream
                    .match_byte_if(|b| b.is_ascii_alphanumeric() || b == b'_');
                if b1 == 0 {
                    break;
                }
                bs.push(b1);
            }

            // Intern the string first.
            let name_symbol = Symbol::from(String::from_utf8(bs.clone()).unwrap());

            // Check if the identifier is a keyword.
            if let Some(keyword) = this.parent_lexer().keyword_map().get(&name_symbol) {
                return Ok(Some((keyword.clone(), this.span(lt_cursor))));
            }

            // For identifier: see if the first character is lowercase or uppercase.
            let mut opt_is_lid_not_uid: Option<bool> = None;
            for b in bs.iter() {
                if b.is_ascii_lowercase() {
                    opt_is_lid_not_uid = Some(true);
                    break;
                }
                if b.is_ascii_uppercase() {
                    opt_is_lid_not_uid = Some(false);
                    break;
                }
            }

            match opt_is_lid_not_uid {
                Some(is_lid_not_uid) => {
                    if is_lid_not_uid {
                        Ok(Some((Token::Lid(name_symbol), this.span(lt_cursor))))
                    } else {
                        Ok(Some((Token::Uid(name_symbol), this.span(lt_cursor))))
                    }
                }
                None => Ok(Some((Token::Hole(name_symbol), this.span(lt_cursor)))),
            }
        })
    }
    fn try_lex_punct(&mut self) -> fb::Result<bool> {
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<(Token, Span)>> {
            let lt_cursor = this.cursor();
            macro_rules! ok_spanned_token {
                ($token:expr) => {
                    Ok(Some(($token, this.span(lt_cursor))))
                };
            }
            macro_rules! push_bookend {
                ($bookend:expr) => {
                    this.bookend_stack.push($bookend);
                };
            }
            macro_rules! pop_bookend_else_return_error {
                ($bookend:expr) => {
                    match this.bookend_stack.pop() {
                        None => {
                            return syntax_err(
                                format!("Unmatched closing bookend: see {:?}.", $bookend),
                                this.loc(lt_cursor),
                            );
                        }
                        Some(b) if b == $bookend => {
                            // OK
                        }
                        Some(b) => {
                            return syntax_err(
                                format!(
                                    "Unmatched closing bookend: expected {:?}, but matched {:?}.",
                                    $bookend, b,
                                ),
                                this.loc(lt_cursor),
                            );
                        }
                    }
                };
            }
            if this.input_stream.match_byte(b'(') {
                push_bookend!(Bookend::Paren);
                return ok_spanned_token!(Token::LParen);
            }
            if this.input_stream.match_byte(b')') {
                pop_bookend_else_return_error!(Bookend::Paren);
                return ok_spanned_token!(Token::RParen);
            }
            if this.input_stream.match_byte(b'[') {
                push_bookend!(Bookend::SqBrk);
                return ok_spanned_token!(Token::LSqBrk);
            }
            if this.input_stream.match_byte(b']') {
                pop_bookend_else_return_error!(Bookend::SqBrk);
                return ok_spanned_token!(Token::RSqBrk);
            }
            if this.input_stream.match_byte(b'{') {
                push_bookend!(Bookend::Curly);
                return ok_spanned_token!(Token::LCurly);
            }
            if this.input_stream.match_byte(b'}') {
                pop_bookend_else_return_error!(Bookend::Curly);
                return ok_spanned_token!(Token::RCurly);
            }
            if this.input_stream.match_byte(b'.') {
                return ok_spanned_token!(Token::Dot);
            }
            if this.input_stream.match_byte(b',') {
                return ok_spanned_token!(Token::Comma);
            }
            if this.input_stream.match_byte(b'\'') {
                return ok_spanned_token!(Token::Quote);
            }
            if this.input_stream.match_byte(b':') {
                return ok_spanned_token!(Token::Colon);
            }
            if this.input_stream.match_byte(b';') {
                return ok_spanned_token!(Token::Semicolon);
            }
            if this.input_stream.match_byte(b'*') {
                return ok_spanned_token!(Token::Asterisk);
            }
            if this.input_stream.match_byte(b'/') {
                if this.input_stream.match_byte(b'/') {
                    return ok_spanned_token!(Token::DblFSlash);
                }
                return ok_spanned_token!(Token::FSlash);
            }
            if this.input_stream.match_byte(b'%') {
                return ok_spanned_token!(Token::Percent);
            }
            if this.input_stream.match_byte(b'+') {
                return ok_spanned_token!(Token::Plus);
            }
            if this.input_stream.match_byte(b'-') {
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(Token::ThinRtArrow);
                }
                return ok_spanned_token!(Token::Minus);
            }
            if this.input_stream.match_byte(b'<') {
                if this.input_stream.match_byte(b'<') {
                    return ok_spanned_token!(Token::LShift);
                }
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(Token::LessThanEq);
                }
                if this.input_stream.match_byte(b'-') {
                    return ok_spanned_token!(Token::ThinLtArrow);
                }
                return ok_spanned_token!(Token::LessThan);
            }
            if this.input_stream.match_byte(b'>') {
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(Token::RShift);
                }
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(Token::GreaterThanEq);
                }
                return ok_spanned_token!(Token::GreaterThan);
            }
            if this.input_stream.match_byte(b'=') {
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(Token::DblEq);
                }
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(Token::ThickRtArrow);
                }
                return ok_spanned_token!(Token::Eq);
            }
            if this.input_stream.match_byte(b'!') {
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(Token::NotEq);
                }
                return ok_spanned_token!(Token::Bang);
            }
            if this.input_stream.match_byte(b'|') {
                if this.input_stream.match_byte(b'|') {
                    return ok_spanned_token!(Token::DblPipe);
                }
                return ok_spanned_token!(Token::Pipe);
            }
            if this.input_stream.match_byte(b'&') {
                if this.input_stream.match_byte(b'&') {
                    return ok_spanned_token!(Token::DblAmpersand);
                }
                return ok_spanned_token!(Token::Ampersand);
            }
            if this.input_stream.match_byte(b'^') {
                return ok_spanned_token!(Token::Caret);
            }
            Ok(None)
        })
    }
    fn try_lex_number(&mut self) -> fb::Result<bool> {
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<(Token, Span)>> {
            let lt_cursor = this.cursor();

            // Scan the number's characters:
            let mut bs = vec![];
            let mut force_float = false;
            let mut has_digits_before_exponent = false;

            // First character must be a digit or a dot.
            let b0 = this
                .input_stream
                .match_byte_if(|b| b.is_ascii_digit() || b == b'.');
            if b0 == 0 {
                return Ok(None);
            }
            bs.push(b0);
            if b0 == b'.' {
                force_float = true;
            } else {
                has_digits_before_exponent = true;
            }

            // Continue scanning digits or dots.
            loop {
                let b1 = this
                    .input_stream
                    .match_byte_if(|b| b.is_ascii_digit() || b == b'.');
                if b1 == 0 {
                    break;
                }
                if b1 == b'.' {
                    if force_float {
                        return syntax_err(
                            r"Unexpected '.' in number literal.",
                            this.loc(lt_cursor),
                        );
                    }
                    force_float = true;
                } else {
                    has_digits_before_exponent = true;
                }
                bs.push(b1);
            }

            // Ensure at least one digit was seen so far.
            if !has_digits_before_exponent {
                return syntax_err(
                    r"Expected at least one digit in number literal.",
                    this.loc(lt_cursor),
                );
            }

            // Scan an optional exponent part.
            if this.input_stream.match_byte(b'e') || this.input_stream.match_byte(b'E') {
                bs.push(b'e');

                let sign = this.input_stream.match_byte_if(|b| b == b'+' || b == b'-');
                if sign != 0 {
                    bs.push(sign);
                }

                let mut exponent_has_digits = false;
                loop {
                    let b3 = this.input_stream.match_byte_if(|b| b.is_ascii_digit());
                    if b3 == 0 {
                        break;
                    }
                    exponent_has_digits = true;
                    bs.push(b3);
                }
                if !exponent_has_digits {
                    return syntax_err(
                        r"Expected at least one digit in exponent of number literal.",
                        this.loc(lt_cursor),
                    );
                }

                if sign == b'-' {
                    force_float = true;
                }
            }

            let content = String::from_utf8(bs).unwrap();
            let value = num::BigRational::from_str(&content).unwrap();
            Ok(Some((
                Token::LiteralNumber(Box::new(LiteralNumber { value, force_float })),
                this.span(lt_cursor),
            )))
        })
    }

    fn try_lex_string_literal(&mut self) -> fb::Result<bool> {
        const QUOTE_CHAR: u8 = b'"';

        self.wrap_single_token_lexer(|this| -> fb::Result<Option<(Token, Span)>> {
            let lt_cursor = this.cursor();

            // Check for opening quote
            if !this.input_stream.match_byte(QUOTE_CHAR) {
                return Ok(None);
            }

            let mut content = String::new();

            loop {
                let b = this.input_stream.peek();

                // Check for EOF
                if b == 0 {
                    return syntax_err("Unterminated string literal", this.loc(lt_cursor));
                }

                // Check for closing quote
                if b == QUOTE_CHAR {
                    this.input_stream.match_byte(QUOTE_CHAR);
                    break;
                }

                // Check for escape sequence
                if b == b'\\' {
                    let escaped_char = this.try_lex_escape_sequence(QUOTE_CHAR)?;
                    content.push(escaped_char);
                } else {
                    // Regular character
                    this.input_stream.match_byte(b);
                    content.push(b as char);
                }
            }

            let token = Token::LiteralString(Box::new(LiteralString { content }));
            Ok(Some((token, this.span(lt_cursor))))
        })
    }

    fn try_lex_escape_sequence(&mut self, quote_char: u8) -> fb::Result<char> {
        // Consume the backslash
        if !self.input_stream.match_byte(b'\\') {
            return syntax_err(
                "Internal error: expected backslash",
                self.loc(self.cursor()),
            );
        }

        let b = self.input_stream.peek();
        if b == 0 {
            return syntax_err("Unexpected EOF in escape sequence", self.loc(self.cursor()));
        }

        self.input_stream.match_byte(b);

        match b {
            b'\\' => Ok('\\'),
            b'n' => Ok('\n'),
            b't' => Ok('\t'),
            b'r' => Ok('\r'),
            b'0' => Ok('\0'),
            c if c == quote_char => Ok(c as char),
            _ => syntax_err(
                format!("Unknown escape sequence: \\{}", b as char),
                self.loc(self.cursor()),
            ),
        }
    }
    fn try_lex_eol(&mut self) -> fb::Result<bool> {
        let lt_cursor = self.cursor();
        if self.input_stream.match_byte(b'\n') {
            // Count the number of spaces in the indentation.
            let mut indent_count = 0;
            loop {
                if self.input_stream.match_byte(b' ') {
                    indent_count += 1;
                    continue;
                }
                if self.input_stream.match_byte(b'\n') {
                    indent_count = 0;
                    continue;
                }
                if self.try_skip_line_comment() {
                    continue;
                }
                if self.input_stream.match_byte(b'\t') {
                    return syntax_err(
                        r"Tab character '\t' not allowed in indentation. Use spaces instead.",
                        self.loc(lt_cursor),
                    );
                }
                if self.input_stream.match_byte_if(|b| b.is_ascii_whitespace()) != 0 {
                    return syntax_err(
                        r"Expected only space characters in indentation.",
                        self.loc(lt_cursor),
                    );
                }
                break;
            }

            // Compute a span for the EOL, Indent, and Dedent tokens we will generate next.
            let indent_span = self.span(lt_cursor);

            // Always enqueue an Eol token.
            self.lookahead_buffer
                .push_back((Token::Eol, indent_span.clone()));

            // If the indent count changed, push one Indent, or several Dedent tokens. Update the indent stack.
            if indent_count > *self.indent_stack.last().unwrap() {
                // Indent => push an indent to the stack, enqueue an indent token.
                self.indent_stack.push(indent_count);
                self.lookahead_buffer
                    .push_back((Token::Indent, indent_span));
            } else if indent_count < *self.indent_stack.last().unwrap() {
                // Dedent => pop the stack until the indent count matches, enqueue dedent tokens.
                while indent_count < *self.indent_stack.last().unwrap() {
                    self.lookahead_buffer
                        .push_back((Token::Dedent, indent_span.clone()));
                    self.indent_stack.pop();
                }
                if indent_count != *self.indent_stack.last().unwrap() {
                    return syntax_err(
                        format!(
                            r"Unexpected indentation level. Expected {}, found {}.",
                            self.indent_stack.last().unwrap(),
                            indent_count
                        ),
                        self.loc(lt_cursor),
                    );
                }
            }

            // Done:
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn wrap_single_token_lexer(
        &mut self,
        lex_one: impl FnOnce(&mut Self) -> fb::Result<Option<(Token, Span)>>,
    ) -> fb::Result<bool> {
        match lex_one(self)? {
            Some((token, span)) => {
                self.lookahead_buffer.push_back((token, span));
                Ok(true)
            }
            None => Ok(false),
        }
    }
    fn loc(&self, lt_cursor: usize) -> fb::Loc {
        fb::Loc::File(self.span(lt_cursor))
    }
    fn span(&self, lt_cursor: usize) -> Span {
        Span::new(
            self.input_stream.source(),
            lt_cursor,
            self.input_stream.offset(),
        )
    }
    fn cursor(&self) -> usize {
        self.input_stream.offset()
    }
    fn parent_lexer(&self) -> &Lexer {
        &self.parent_lexer
    }
    pub fn source(&self) -> &Source {
        self.input_stream.source()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Lid(Symbol),
    Uid(Symbol),
    Hole(Symbol),

    LiteralNumber(Box<LiteralNumber>),
    LiteralString(Box<LiteralString>),

    Indent,
    Dedent,
    Eol,

    KwDef,
    KwLet,
    KwIf,
    KwThen,
    KwElif,
    KwElse,
    KwMatch,
    KwU8,
    KwU16,
    KwU32,
    KwU64,
    KwI8,
    KwI16,
    KwI32,
    KwI64,
    KwF32,
    KwF64,
    KwBool,
    KwString,
    KwSymbol,
    KwUnit,
    KwFnUid,
    KwTrue,
    KwFalse,

    LParen,
    RParen,
    LSqBrk,
    RSqBrk,
    LCurly,
    RCurly,
    Dot,
    Comma,
    Quote,
    Colon,
    Semicolon,
    Asterisk,
    Percent,
    FSlash,
    DblFSlash,
    Plus,
    Minus,
    LShift,
    RShift,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,
    Bang,
    NotEq,
    Eq,
    DblEq,
    ThinRtArrow,
    ThinLtArrow,
    ThickRtArrow,
    Pipe,
    Ampersand,
    DblPipe,
    DblAmpersand,
    Caret,
}

//
// ByteStream
//

struct ByteStream {
    source: Source,
    offset: usize,
}
impl ByteStream {
    fn new(source: Source) -> Self {
        Self { source, offset: 0 }
    }
    fn source(&self) -> &Source {
        &self.source
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn source_bytes(&self) -> &[u8] {
        self.source.text().as_bytes()
    }
    fn peek(&self) -> u8 {
        self.source_bytes().get(self.offset).cloned().unwrap_or(0)
    }
    fn match_byte(&mut self, byte: u8) -> bool {
        self.match_byte_if(|b| b == byte) != 0
    }
    fn match_byte_if(&mut self, byte_predicate: impl FnOnce(u8) -> bool) -> u8 {
        let peek_byte = self.peek();
        if peek_byte != 0 && byte_predicate(peek_byte) {
            self.offset += 1;
            peek_byte
        } else {
            0
        }
    }
    fn match_bstr<const N: usize>(&mut self, s: &'static [u8; N]) -> bool {
        if self.offset + s.len() > self.source_bytes().len() {
            return false;
        }
        for (i, b) in s.iter().cloned().enumerate() {
            if self.source_bytes()[self.offset + i] != b {
                return false;
            }
        }
        self.offset += s.len();
        true
    }
}

//
// Helpers
//

fn syntax_err<T>(title: impl Into<String>, loc: fb::Loc) -> fb::Result<T> {
    Err(fb::Error::SyntaxError(fb::SyntaxError {
        title: title.into(),
        expected: None,
        loc,
    }))
}

//
// Tests:
//

#[cfg(test)]
struct TestFixture {
    pub source: Source,
    pub lexer_manager: Arc<Lexer>,
}
#[cfg(test)]
impl TestFixture {
    pub fn new<S: Into<String>, C: AsRef<[u8]>>(name: S, content: C) -> Self {
        let lexer_manager = Lexer::new(Config::default());
        let text = std::str::from_utf8(content.as_ref()).unwrap();
        let source = Source::new(name, text, lexer_manager.config());
        Self {
            source,
            lexer_manager,
        }
    }
    pub fn content(&self) -> &[u8] {
        self.source.text().as_bytes()
    }
    pub fn new_byte_stream(&self) -> ByteStream {
        ByteStream::new(self.source.clone())
    }
    pub fn new_lexer(&self) -> TokenStream {
        self.lexer_manager.lex(self.source.clone())
    }
    pub fn intern(&self, s: &str) -> Symbol {
        Symbol::from(s)
    }
}

#[cfg(test)]
mod lexer_tests {
    use super::*;

    fn dedent(s: &str) -> String {
        let lines = s.lines().collect::<Vec<_>>();
        let min_indent = lines
            .iter()
            .filter(|line| !line.is_empty())
            .map(|line| line.chars().take_while(|&c| c.is_whitespace()).count())
            .min()
            .unwrap_or(0);
        eprintln!("Min indent: {min_indent}");
        lines
            .iter()
            .map(|line| String::from(&line[std::cmp::min(min_indent, line.len())..]))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_punct() {
        let test_set = vec![
            ("(", Token::LParen),
            (")", Token::RParen),
            ("[", Token::LSqBrk),
            ("]", Token::RSqBrk),
            ("{", Token::LCurly),
            ("}", Token::RCurly),
            (".", Token::Dot),
            (",", Token::Comma),
            ("'", Token::Quote),
            (":", Token::Colon),
            (";", Token::Semicolon),
            ("*", Token::Asterisk),
            ("%", Token::Percent),
            ("/", Token::FSlash),
            ("//", Token::DblFSlash),
            ("+", Token::Plus),
            ("-", Token::Minus),
            ("<<", Token::LShift),
            (">>", Token::RShift),
            ("<", Token::LessThan),
            (">", Token::GreaterThan),
            ("<=", Token::LessThanEq),
            (">=", Token::GreaterThanEq),
            ("!", Token::Bang),
            ("!=", Token::NotEq),
            ("==", Token::DblEq),
            ("->", Token::ThinRtArrow),
            ("<-", Token::ThinLtArrow),
            ("=>", Token::ThickRtArrow),
            ("|", Token::Pipe),
            ("&", Token::Ampersand),
            ("||", Token::DblPipe),
            ("&&", Token::DblAmpersand),
            ("^", Token::Caret),
        ];

        let s: Vec<_> = test_set.iter().map(|(s, _)| *s).collect();
        let t: Vec<_> = test_set.iter().map(|(_, t)| (*t).clone()).collect();

        let fixture = TestFixture::new("test_punct", s.join(" "));
        let mut lexer = fixture.new_lexer();

        let text = std::str::from_utf8(fixture.content()).unwrap();
        eprintln!("{text}");

        for expected_token in t {
            let x = lexer.next_token();
            match x {
                Ok(Some((lexed_token, _))) => assert_eq!(lexed_token, expected_token),
                Ok(other) => panic!("Unexpected token: {other:#?}"),
                Err(e) => panic!("Expected successful parse, got Err:\n\n{e}"),
            }
        }
    }

    #[test]
    fn test_identifier() {
        let fixture = TestFixture::new(
            "test_ident",
            vec!["hello", "World", "_1", "a1B2", "indent_0"].join(" "),
        );
        let mut lexer = fixture.new_lexer();

        let t = vec![
            Token::Lid(fixture.intern("hello")),
            Token::Uid(fixture.intern("World")),
            Token::Hole(fixture.intern("_1")),
            Token::Lid(fixture.intern("a1B2")),
            Token::Lid(fixture.intern("indent_0")),
        ];

        let text = std::str::from_utf8(fixture.content()).unwrap();
        eprintln!("{text}");

        for expected_token in t {
            let x = lexer.next_token();
            match x {
                Ok(Some((lexed_token, _))) => assert_eq!(lexed_token, expected_token),
                Ok(other) => panic!("Unexpected token: {other:#?}"),
                Err(e) => panic!("Expected successful parse, got Err:\n\n{e}"),
            }
        }
    }

    #[test]
    fn test_indent_dedent_1() {
        let fixture = TestFixture::new(
            "test_indent_dedent",
            dedent(
                "
        indent_0
            indent_1
            indent_1
                indent_2
                    indent_3
                indent_2
        indent_0
        ",
            ),
        );
        let mut lexer = fixture.new_lexer();

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let t = vec![
            Token::Eol,
            Token::Lid(fixture.intern("indent_0")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("indent_1")),
            Token::Eol,
            Token::Lid(fixture.intern("indent_1")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("indent_2")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("indent_3")),
            Token::Eol,
            Token::Dedent,
            Token::Lid(fixture.intern("indent_2")),
            Token::Eol,
            Token::Dedent,
            Token::Dedent,
            Token::Lid(fixture.intern("indent_0")),
            Token::Eol,
        ];

        for expected_token in t {
            let x = lexer.next_token();
            match x {
                Ok(Some((lexed_token, _))) => {
                    eprintln!("{lexed_token:#?}");
                    assert_eq!(lexed_token, expected_token);
                }
                Ok(other) => panic!("Unexpected token: {other:#?}"),
                Err(e) => panic!("Expected successful parse, got Err:\n\n{e}"),
            }
        }
    }

    #[test]
    fn test_indent_dedent_2() {
        let fixture = TestFixture::new(
            "test_indent_dedent_2",
            dedent(
                "
        (
            there
                should
                    be
                        no
                            indentation
            )
        ",
            ),
        );
        let lexer = fixture.new_lexer();

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let t = vec![
            Token::Eol,
            Token::LParen,
            Token::Lid(fixture.intern("there")),
            Token::Lid(fixture.intern("should")),
            Token::Lid(fixture.intern("be")),
            Token::Lid(fixture.intern("no")),
            Token::Lid(fixture.intern("indentation")),
            Token::RParen,
            Token::Eol,
        ];

        assert_eq!(
            lexer
                .into_token_vec()
                .unwrap_or_else(|e| panic!("Lex failed: {e}"))
                .into_iter()
                .map(|(tok, _)| tok)
                .collect::<Vec<_>>(),
            t
        );
    }

    #[test]
    fn test_string_literals() {
        let fixture = TestFixture::new(
            "test_string_literals",
            b"\"hello\" \"world\\n\" \"with \\\"quotes\\\"\" \"escape\\\\test\" \"\"",
        );
        let lexer = fixture.new_lexer();

        let tokens = lexer
            .into_token_vec()
            .unwrap_or_else(|e| panic!("Lex failed: {e}"))
            .into_iter()
            .map(|(tok, _)| tok)
            .collect::<Vec<_>>();

        assert_eq!(tokens.len(), 5);

        match &tokens[0] {
            Token::LiteralString(s) => assert_eq!(s.content, "hello"),
            _ => panic!("Expected literal token"),
        }

        match &tokens[1] {
            Token::LiteralString(s) => assert_eq!(s.content, "world\n"),
            _ => panic!("Expected literal token"),
        }

        match &tokens[2] {
            Token::LiteralString(s) => assert_eq!(s.content, "with \"quotes\""),
            _ => panic!("Expected literal token"),
        }

        match &tokens[3] {
            Token::LiteralString(s) => assert_eq!(s.content, "escape\\test"),
            _ => panic!("Expected literal token"),
        }

        match &tokens[4] {
            Token::LiteralString(s) => assert_eq!(s.content, ""),
            _ => panic!("Expected literal token"),
        }
    }

    #[test]
    fn test_unterminated_string_literal() {
        let fixture = TestFixture::new("test_unterminated_string", b"\"hello");
        let lexer = fixture.new_lexer();

        let result = lexer.into_token_vec();
        assert!(
            result.is_err(),
            "Expected error for unterminated string literal"
        );
    }

    #[test]
    fn test_invalid_escape_sequence() {
        let fixture = TestFixture::new("test_invalid_escape", b"\"hello\\x\"");
        let lexer = fixture.new_lexer();

        let result = lexer.into_token_vec();
        assert!(
            result.is_err(),
            "Expected error for invalid escape sequence"
        );
    }

    #[test]
    fn test_simple_string() {
        let fixture = TestFixture::new("test_simple_string", b"\"hello\"");
        let mut lexer = fixture.new_lexer();

        match lexer.next_token() {
            Ok(Some((token, _))) => match token {
                Token::LiteralString(s) => assert_eq!(s.content, "hello"),
                _ => panic!("Expected literal token, got: {:?}", token),
            },
            Ok(other) => panic!("Unexpected result: {:?}", other),
            Err(e) => panic!("Lex error: {e}"),
        }
    }

    #[test]
    fn test_non_consecutive_indentation() {
        let fixture = TestFixture::new(
            "test_non_consecutive_indentation",
            dedent(
                "
          level_0
            level_2
              level_4
                 level_7
              level_4
            level_2
          level_0
          ",
            ),
        );
        let mut lexer = fixture.new_lexer();

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let t = vec![
            Token::Eol,
            Token::Lid(fixture.intern("level_0")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("level_2")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("level_4")),
            Token::Eol,
            Token::Indent,
            Token::Lid(fixture.intern("level_7")),
            Token::Eol,
            Token::Dedent,
            Token::Lid(fixture.intern("level_4")),
            Token::Eol,
            Token::Dedent,
            Token::Lid(fixture.intern("level_2")),
            Token::Eol,
            Token::Dedent,
            Token::Lid(fixture.intern("level_0")),
            Token::Eol,
        ];

        for expected_token in t {
            let x = lexer.next_token();
            match x {
                Ok(Some((lexed_token, _))) => {
                    eprintln!("{lexed_token:#?}");
                    assert_eq!(lexed_token, expected_token);
                }
                Ok(other) => panic!("Unexpected token: {other:#?}"),
                Err(e) => panic!("Expected successful parse, got Err:\n\n{e}"),
            }
        }
    }

    #[test]
    fn test_tab_rejection_in_indentation() {
        let fixture = TestFixture::new("test_tab_rejection", "level_0\n\tlevel_with_tab\n");
        let mut lexer = fixture.new_lexer();

        // First token should be level_0
        match lexer.next_token() {
            Ok(Some((Token::Lid(_), _))) => {}
            Ok(other) => panic!("Expected Lid, got: {other:#?}"),
            Err(e) => panic!("Expected Lid, got Err: {e}"),
        }

        // Second token should fail due to tab character in indentation
        match lexer.next_token() {
            Err(e) => {
                let error_msg = format!("{e}");
                assert!(
                    error_msg.contains("Tab character")
                        && error_msg.contains("not allowed in indentation")
                );
            }
            Ok(other) => {
                panic!("Expected error due to tab character in indentation, got: {other:#?}",)
            }
        }
    }
}

#[cfg(test)]
mod byte_stream_tests {
    use super::*;

    #[test]
    fn test_match_byte() {
        let fixture = TestFixture::new("test_match_byte", b"hi");
        let mut bs = fixture.new_byte_stream();

        assert_eq!(bs.peek(), b'h');
        assert!(bs.match_byte(b'h'));
        assert_ne!(bs.peek(), b'h');
        assert!(!bs.match_byte(b'h'));

        assert_eq!(bs.peek(), b'i');
        assert!(bs.match_byte(b'i'));
        assert_ne!(bs.peek(), b'i');
        assert!(!bs.match_byte(b'i'));

        assert_eq!(bs.peek(), 0);
        assert!(!bs.match_byte(b'_'));
        assert!(!bs.match_byte(0));
    }

    #[test]
    fn test_match_byte_if() {
        let fixture = TestFixture::new("test_match_byte_if", b"hi");
        let mut bs = fixture.new_byte_stream();

        assert_eq!(bs.peek(), b'h');
        assert_eq!(bs.match_byte_if(|b| b == b'h' || b == b'H'), b'h');
    }

    #[test]
    fn test_match_str() {
        let fixture = TestFixture::new("test_match_str", b"hello world");
        let mut bs = fixture.new_byte_stream();

        assert!(!bs.match_bstr(b"WRONG"));
        assert!(!bs.match_bstr(b"hellO"));
        assert!(!bs.match_bstr(b"hello_"));

        assert!(bs.match_bstr(b"hello"));
        assert!(bs.match_bstr(b" "));
        assert!(bs.match_bstr(b"world"));
    }
}
