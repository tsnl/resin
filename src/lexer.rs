use hashbrown::HashMap;
use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::Arc;

use crate::token::{Token, TokenKind};
use crate::vocab::{LiteralNumber, LiteralString};
use crate::{Config, Source, Span, Symbol, fb};

//
// Lexer
//

pub struct Lexer {
    config: Config,
    keyword_map: HashMap<Symbol, TokenKind>,
}
impl Lexer {
    pub fn new(config: Config) -> Arc<Self> {
        let keyword_map = HashMap::from([
            (Symbol::from("def"), TokenKind::KwDef),
            (Symbol::from("let"), TokenKind::KwLet),
            (Symbol::from("if"), TokenKind::KwIf),
            (Symbol::from("then"), TokenKind::KwThen),
            (Symbol::from("elif"), TokenKind::KwElif),
            (Symbol::from("else"), TokenKind::KwElse),
            (Symbol::from("match"), TokenKind::KwMatch),
            (Symbol::from("u8"), TokenKind::KwU8),
            (Symbol::from("u16"), TokenKind::KwU16),
            (Symbol::from("u32"), TokenKind::KwU32),
            (Symbol::from("u64"), TokenKind::KwU64),
            (Symbol::from("i8"), TokenKind::KwI8),
            (Symbol::from("i16"), TokenKind::KwI16),
            (Symbol::from("i32"), TokenKind::KwI32),
            (Symbol::from("i64"), TokenKind::KwI64),
            (Symbol::from("f32"), TokenKind::KwF32),
            (Symbol::from("f64"), TokenKind::KwF64),
            (Symbol::from("bool"), TokenKind::KwBool),
            (Symbol::from("string"), TokenKind::KwString),
            (Symbol::from("symbol"), TokenKind::KwSymbol),
            (Symbol::from("unit"), TokenKind::KwUnit),
            (Symbol::from("Fn"), TokenKind::KwFnUid),
            (Symbol::from("true"), TokenKind::KwTrue),
            (Symbol::from("false"), TokenKind::KwFalse),
            (Symbol::from("grad"), TokenKind::KwGrad),
            (Symbol::from("as"), TokenKind::KwAs),
        ]);
        Arc::new(Self {
            config,
            keyword_map,
        })
    }
    pub fn lex(self: &Arc<Self>, source: Source) -> fb::Result<Vec<Token>> {
        let mut state = LexState::new(source, self.clone());
        let mut tokens = vec![];
        while let Some(tok) = state.next()? {
            tokens.push(tok);
        }
        Ok(tokens)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    fn keyword_map(&self) -> &HashMap<Symbol, TokenKind> {
        &self.keyword_map
    }
}

//
// LexState (internal lexer state machine)
//

struct LexState {
    input_stream: ByteStream,
    indent_stack: Vec<usize>,
    bookend_stack: Vec<Bookend>, // (), [], {}
    lookahead_buffer: VecDeque<Token>,
    parent_lexer: Arc<Lexer>,
}
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bookend {
    Paren,
    SqBrk,
    Curly,
}
impl LexState {
    const INDENT_CAPACITY: usize = 16;
    const BOOKEND_CAPACITY: usize = 32;

    fn new(source: Source, manager: Arc<Lexer>) -> Self {
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
    fn next(&mut self) -> fb::Result<Option<Token>> {
        self.try_growing_lookahead_buffer_to_n(1)?;
        Ok(self.lookahead_buffer.pop_front())
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

        // If at EOF, emit trailing Dedents and early out:
        if self.input_stream.peek() == 0 {
            let eof_span = self.span(self.cursor());
            while self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                self.lookahead_buffer.push_back(Token {
                    kind: TokenKind::Dedent,
                    span: eof_span.clone(),
                });
            }
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
        lexer_err(
            format!(
                "Unexpected byte in source file: {:#?} (0x{:02x})",
                str::from_utf8(&[self.input_stream.peek()]).unwrap_or("(?)"),
                self.input_stream.peek()
            ),
            self.span(self.cursor()),
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
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<Token>> {
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
                return Ok(Some(Token {
                    kind: keyword.clone(),
                    span: this.span(lt_cursor),
                }));
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

            let kind = match opt_is_lid_not_uid {
                Some(true) => TokenKind::Lid(name_symbol),
                Some(false) => TokenKind::Uid(name_symbol),
                None => TokenKind::Hole(name_symbol),
            };
            Ok(Some(Token {
                kind,
                span: this.span(lt_cursor),
            }))
        })
    }
    fn try_lex_punct(&mut self) -> fb::Result<bool> {
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<Token>> {
            let lt_cursor = this.cursor();
            macro_rules! ok_spanned_token {
                ($kind:expr) => {
                    Ok(Some(Token {
                        kind: $kind,
                        span: this.span(lt_cursor),
                    }))
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
                            return lexer_err(
                                format!("Unmatched closing bookend: see {:?}.", $bookend),
                                this.span(lt_cursor),
                            );
                        }
                        Some(b) if b == $bookend => {
                            // OK
                        }
                        Some(b) => {
                            return lexer_err(
                                format!(
                                    "Unmatched closing bookend: expected {:?}, but matched {:?}.",
                                    $bookend, b,
                                ),
                                this.span(lt_cursor),
                            );
                        }
                    }
                };
            }
            if this.input_stream.match_byte(b'(') {
                push_bookend!(Bookend::Paren);
                return ok_spanned_token!(TokenKind::LParen);
            }
            if this.input_stream.match_byte(b')') {
                pop_bookend_else_return_error!(Bookend::Paren);
                return ok_spanned_token!(TokenKind::RParen);
            }
            if this.input_stream.match_byte(b'[') {
                push_bookend!(Bookend::SqBrk);
                return ok_spanned_token!(TokenKind::LSqBrk);
            }
            if this.input_stream.match_byte(b']') {
                pop_bookend_else_return_error!(Bookend::SqBrk);
                return ok_spanned_token!(TokenKind::RSqBrk);
            }
            if this.input_stream.match_byte(b'{') {
                push_bookend!(Bookend::Curly);
                return ok_spanned_token!(TokenKind::LCurly);
            }
            if this.input_stream.match_byte(b'}') {
                pop_bookend_else_return_error!(Bookend::Curly);
                return ok_spanned_token!(TokenKind::RCurly);
            }
            if this.input_stream.match_byte(b'.') {
                return ok_spanned_token!(TokenKind::Dot);
            }
            if this.input_stream.match_byte(b',') {
                return ok_spanned_token!(TokenKind::Comma);
            }
            if this.input_stream.match_byte(b'\'') {
                return ok_spanned_token!(TokenKind::Quote);
            }
            if this.input_stream.match_byte(b':') {
                if this.input_stream.match_byte(b':') {
                    return ok_spanned_token!(TokenKind::DblColon);
                }
                return ok_spanned_token!(TokenKind::Colon);
            }
            if this.input_stream.match_byte(b';') {
                return ok_spanned_token!(TokenKind::Semicolon);
            }
            if this.input_stream.match_byte(b'*') {
                return ok_spanned_token!(TokenKind::Asterisk);
            }
            if this.input_stream.match_byte(b'/') {
                if this.input_stream.match_byte(b'/') {
                    return ok_spanned_token!(TokenKind::DblFSlash);
                }
                return ok_spanned_token!(TokenKind::FSlash);
            }
            if this.input_stream.match_byte(b'%') {
                return ok_spanned_token!(TokenKind::Percent);
            }
            if this.input_stream.match_byte(b'+') {
                return ok_spanned_token!(TokenKind::Plus);
            }
            if this.input_stream.match_byte(b'-') {
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(TokenKind::ThinRtArrow);
                }
                return ok_spanned_token!(TokenKind::Minus);
            }
            if this.input_stream.match_byte(b'<') {
                if this.input_stream.match_byte(b'<') {
                    return ok_spanned_token!(TokenKind::LShift);
                }
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(TokenKind::LessThanEq);
                }
                if this.input_stream.match_byte(b'-') {
                    return ok_spanned_token!(TokenKind::ThinLtArrow);
                }
                return ok_spanned_token!(TokenKind::LessThan);
            }
            if this.input_stream.match_byte(b'>') {
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(TokenKind::RShift);
                }
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(TokenKind::GreaterThanEq);
                }
                return ok_spanned_token!(TokenKind::GreaterThan);
            }
            if this.input_stream.match_byte(b'=') {
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(TokenKind::DblEq);
                }
                if this.input_stream.match_byte(b'>') {
                    return ok_spanned_token!(TokenKind::ThickRtArrow);
                }
                return ok_spanned_token!(TokenKind::Eq);
            }
            if this.input_stream.match_byte(b'!') {
                if this.input_stream.match_byte(b'=') {
                    return ok_spanned_token!(TokenKind::NotEq);
                }
                return ok_spanned_token!(TokenKind::Bang);
            }
            if this.input_stream.match_byte(b'|') {
                if this.input_stream.match_byte(b'|') {
                    return ok_spanned_token!(TokenKind::DblPipe);
                }
                return ok_spanned_token!(TokenKind::Pipe);
            }
            if this.input_stream.match_byte(b'&') {
                if this.input_stream.match_byte(b'&') {
                    return ok_spanned_token!(TokenKind::DblAmpersand);
                }
                return ok_spanned_token!(TokenKind::Ampersand);
            }
            if this.input_stream.match_byte(b'^') {
                return ok_spanned_token!(TokenKind::Caret);
            }
            if this.input_stream.match_byte(b'~') {
                return ok_spanned_token!(TokenKind::Tilde);
            }
            Ok(None)
        })
    }
    fn try_lex_number(&mut self) -> fb::Result<bool> {
        self.wrap_single_token_lexer(|this| -> fb::Result<Option<Token>> {
            let lt_cursor = this.cursor();

            // Scan the number's characters:
            let mut bs = vec![];
            let mut force_float = false;
            let mut has_digits_before_exponent = false;

            // First character must be a digit (or a dot followed by a digit,
            // for literals like `.5`, though in practice the dot is consumed
            // as punctuation before we get here).
            let b0 = this.input_stream.match_byte_if(|b| b.is_ascii_digit());
            if b0 == 0 {
                // Check for leading-dot float (e.g. `.5`)
                if this.input_stream.peek() == b'.'
                    && this.input_stream.peek_ahead(1).is_ascii_digit()
                {
                    this.input_stream.match_byte(b'.');
                    bs.push(b'.');
                    force_float = true;
                } else {
                    return Ok(None);
                }
            } else {
                bs.push(b0);
                has_digits_before_exponent = true;
            }

            // Continue scanning digits. Only consume a '.' if followed by a
            // digit — otherwise it's a dot-access (e.g. `3.field`).
            loop {
                let b1 = this.input_stream.match_byte_if(|b| b.is_ascii_digit());
                if b1 != 0 {
                    has_digits_before_exponent = true;
                    bs.push(b1);
                    continue;
                }
                if this.input_stream.peek() == b'.'
                    && this.input_stream.peek_ahead(1).is_ascii_digit()
                {
                    if force_float {
                        return lexer_err(
                            r"Unexpected '.' in number literal.",
                            this.span(lt_cursor),
                        );
                    }
                    force_float = true;
                    this.input_stream.match_byte(b'.');
                    bs.push(b'.');
                    continue;
                }
                break;
            }

            // Ensure at least one digit was seen so far.
            if !has_digits_before_exponent {
                return lexer_err(
                    r"Expected at least one digit in number literal.",
                    this.span(lt_cursor),
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
                    return lexer_err(
                        r"Expected at least one digit in exponent of number literal.",
                        this.span(lt_cursor),
                    );
                }

                if sign == b'-' {
                    force_float = true;
                }
            }

            let content = String::from_utf8(bs).unwrap();
            let value = num::BigRational::from_str(&content).unwrap();
            Ok(Some(Token {
                kind: TokenKind::LiteralNumber(Box::new(LiteralNumber { value, force_float })),
                span: this.span(lt_cursor),
            }))
        })
    }

    fn try_lex_string_literal(&mut self) -> fb::Result<bool> {
        const QUOTE_CHAR: u8 = b'"';

        self.wrap_single_token_lexer(|this| -> fb::Result<Option<Token>> {
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
                    return lexer_err("Unterminated string literal", this.span(lt_cursor));
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

            Ok(Some(Token {
                kind: TokenKind::LiteralString(Box::new(LiteralString { content })),
                span: this.span(lt_cursor),
            }))
        })
    }

    fn try_lex_escape_sequence(&mut self, quote_char: u8) -> fb::Result<char> {
        // Consume the backslash
        if !self.input_stream.match_byte(b'\\') {
            return lexer_err(
                "Internal error: expected backslash",
                self.span(self.cursor()),
            );
        }

        let b = self.input_stream.peek();
        if b == 0 {
            return lexer_err(
                "Unexpected EOF in escape sequence",
                self.span(self.cursor()),
            );
        }

        self.input_stream.match_byte(b);

        match b {
            b'\\' => Ok('\\'),
            b'n' => Ok('\n'),
            b't' => Ok('\t'),
            b'r' => Ok('\r'),
            b'0' => Ok('\0'),
            c if c == quote_char => Ok(c as char),
            _ => lexer_err(
                format!("Unknown escape sequence: \\{}", b as char),
                self.span(self.cursor()),
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
                    return lexer_err(
                        r"Tab character '\t' not allowed in indentation. Use spaces instead.",
                        self.span(lt_cursor),
                    );
                }
                if self.input_stream.match_byte_if(|b| b.is_ascii_whitespace()) != 0 {
                    return lexer_err(
                        r"Expected only space characters in indentation.",
                        self.span(lt_cursor),
                    );
                }
                break;
            }

            // Compute a span for the EOL, Indent, and Dedent tokens we will generate next.
            let indent_span = self.span(lt_cursor);

            // Always enqueue an Eol token.
            self.lookahead_buffer.push_back(Token {
                kind: TokenKind::Eol,
                span: indent_span.clone(),
            });

            // If the indent count changed, push one Indent, or several Dedent tokens. Update the indent stack.
            if indent_count > *self.indent_stack.last().unwrap() {
                // Indent => push an indent to the stack, enqueue an indent token.
                self.indent_stack.push(indent_count);
                self.lookahead_buffer.push_back(Token {
                    kind: TokenKind::Indent,
                    span: indent_span,
                });
            } else if indent_count < *self.indent_stack.last().unwrap() {
                // Dedent => pop the stack until the indent count matches, enqueue dedent tokens.
                while indent_count < *self.indent_stack.last().unwrap() {
                    self.lookahead_buffer.push_back(Token {
                        kind: TokenKind::Dedent,
                        span: indent_span.clone(),
                    });
                    self.indent_stack.pop();
                }
                if indent_count != *self.indent_stack.last().unwrap() {
                    return lexer_err(
                        format!(
                            r"Unexpected indentation level. Expected {}, found {}.",
                            self.indent_stack.last().unwrap(),
                            indent_count
                        ),
                        self.span(lt_cursor),
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
        lex_one: impl FnOnce(&mut Self) -> fb::Result<Option<Token>>,
    ) -> fb::Result<bool> {
        match lex_one(self)? {
            Some(token) => {
                self.lookahead_buffer.push_back(token);
                Ok(true)
            }
            None => Ok(false),
        }
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
    fn peek_ahead(&self, n: usize) -> u8 {
        self.source_bytes()
            .get(self.offset + n)
            .cloned()
            .unwrap_or(0)
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
// Errors
//

fn lexer_err<T>(title: impl Into<String>, span: Span) -> fb::Result<T> {
    let title = format!("[LexerError] {}", title.into());
    let span = Some(span);
    let notes = Default::default();
    let messages = vec![fb::Message { title, span, notes }];
    let outbox = fb::Outbox { messages };
    Err(outbox)
}

//
// Tests:
//

#[cfg(test)]
struct TestFixture {
    pub source: Source,
    pub lexer: Arc<Lexer>,
}
#[cfg(test)]
impl TestFixture {
    pub fn new<S: Into<String>, C: AsRef<[u8]>>(name: S, content: C) -> Self {
        let lexer = Lexer::new(Config::default());
        let text = std::str::from_utf8(content.as_ref()).unwrap();
        let source = Source::new(name, text, lexer.config());
        Self { source, lexer }
    }
    pub fn lex(&self) -> Vec<Token> {
        self.lexer
            .lex(self.source.clone())
            .unwrap_or_else(|e| panic!("Lex failed: {e:?}"))
    }
    pub fn try_lex(&self) -> fb::Result<Vec<Token>> {
        self.lexer.lex(self.source.clone())
    }
    pub fn content(&self) -> &[u8] {
        self.source.text().as_bytes()
    }
    pub fn new_byte_stream(&self) -> ByteStream {
        ByteStream::new(self.source.clone())
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

    fn assert_kinds(tokens: &[Token], expected: &[TokenKind]) {
        let kinds: Vec<&TokenKind> = tokens.iter().map(|t| &t.kind).collect();
        let expected_refs: Vec<&TokenKind> = expected.iter().collect();
        assert_eq!(kinds, expected_refs);
    }

    #[test]
    fn test_punct() {
        let test_set = vec![
            ("(", TokenKind::LParen),
            (")", TokenKind::RParen),
            ("[", TokenKind::LSqBrk),
            ("]", TokenKind::RSqBrk),
            ("{", TokenKind::LCurly),
            ("}", TokenKind::RCurly),
            (".", TokenKind::Dot),
            (",", TokenKind::Comma),
            ("'", TokenKind::Quote),
            (":", TokenKind::Colon),
            (";", TokenKind::Semicolon),
            ("*", TokenKind::Asterisk),
            ("%", TokenKind::Percent),
            ("/", TokenKind::FSlash),
            ("//", TokenKind::DblFSlash),
            ("+", TokenKind::Plus),
            ("-", TokenKind::Minus),
            ("<<", TokenKind::LShift),
            (">>", TokenKind::RShift),
            ("<", TokenKind::LessThan),
            (">", TokenKind::GreaterThan),
            ("<=", TokenKind::LessThanEq),
            (">=", TokenKind::GreaterThanEq),
            ("!", TokenKind::Bang),
            ("!=", TokenKind::NotEq),
            ("==", TokenKind::DblEq),
            ("->", TokenKind::ThinRtArrow),
            ("<-", TokenKind::ThinLtArrow),
            ("=>", TokenKind::ThickRtArrow),
            ("|", TokenKind::Pipe),
            ("&", TokenKind::Ampersand),
            ("||", TokenKind::DblPipe),
            ("&&", TokenKind::DblAmpersand),
            ("^", TokenKind::Caret),
            ("~", TokenKind::Tilde),
        ];

        let s: Vec<_> = test_set.iter().map(|(s, _)| *s).collect();
        let expected: Vec<_> = test_set.into_iter().map(|(_, t)| t).collect();

        let fixture = TestFixture::new("test_punct", s.join(" "));
        let tokens = fixture.lex();

        eprintln!("{}", std::str::from_utf8(fixture.content()).unwrap());

        assert_kinds(&tokens, &expected);
    }

    #[test]
    fn test_identifier() {
        let fixture = TestFixture::new(
            "test_ident",
            vec!["hello", "World", "_1", "a1B2", "indent_0"].join(" "),
        );

        let expected = vec![
            TokenKind::Lid(fixture.intern("hello")),
            TokenKind::Uid(fixture.intern("World")),
            TokenKind::Hole(fixture.intern("_1")),
            TokenKind::Lid(fixture.intern("a1B2")),
            TokenKind::Lid(fixture.intern("indent_0")),
        ];

        let tokens = fixture.lex();
        assert_kinds(&tokens, &expected);
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

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let expected = vec![
            TokenKind::Eol,
            TokenKind::Lid(fixture.intern("indent_0")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("indent_1")),
            TokenKind::Eol,
            TokenKind::Lid(fixture.intern("indent_1")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("indent_2")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("indent_3")),
            TokenKind::Eol,
            TokenKind::Dedent,
            TokenKind::Lid(fixture.intern("indent_2")),
            TokenKind::Eol,
            TokenKind::Dedent,
            TokenKind::Dedent,
            TokenKind::Lid(fixture.intern("indent_0")),
            TokenKind::Eol,
        ];

        assert_kinds(&fixture.lex(), &expected);
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

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let expected = vec![
            TokenKind::Eol,
            TokenKind::LParen,
            TokenKind::Lid(fixture.intern("there")),
            TokenKind::Lid(fixture.intern("should")),
            TokenKind::Lid(fixture.intern("be")),
            TokenKind::Lid(fixture.intern("no")),
            TokenKind::Lid(fixture.intern("indentation")),
            TokenKind::RParen,
            TokenKind::Eol,
        ];

        assert_kinds(&fixture.lex(), &expected);
    }

    #[test]
    fn test_string_literals() {
        let fixture = TestFixture::new(
            "test_string_literals",
            b"\"hello\" \"world\\n\" \"with \\\"quotes\\\"\" \"escape\\\\test\" \"\"",
        );
        let tokens = fixture.lex();

        assert_eq!(tokens.len(), 5);

        match &tokens[0].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, "hello"),
            _ => panic!("Expected literal token"),
        }
        match &tokens[1].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, "world\n"),
            _ => panic!("Expected literal token"),
        }
        match &tokens[2].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, "with \"quotes\""),
            _ => panic!("Expected literal token"),
        }
        match &tokens[3].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, "escape\\test"),
            _ => panic!("Expected literal token"),
        }
        match &tokens[4].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, ""),
            _ => panic!("Expected literal token"),
        }
    }

    #[test]
    fn test_unterminated_string_literal() {
        let fixture = TestFixture::new("test_unterminated_string", b"\"hello");
        assert!(
            fixture.try_lex().is_err(),
            "Expected error for unterminated string literal"
        );
    }

    #[test]
    fn test_invalid_escape_sequence() {
        let fixture = TestFixture::new("test_invalid_escape", b"\"hello\\x\"");
        assert!(
            fixture.try_lex().is_err(),
            "Expected error for invalid escape sequence"
        );
    }

    #[test]
    fn test_simple_string() {
        let fixture = TestFixture::new("test_simple_string", b"\"hello\"");
        let tokens = fixture.lex();

        assert_eq!(tokens.len(), 1);
        match &tokens[0].kind {
            TokenKind::LiteralString(s) => assert_eq!(s.content, "hello"),
            other => panic!("Expected literal token, got: {:?}", other),
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

        eprintln!(
            "Source:\n{}\n",
            std::str::from_utf8(fixture.content()).unwrap()
        );

        let expected = vec![
            TokenKind::Eol,
            TokenKind::Lid(fixture.intern("level_0")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("level_2")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("level_4")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("level_7")),
            TokenKind::Eol,
            TokenKind::Dedent,
            TokenKind::Lid(fixture.intern("level_4")),
            TokenKind::Eol,
            TokenKind::Dedent,
            TokenKind::Lid(fixture.intern("level_2")),
            TokenKind::Eol,
            TokenKind::Dedent,
            TokenKind::Lid(fixture.intern("level_0")),
            TokenKind::Eol,
        ];

        assert_kinds(&fixture.lex(), &expected);
    }

    #[test]
    fn test_eof_emits_trailing_dedents() {
        // Source ends mid-indent without trailing newline
        let fixture = TestFixture::new("test_eof_dedents", "level_0\n  level_1\n    level_2");

        let expected = vec![
            TokenKind::Lid(fixture.intern("level_0")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("level_1")),
            TokenKind::Eol,
            TokenKind::Indent,
            TokenKind::Lid(fixture.intern("level_2")),
            TokenKind::Dedent,
            TokenKind::Dedent,
        ];

        assert_kinds(&fixture.lex(), &expected);
    }

    #[test]
    fn test_tab_rejection_in_indentation() {
        let fixture = TestFixture::new("test_tab_rejection", "level_0\n\tlevel_with_tab\n");
        let result = fixture.try_lex();
        match result {
            Err(e) => {
                let error_msg = format!("{e:?}");
                assert!(
                    error_msg.contains("Tab character")
                        && error_msg.contains("not allowed in indentation")
                );
            }
            Ok(_) => {
                panic!("Expected error due to tab character in indentation");
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
