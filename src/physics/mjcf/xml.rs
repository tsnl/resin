//! A minimal XML parser — just enough for MJCF files, from scratch.
//!
//! MJCF keeps all data in attributes, so this parser keeps only element
//! names, attributes, and children; text content is skipped. No namespaces,
//! no CDATA, no DTDs — parse errors carry a line number instead.

/// One parsed element: `<name attr="value" ...> children </name>`.
#[derive(Debug, Clone)]
pub struct Element {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Element>,
}

impl Element {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// Direct children with the given tag name.
    pub fn children_named<'a>(&'a self, name: &str) -> impl Iterator<Item = &'a Element> + use<'a> {
        let name = name.to_string();
        self.children.iter().filter(move |child| child.name == name)
    }

    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children_named(name).next()
    }
}

/// Parse a whole document and return its root element.
pub fn parse(text: &str) -> Result<Element, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
        line: 1,
    };
    parser.skip_misc()?;
    let root = parser.element()?;
    parser.skip_misc()?;
    if parser.at < parser.bytes.len() {
        return Err(parser.fail("content after the root element"));
    }
    Ok(root)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    line: usize,
}

impl Parser<'_> {
    fn fail(&self, message: &str) -> String {
        format!("XML parse error, line {}: {message}", self.line)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.at += 1;
        if byte == b'\n' {
            self.line += 1;
        }
        Some(byte)
    }

    fn eat(&mut self, expected: &str) -> bool {
        if self.bytes[self.at..].starts_with(expected.as_bytes()) {
            for _ in 0..expected.len() {
                self.bump();
            }
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.bump();
        }
    }

    /// Skip whitespace, comments, `<?...?>` declarations, and text content
    /// (MJCF elements carry no meaningful text).
    fn skip_misc(&mut self) -> Result<(), String> {
        loop {
            self.skip_whitespace();
            if self.eat("<!--") {
                while !self.eat("-->") {
                    if self.bump().is_none() {
                        return Err(self.fail("unterminated comment"));
                    }
                }
            } else if self.eat("<?") {
                while !self.eat("?>") {
                    if self.bump().is_none() {
                        return Err(self.fail("unterminated <?...?> declaration"));
                    }
                }
            } else if self.peek().is_some() && self.peek() != Some(b'<') {
                self.bump(); // text content: skip
            } else {
                return Ok(());
            }
        }
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.at;
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b':' | b'.')
        ) {
            self.bump();
        }
        if self.at == start {
            return Err(self.fail("expected a name"));
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned())
    }

    fn element(&mut self) -> Result<Element, String> {
        if !self.eat("<") {
            return Err(self.fail("expected '<'"));
        }
        let name = self.name()?;
        let mut attrs = Vec::new();

        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(b'/') => {
                    self.bump();
                    if !self.eat(">") {
                        return Err(self.fail("expected '>' after '/'"));
                    }
                    return Ok(Element {
                        name,
                        attrs,
                        children: Vec::new(),
                    });
                }
                Some(b'>') => {
                    self.bump();
                    break;
                }
                Some(_) => {
                    let key = self.name()?;
                    self.skip_whitespace();
                    if !self.eat("=") {
                        return Err(self.fail("expected '=' in attribute"));
                    }
                    self.skip_whitespace();
                    let quote = match self.bump() {
                        Some(q @ (b'"' | b'\'')) => q,
                        _ => return Err(self.fail("expected a quoted attribute value")),
                    };
                    let start = self.at;
                    while self.peek() != Some(quote) {
                        if self.bump().is_none() {
                            return Err(self.fail("unterminated attribute value"));
                        }
                    }
                    let raw = String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned();
                    self.bump(); // closing quote
                    attrs.push((key, unescape(&raw)));
                }
                None => return Err(self.fail("unterminated element")),
            }
        }

        // Children until our closing tag.
        let mut children = Vec::new();
        loop {
            self.skip_misc()?;
            if self.eat("</") {
                let closing = self.name()?;
                if closing != name {
                    return Err(self.fail(&format!("expected </{name}>, found </{closing}>")));
                }
                self.skip_whitespace();
                if !self.eat(">") {
                    return Err(self.fail("expected '>' in closing tag"));
                }
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if self.peek().is_none() {
                return Err(self.fail(&format!("missing </{name}>")));
            }
            children.push(self.element()?);
        }
    }
}

/// The five predefined XML entities.
fn unescape(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_elements_and_attributes() {
        let root = parse(
            r#"<?xml version="1.0"?>
            <!-- a comment -->
            <mujoco model="test">
              <option timestep="0.002" gravity="0 0 -9.81"/>
              <worldbody>
                <body name="ball" pos="0 0 1">
                  <geom type="sphere" size="0.1"/>
                </body>
              </worldbody>
            </mujoco>"#,
        )
        .unwrap();

        assert_eq!(root.name, "mujoco");
        assert_eq!(root.attr("model"), Some("test"));
        let option = root.child("option").unwrap();
        assert_eq!(option.attr("timestep"), Some("0.002"));
        let body = root.child("worldbody").unwrap().child("body").unwrap();
        assert_eq!(body.attr("pos"), Some("0 0 1"));
        assert_eq!(body.child("geom").unwrap().attr("type"), Some("sphere"));
    }

    #[test]
    fn handles_entities_and_both_quote_styles() {
        let root = parse(r#"<a note='5 &lt; 6 &amp; "fine"'/>"#).unwrap();
        assert_eq!(root.attr("note"), Some(r#"5 < 6 & "fine""#));
    }

    #[test]
    fn reports_line_numbers() {
        let err = parse("<a>\n  <b>\n</a>").unwrap_err();
        assert!(err.contains("line 3"), "{err}");
    }
}
