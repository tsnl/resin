use crate::Error;
use resin_source::prelude::*;
use resin_types::prelude::*;

impl Error {
    pub(crate) fn at(
        module: &resin_lir::Module,
        function: usize,
        instruction: Option<(usize, usize)>,
        error: Self,
    ) -> Self {
        use resin_lir::BlockId;
        let id = FunctionId::from_index(function);
        let origin = instruction
            .and_then(|(b, i)| {
                module
                    .origins
                    .instructions
                    .get(&(id, BlockId::from_index(b), i))
            })
            .or_else(|| module.origins.functions.get(&id));
        let name = module.functions[function]
            .name
            .as_deref()
            .unwrap_or("<anonymous>");
        let internal = instruction.map_or_else(
            || format!("function {name}"),
            |(b, i)| format!("function {name}, block {b}, instruction {i}"),
        );
        let Some(origin) = origin else {
            return Self(format!("{internal}: {error}"));
        };
        let location = source_context(origin);
        Self(format!("{location} ({internal}): {error}"))
    }
}

fn source_context(origin: &SourceLocation) -> String {
    let source = origin.source.text();
    let prefix = source.get(..origin.span.start).unwrap_or("");
    let line = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
    let column = prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1;
    let expression = source.get(origin.span.start..origin.span.end).unwrap_or("");
    format!("{}:{line}:{column}: {expression:?}", origin.source.name())
}

#[cfg(test)]
mod tests {
    use super::source_context;
    use resin_source::prelude::*;

    #[test]
    fn error_context_uses_the_exact_source_version_and_unicode_columns() {
        let source = Source::new("editor://buffer/λ", "first\nα + β");
        let start = source.text().find('β').unwrap();
        let origin = SourceLocation {
            source: source.clone(),
            span: Span {
                start,
                end: start + 'β'.len_utf8(),
            },
        };
        let _updated = source.with_text("replacement");
        assert_eq!(source_context(&origin), "editor://buffer/λ:2:5: \"β\"");
    }

    #[test]
    fn malformed_external_spans_do_not_panic_while_reporting_errors() {
        let source = Source::new("memory:invalid-span", "λ");
        for span in [
            Span { start: 1, end: 2 },
            Span {
                start: 99,
                end: 100,
            },
        ] {
            let origin = SourceLocation {
                source: source.clone(),
                span,
            };
            assert_eq!(source_context(&origin), "memory:invalid-span:1:1: \"\"");
        }
    }
}
