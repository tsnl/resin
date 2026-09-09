use crate::Error;
use resin_common::prelude::*;
use std::fmt;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<resin_lir_verifier::VerifyError> for Error {
    fn from(error: resin_lir_verifier::VerifyError) -> Self {
        Self(error.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<SourceError> for Error {
    fn from(error: SourceError) -> Self {
        Self(error.to_string())
    }
}

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
        let location = if let Some(source) = module.origins.sources.get(&origin.path) {
            let prefix = source.get(..origin.span.start).unwrap_or("");
            let line = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
            let column = prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1;
            let expression = source.get(origin.span.start..origin.span.end).unwrap_or("");
            format!("{}:{line}:{column}: {expression:?}", origin.path.display())
        } else {
            format!(
                "{}:{}..{}",
                origin.path.display(),
                origin.span.start,
                origin.span.end
            )
        };
        Self(format!("{location} ({internal}): {error}"))
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self(message)
    }
}
