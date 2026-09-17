//! Source documentation → Markdown. This local operation does not resolve imports.
use resin_cst::{DeclarationDocumentation, Documentation};
use resin_executor::{Cancellation, Execution};
use std::{io::Write, path::Path};

pub(super) async fn run(input: &Path, output: Option<&Path>) -> super::Result<i32> {
    let text = std::fs::read_to_string(input)?;
    let document =
        resin_cst::build_cst(text, None, &Execution::default(), &Cancellation::new()).await?;
    if document.tree().root_node().has_error() {
        return Err(format!("{}: cannot document malformed syntax", input.display()).into());
    }
    let documentation = document.documentation();
    if let Some(error) = documentation.diagnostics.first() {
        let source = resin_source::Source::new(input.display().to_string(), document.source());
        return Err(
            resin_source::SourceError::new(source, Some(error.span), error.val.clone()).into(),
        );
    }
    let title = input
        .file_name()
        .unwrap_or(input.as_os_str())
        .to_string_lossy();
    let markdown = render(&title, &documentation);
    if let Some(output) = output {
        // Publish only a completed document; errors preserve the previous output.
        let mut temporary =
            tempfile::NamedTempFile::new_in(output.parent().unwrap_or(Path::new(".")))?;
        temporary.write_all(markdown.as_bytes())?;
        temporary.persist(output)?;
    } else {
        std::io::stdout().lock().write_all(markdown.as_bytes())?;
    }
    Ok(0)
}

fn render(title: &str, documentation: &Documentation) -> String {
    let mut result = format!("# {}\n\n", title.replace(['\r', '\n'], " "));
    if !documentation.module.is_empty() {
        result.push_str(&documentation.module);
        result.push_str("\n\n");
    }
    for item in documentation
        .declarations
        .iter()
        .filter(|item| item.exported)
    {
        declaration(&mut result, item, "##");
        for field in documentation
            .declarations
            .iter()
            .filter(|field| field.field_owner == Some(item.span))
        {
            declaration(&mut result, field, "###");
        }
    }
    result
}

fn declaration(result: &mut String, item: &DeclarationDocumentation, heading: &str) {
    result.push_str(&format!("{heading} `{}`\n\n", item.name.val));
    let length = item
        .signature
        .split(|character| character != '`')
        .map(str::len)
        .max()
        .unwrap_or(0);
    let fence = "`".repeat(3.max(length + 1));
    result.push_str(&format!("{fence}resin\n{}\n{fence}\n\n", item.signature));
    if !item.markdown.is_empty() {
        result.push_str(&item.markdown);
        result.push_str("\n\n");
    }
}
