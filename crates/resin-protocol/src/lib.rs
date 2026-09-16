//! Versioned compiler service messages. Names are logical identities, never server paths.
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;
pub const METADATA_HEADER: &str = "resin-metadata";
pub const MAX_METADATA_BYTES: usize = 4096;

//
// Immutable source and native input snapshots
//

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputHandle {
    pub instance: String,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub name: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportBinding {
    pub importer: String,
    pub reference: String,
    pub target: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub entry: String,
    pub sources: Vec<SourceFile>,
    pub imports: Vec<ImportBinding>,
    pub headers: HeaderInputs,
    pub acquisition_diagnostics: Vec<Diagnostic>,
    pub managed_snapshot: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InputSelection {
    Full {
        inputs: Inputs,
    },
    Delta {
        base: InputHandle,
        entry: String,
        replacements: Vec<SourceFile>,
        deleted: Vec<String>,
        imports: Vec<ImportBinding>,
        headers: HeaderInputs,
        acquisition_diagnostics: Vec<Diagnostic>,
        managed_snapshot: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderInputs {
    pub bundles: Vec<HeaderBundle>,
    pub bindings: Vec<HeaderBinding>,
    pub include_roots: Vec<IncludeRoot>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderBundle {
    pub id: String,
    pub files: Vec<BundleFile>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleFile {
    pub path: String,
    pub contents_base64: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderBinding {
    pub source: String,
    pub spelling: String,
    pub target: HeaderTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HeaderTarget {
    Uploaded { bundle: String, path: String },
    Managed { root: String, path: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IncludeRoot {
    Uploaded { bundle: String, directory: String },
    Managed { root: String, directory: String },
}

//
// Diagnostics and editor queries
//

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub source: String,
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Information,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelatedDiagnostic {
    pub message: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    pub related: Vec<RelatedDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryPosition {
    pub source: String,
    pub offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Query {
    Hover { position: QueryPosition },
    Definition { position: QueryPosition },
    Completion { position: QueryPosition },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionKind {
    Function,
    Constant,
    Variable,
    Field,
    Type,
    Module,
    Keyword,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionItem {
    pub label: String,
    pub detail: Option<String>,
    pub kind: CompletionKind,
    pub insert_text: String,
    pub replace: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryResult {
    Hover {
        markdown: Option<String>,
        span: Option<Span>,
    },
    Definition {
        span: Option<Span>,
    },
    Completion {
        items: Vec<CompletionItem>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeRequest {
    pub request: String,
    pub revision: u64,
    pub inputs: InputSelection,
    pub queries: Vec<Query>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeResponse {
    pub revision: u64,
    pub input: InputHandle,
    pub diagnostics: Vec<Diagnostic>,
    pub results: Vec<QueryResult>,
    pub managed_sources: Vec<SourceFile>,
}

//
// Negotiated build contracts and artifacts
//

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub os: String,
    pub architecture: String,
    pub abi: String,
    pub runtime: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryProfile {
    Host,
    Shader,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildProfile {
    Debug,
    Release,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryTarget {
    pub export: String,
    pub profile: EntryProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildOptions {
    pub max_instances_per_function: u64,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            max_instances_per_function: 16_384,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildContract {
    pub target: Target,
    pub entry: EntryTarget,
    pub profile: BuildProfile,
    pub options: BuildOptions,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildRequest {
    pub request: String,
    pub revision: u64,
    pub inputs: InputSelection,
    pub contract: BuildContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Executable,
    Spirv,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactMetadata {
    pub name: String,
    pub kind: ArtifactKind,
    pub target: Target,
    pub length: u64,
    pub blake3: String,
    pub executable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildMetadata {
    pub revision: u64,
    pub input: InputHandle,
    pub artifact: ArtifactMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    IncompatibleProtocol,
    InputUnavailable,
    ManagedSnapshotUnavailable,
    UnsupportedTarget,
    CompilationFailed,
    Cancelled,
    Busy,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub code: ErrorCode,
    pub message: String,
    pub diagnostics: Vec<Diagnostic>,
    pub managed_sources: Vec<SourceFile>,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(output)
    }
}
impl std::error::Error for Failure {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedHeaderRoot {
    pub id: String,
    pub headers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub protocol: u32,
    pub instance: String,
    pub managed_snapshot: String,
    pub targets: Vec<Target>,
    pub entry_profiles: Vec<EntryProfile>,
    pub header_roots: Vec<ManagedHeaderRoot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_reject_unknown_fields_and_unrecognized_variants() {
        assert!(
            serde_json::from_str::<InputHandle>(r#"{"instance":"server","id":"input","extra":1}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<Query>(r#"{"kind":"hover","position":{"source":"main.resin","offset":1},"unexpected":true}"#).is_err());
        assert!(serde_json::from_str::<EntryProfile>(r#""future_target""#).is_err());
    }

    #[test]
    fn source_names_and_text_round_trip_without_path_interpretation() {
        let source = SourceFile {
            name: "../dir%25/module.resin".into(),
            text: "fn main() = { \\\"λ\\\" };\\n".into(),
        };
        assert_eq!(
            serde_json::from_slice::<SourceFile>(&serde_json::to_vec(&source).unwrap()).unwrap(),
            source
        );
    }
}
