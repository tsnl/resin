use arc_swap::ArcSwap;
use resin_cache::Cache;

/// Application-owned heads. Published maps and their completed values stay immutable.
pub(crate) struct Caches {
    pub sources: ArcSwap<Cache<resin_source::Source, resin_source::Source>>,
    pub syntax: ArcSwap<Cache<resin_source::Source, resin_cst::Document>>,
    pub ast: ArcSwap<Cache<resin_source::Source, resin_ast::ModuleDocument>>,
    pub hir: ArcSwap<Cache<resin_source::SourceGraph, resin_hir::Hir>>,
    pub verified: ArcSwap<Cache<resin_lir::LirKey, resin_lir::VerifiedModule>>,
    pub generated: ArcSwap<Cache<GenerationKey, resin_codegen::GeneratedProject>>,
}

impl Default for Caches {
    fn default() -> Self {
        Self {
            sources: ArcSwap::from_pointee(Cache::new(4096)),
            syntax: ArcSwap::from_pointee(Cache::new(4096)),
            ast: ArcSwap::from_pointee(Cache::new(4096)),
            hir: ArcSwap::from_pointee(Cache::new(64)),
            verified: ArcSwap::from_pointee(Cache::new(64)),
            generated: ArcSwap::from_pointee(Cache::new(32)),
        }
    }
}

/// Complete generation inputs within this process's fixed compiler/ABI contract.
/// Native settings and input validation belong to each separate toolchain request.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GenerationKey {
    pub lir: resin_lir::LirKey,
    pub host_entry: Option<String>,
}
