use arc_swap::ArcSwap;
use resin_cache::Cache;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub(crate) struct Caches {
    pub inputs: ArcSwap<Cache<String, resin_protocol::Inputs>>,
    pub sources: ArcSwap<Cache<resin_source::Source, resin_source::Source>>,
    pub syntax: ArcSwap<Cache<resin_source::Source, resin_cst::Document>>,
    pub ast: ArcSwap<Cache<resin_source::Source, resin_ast::ModuleDocument>>,
    pub hir: ArcSwap<Cache<resin_source::SourceGraph, resin_hir::Hir>>,
    pub verified: ArcSwap<Cache<resin_lir::LirKey, resin_lir::VerifiedModule>>,
    pub generated: ArcSwap<Cache<GenerationKey, resin_codegen::GeneratedProject>>,
    pub source_builds: AtomicU64,
    pub syntax_builds: AtomicU64,
    pub ast_builds: AtomicU64,
    pub hir_builds: AtomicU64,
    pub verified_builds: AtomicU64,
    pub generated_builds: AtomicU64,
}

impl Caches {
    pub fn new(capacity: &crate::Capacities) -> Self {
        Self {
            inputs: ArcSwap::from_pointee(Cache::new(capacity.inputs)),
            sources: ArcSwap::from_pointee(Cache::new(capacity.sources)),
            syntax: ArcSwap::from_pointee(Cache::new(capacity.syntax)),
            ast: ArcSwap::from_pointee(Cache::new(capacity.ast)),
            hir: ArcSwap::from_pointee(Cache::new(capacity.hir)),
            verified: ArcSwap::from_pointee(Cache::new(capacity.verified)),
            generated: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            source_builds: AtomicU64::new(0),
            syntax_builds: AtomicU64::new(0),
            ast_builds: AtomicU64::new(0),
            hir_builds: AtomicU64::new(0),
            verified_builds: AtomicU64::new(0),
            generated_builds: AtomicU64::new(0),
        }
    }
    pub fn counters(&self) -> crate::Counters {
        crate::Counters {
            source_builds: self.source_builds.load(Ordering::Relaxed),
            syntax_builds: self.syntax_builds.load(Ordering::Relaxed),
            ast_builds: self.ast_builds.load(Ordering::Relaxed),
            hir_builds: self.hir_builds.load(Ordering::Relaxed),
            verified_builds: self.verified_builds.load(Ordering::Relaxed),
            generated_builds: self.generated_builds.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GenerationKey {
    pub lir: resin_lir::LirKey,
    pub host_entry: Option<String>,
    pub headers: Arc<resin_codegen::NativeHeaders>,
    pub target: resin_protocol::Target,
    pub managed_snapshot: String,
}
