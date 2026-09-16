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
    pub foreign: ArcSwap<Cache<ForeignKey, resin_toolchain::ForeignObject>>,
    pub shaders: ArcSwap<Cache<ShaderKey, Arc<[u8]>>>,
    pub optimized: ArcSwap<Cache<ToolBytesKey, Arc<[u8]>>>,
    pub native: ArcSwap<Cache<NativeKey, resin_codegen::NativeObject>>,
    pub executables: ArcSwap<Cache<LinkKey, resin_toolchain::Executable>>,
    pub source_builds: AtomicU64,
    pub syntax_builds: AtomicU64,
    pub ast_builds: AtomicU64,
    pub hir_builds: AtomicU64,
    pub verified_builds: AtomicU64,
    pub foreign_builds: AtomicU64,
    pub shader_builds: AtomicU64,
    pub shader_optimizations: AtomicU64,
    pub native_object_builds: AtomicU64,
    pub executable_builds: AtomicU64,
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
            foreign: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            shaders: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            optimized: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            native: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            executables: ArcSwap::from_pointee(Cache::new(capacity.generated)),
            source_builds: AtomicU64::new(0),
            syntax_builds: AtomicU64::new(0),
            ast_builds: AtomicU64::new(0),
            hir_builds: AtomicU64::new(0),
            verified_builds: AtomicU64::new(0),
            foreign_builds: AtomicU64::new(0),
            shader_builds: AtomicU64::new(0),
            shader_optimizations: AtomicU64::new(0),
            native_object_builds: AtomicU64::new(0),
            executable_builds: AtomicU64::new(0),
        }
    }
    pub fn counters(&self) -> crate::Counters {
        crate::Counters {
            source_builds: self.source_builds.load(Ordering::Relaxed),
            syntax_builds: self.syntax_builds.load(Ordering::Relaxed),
            ast_builds: self.ast_builds.load(Ordering::Relaxed),
            hir_builds: self.hir_builds.load(Ordering::Relaxed),
            verified_builds: self.verified_builds.load(Ordering::Relaxed),
            foreign_builds: self.foreign_builds.load(Ordering::Relaxed),
            shader_builds: self.shader_builds.load(Ordering::Relaxed),
            shader_optimizations: self.shader_optimizations.load(Ordering::Relaxed),
            native_object_builds: self.native_object_builds.load(Ordering::Relaxed),
            executable_builds: self.executable_builds.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ForeignKey {
    pub inputs: Arc<resin_toolchain::ForeignInputs>,
    pub tools: String,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ShaderKey {
    pub lir: resin_lir::LirKey,
    pub function: resin_types::FunctionId,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ToolBytesKey {
    pub bytes: Arc<[u8]>,
    pub tools: String,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NativeKey {
    pub lir: resin_lir::LirKey,
    pub entry: String,
    pub optimization: resin_codegen::NativeOptimization,
    pub inputs: Arc<resin_codegen::NativeInputs>,
    pub target: resin_protocol::Target,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LinkKey {
    /// A removed owned file requires a fresh immutable executable generation.
    pub generation: u64,
    pub objects: Vec<Arc<[u8]>>,
    pub tools: String,
}
