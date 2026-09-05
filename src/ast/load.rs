use std::{
    collections::HashSet,
    fmt, fs,
    path::{Path, PathBuf},
};

use super::{AstGen, SourceFile, StmtKind};

#[derive(Debug)]
pub struct SourceError {
    pub path: PathBuf,
    pub message: String,
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for SourceError {}

pub fn load(path: &Path) -> Result<SourceFile, SourceError> {
    let mut file = SourceFile { stmts: Vec::new() };
    visit(path, &mut Vec::new(), &mut HashSet::new(), &mut file)?;
    Ok(file)
}

fn visit(
    path: &Path,
    active: &mut Vec<PathBuf>,
    loaded: &mut HashSet<PathBuf>,
    file: &mut SourceFile,
) -> Result<(), SourceError> {
    let error = |message: String| SourceError {
        path: path.to_path_buf(),
        message,
    };
    let canonical = fs::canonicalize(path).map_err(|e| error(e.to_string()))?;
    if active.contains(&canonical) {
        return Err(error("cyclic source include".into()));
    }
    if !loaded.insert(canonical.clone()) {
        return Ok(());
    }
    let source = fs::read_to_string(&canonical).map_err(|e| error(e.to_string()))?;
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .map_err(|e| error(e.to_string()))?;
    let tree = parser.parse(&source, None).expect("parser language is set");
    let ast = AstGen::new(&source)
        .gen_source_file(tree.root_node())
        .map_err(|e| error(e.to_string()))?;
    active.push(canonical.clone());
    for stmt in ast.stmts {
        if let StmtKind::Include { path } = &stmt.val {
            visit(
                &canonical.parent().unwrap().join(path.as_ref()),
                active,
                loaded,
                file,
            )?;
        } else {
            file.stmts.push(stmt);
        }
    }
    active.pop();
    Ok(())
}
