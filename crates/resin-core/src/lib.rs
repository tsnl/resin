mod macros;
mod tree;

// Export basics:
pub use tree::Tree;

// Re-export "vendored" dependencies:
pub use arrayvec;
pub use paste;

// Export common collections
pub use hashbrown::{HashMap, HashSet};
