//! Shared index-type construction for compiler phases.
//! Domain types, sources, and diagnostics belong to their own crates.

/// Define distinct index types without giving them interchangeable integer identities.
#[macro_export]
macro_rules! define_id {
    (
        $(
            $(#[$attr:meta])*
            $visibility:vis struct $name:ident(usize);
        )+
    ) => {
        $(
            $(#[$attr])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $visibility struct $name(usize);

            impl $name {
                pub const fn from_index(index: usize) -> Self {
                    Self(index)
                }

                pub const fn index(self) -> usize {
                    self.0
                }
            }
        )+
    };
}
