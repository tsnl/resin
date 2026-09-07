//! Small implementation helpers shared across the compiler.

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

pub(crate) use define_id;
