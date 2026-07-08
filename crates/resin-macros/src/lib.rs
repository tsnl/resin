//! `#[derive(Tree)]`: implement `resin::Tree<T>` for a struct or enum with
//! exactly one type parameter. Fields of type `T` are leaves; every other
//! field must itself implement `Tree<T>` (e.g. `Vec<T>` or a nested derive).

mod derive_tree;

extern crate proc_macro;

#[proc_macro_derive(Tree)]
pub fn derive_tree(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_tree::main(item.into()).into()
}
