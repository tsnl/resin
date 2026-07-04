mod derive_tree;

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Tree)]
pub fn derive_tree(item: TokenStream) -> TokenStream {
    derive_tree::main(item)
}
