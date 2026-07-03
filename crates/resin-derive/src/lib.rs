//! Proc-macros for resin crates.

mod tree;

use proc_macro::TokenStream;
use syn::parse_macro_input;
use syn::DeriveInput;

/// `#[derive(Tree)]` for host module trees.
#[proc_macro_derive(Tree, attributes(tree))]
pub fn derive_tree(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    tree::expand(&input).into()
}
