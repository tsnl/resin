mod derive_tree;

extern crate proc_macro;

#[proc_macro_derive(Tree)]
pub fn derive_tree(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Wrapper around proc_macro2-based tokenstreams:
    derive_tree::main(item.into()).into()
}
