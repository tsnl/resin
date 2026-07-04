use proc_macro::TokenStream;
use quote::quote;

pub fn main(item: TokenStream) -> TokenStream {
    let item: syn::Item = match syn::parse(item) {
        Ok(item) => item,
        Err(err) => return err.to_compile_error().into(),
    };
    match item {
        syn::Item::Struct(item_struct) => derive_tree_struct(item_struct),
        syn::Item::Enum(item_enum) => derive_tree_enum(item_enum),
        _ => {
            let error_message =
                "Invalid type argument for #[derive(Tree)]: expected struct or enum.";
            syn::Error::new_spanned(item, error_message)
                .to_compile_error()
                .into()
        }
    }
}

fn derive_tree_struct(item_struct: syn::ItemStruct) -> TokenStream {
    let flatten = item_struct.fields.iter().map(|field| {
        let field_name = &field.ident;
        // TODO: how to determine if field is 'T' (leaf) or intermediate?
    });

    let struct_name = &item_struct.ident;
    let tree_impl = quote::quote! {
        impl resin_core::Tree<#struct_name> for #struct_name {
            fn flatten(&self, dir: List<TreePathPart>) -> impl Iterator<Item = TreeLeaf<T>> {
                todo!()
            }
            fn unflatten(leaves: impl Iterator<Item = TreeLeaf<T>>) -> Self {
                todo!()
            }
        }
    };
    tree_impl.into()
}

fn derive_tree_enum(item_enum: syn::ItemEnum) -> TokenStream {
    let enum_name = &item_enum.ident;
    let tree_impl = quote::quote! {
        impl resin_core::Tree<#enum_name> for #enum_name {
            fn flatten(&self) -> impl Iterator<Item = TreeLeaf<T>> {
                todo!()
            }
            fn unflatten(leaves: impl Iterator<Item = TreeLeaf<T>>) -> Self {
                todo!()
            }
        }
    };
    tree_impl.into()
}
