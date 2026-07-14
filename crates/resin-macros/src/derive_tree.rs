use proc_macro2::TokenStream;
use quote::quote;
use syn::{Fields, Generics, Ident, Index, ItemEnum, ItemStruct, Type};

pub fn main(item: TokenStream) -> TokenStream {
    let item: syn::Item = match syn::parse2(item) {
        Ok(item) => item,
        Err(err) => return err.to_compile_error(),
    };
    match item {
        syn::Item::Struct(item) => derive_struct(item),
        syn::Item::Enum(item) => derive_enum(item),
        other => syn::Error::new_spanned(other, "#[derive(Tree)] requires a struct or enum")
            .to_compile_error(),
    }
}

/// The single type parameter that plays the leaf role.
fn leaf_param(generics: &Generics, span: proc_macro2::Span) -> Result<Ident, TokenStream> {
    let mut params = generics.type_params();
    match (params.next(), params.next()) {
        (Some(param), None) => Ok(param.ident.clone()),
        _ => Err(
            syn::Error::new(span, "derive(Tree) requires exactly one type parameter")
                .to_compile_error(),
        ),
    }
}

fn is_leaf(ty: &Type, leaf: &Ident) -> bool {
    matches!(ty, Type::Path(p) if p.path.get_ident() == Some(leaf))
}

/// `f(x)?` for leaves, recursion for subtrees. `value` is a `&` expression.
fn try_map_expr(value: TokenStream, ty: &Type, leaf: &Ident) -> TokenStream {
    if is_leaf(ty, leaf) {
        quote! { f(#value)? }
    } else {
        quote! { resin::Tree::try_map(#value, &mut *f)? }
    }
}

fn for_each_stmt(value: TokenStream, ty: &Type, leaf: &Ident) -> TokenStream {
    if is_leaf(ty, leaf) {
        quote! { f(#value); }
    } else {
        quote! { resin::Tree::for_each(#value, &mut *f); }
    }
}

fn for_each_mut_stmt(value: TokenStream, ty: &Type, leaf: &Ident) -> TokenStream {
    if is_leaf(ty, leaf) {
        quote! { f(#value); }
    } else {
        quote! { resin::Tree::for_each_mut(#value, &mut *f); }
    }
}

/// Bindings `__f0, __f1, …` for tuple fields.
fn tuple_bindings(n: usize) -> Vec<Ident> {
    (0..n)
        .map(|i| Ident::new(&format!("__f{i}"), proc_macro2::Span::call_site()))
        .collect()
}

/// Constructor expression for the mapped container, from per-field exprs.
fn construct(path: TokenStream, fields: &Fields, exprs: Vec<TokenStream>) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let idents = named.named.iter().map(|f| f.ident.as_ref().expect("named"));
            quote! { #path { #(#idents: #exprs,)* } }
        }
        Fields::Unnamed(_) => quote! { #path(#(#exprs,)*) },
        Fields::Unit => path,
    }
}

fn derive_struct(item: ItemStruct) -> TokenStream {
    let leaf = match leaf_param(&item.generics, item.ident.span()) {
        Ok(leaf) => leaf,
        Err(err) => return err,
    };
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    let name = &item.ident;

    // Field access for `&self` methods, and destructured bindings for `&mut`.
    let (shared_values, mut_bindings): (Vec<TokenStream>, TokenStream) = match &item.fields {
        Fields::Named(named) => {
            let idents: Vec<_> = named
                .named
                .iter()
                .map(|f| f.ident.clone().expect("named"))
                .collect();
            let values = idents.iter().map(|i| quote! { &self.#i }).collect();
            (values, quote! { let Self { #(#idents),* } = self; })
        }
        Fields::Unnamed(unnamed) => {
            let values = (0..unnamed.unnamed.len())
                .map(|i| {
                    let index = Index::from(i);
                    quote! { &self.#index }
                })
                .collect();
            let bindings = tuple_bindings(unnamed.unnamed.len());
            (values, quote! { let Self(#(#bindings),*) = self; })
        }
        Fields::Unit => (Vec::new(), TokenStream::new()),
    };
    let mut_values: Vec<TokenStream> = match &item.fields {
        Fields::Named(named) => named
            .named
            .iter()
            .map(|f| {
                let ident = f.ident.as_ref().expect("named");
                quote! { #ident }
            })
            .collect(),
        Fields::Unnamed(unnamed) => tuple_bindings(unnamed.unnamed.len())
            .iter()
            .map(|b| quote! { #b })
            .collect(),
        Fields::Unit => Vec::new(),
    };
    let types: Vec<&Type> = item.fields.iter().map(|f| &f.ty).collect();

    let mapped = construct(
        quote! { #name::<U> },
        &item.fields,
        shared_values
            .iter()
            .zip(&types)
            .map(|(v, ty)| try_map_expr(v.clone(), ty, &leaf))
            .collect(),
    );
    let for_each_body: Vec<_> = shared_values
        .iter()
        .zip(&types)
        .map(|(v, ty)| for_each_stmt(v.clone(), ty, &leaf))
        .collect();
    let for_each_mut_body: Vec<_> = mut_values
        .iter()
        .zip(&types)
        .map(|(v, ty)| for_each_mut_stmt(v.clone(), ty, &leaf))
        .collect();

    quote! {
        impl #impl_generics resin::Tree<#leaf> for #name #ty_generics #where_clause {
            type Mapped<U: Clone> = #name<U>;

            fn try_map<U: Clone, E>(
                &self,
                f: &mut dyn FnMut(&#leaf) -> Result<U, E>,
            ) -> Result<#name<U>, E> {
                Ok(#mapped)
            }

            fn for_each<'a>(&'a self, f: &mut dyn FnMut(&'a #leaf)) {
                #(#for_each_body)*
            }

            fn for_each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut #leaf)) {
                #mut_bindings
                #(#for_each_mut_body)*
            }
        }
    }
}

fn derive_enum(item: ItemEnum) -> TokenStream {
    let leaf = match leaf_param(&item.generics, item.ident.span()) {
        Ok(leaf) => leaf,
        Err(err) => return err,
    };
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    let name = &item.ident;

    // Every method destructures each variant; match ergonomics bind fields
    // by shared or mutable reference as needed.
    let pattern = |variant: &syn::Variant| {
        let v = &variant.ident;
        match &variant.fields {
            Fields::Named(named) => {
                let idents = named.named.iter().map(|f| f.ident.as_ref().expect("named"));
                quote! { Self::#v { #(#idents),* } }
            }
            Fields::Unnamed(unnamed) => {
                let bindings = tuple_bindings(unnamed.unnamed.len());
                quote! { Self::#v(#(#bindings),*) }
            }
            Fields::Unit => quote! { Self::#v },
        }
    };
    let bindings = |fields: &Fields| -> Vec<TokenStream> {
        match fields {
            Fields::Named(named) => named
                .named
                .iter()
                .map(|f| {
                    let ident = f.ident.as_ref().expect("named");
                    quote! { #ident }
                })
                .collect(),
            Fields::Unnamed(unnamed) => tuple_bindings(unnamed.unnamed.len())
                .iter()
                .map(|b| quote! { #b })
                .collect(),
            Fields::Unit => Vec::new(),
        }
    };

    let try_map_arms = item.variants.iter().map(|variant| {
        let pat = pattern(variant);
        let v = &variant.ident;
        let exprs = bindings(&variant.fields)
            .into_iter()
            .zip(variant.fields.iter())
            .map(|(value, field)| try_map_expr(value, &field.ty, &leaf))
            .collect();
        let constructed = construct(quote! { #name::<U>::#v }, &variant.fields, exprs);
        quote! { #pat => #constructed }
    });
    let for_each_arms = item.variants.iter().map(|variant| {
        let pat = pattern(variant);
        let stmts: Vec<_> = bindings(&variant.fields)
            .into_iter()
            .zip(variant.fields.iter())
            .map(|(value, field)| for_each_stmt(value, &field.ty, &leaf))
            .collect();
        quote! { #pat => { #(#stmts)* } }
    });
    let for_each_mut_arms = item.variants.iter().map(|variant| {
        let pat = pattern(variant);
        let stmts: Vec<_> = bindings(&variant.fields)
            .into_iter()
            .zip(variant.fields.iter())
            .map(|(value, field)| for_each_mut_stmt(value, &field.ty, &leaf))
            .collect();
        quote! { #pat => { #(#stmts)* } }
    });

    quote! {
        impl #impl_generics resin::Tree<#leaf> for #name #ty_generics #where_clause {
            type Mapped<U: Clone> = #name<U>;

            fn try_map<U: Clone, E>(
                &self,
                f: &mut dyn FnMut(&#leaf) -> Result<U, E>,
            ) -> Result<#name<U>, E> {
                Ok(match self { #(#try_map_arms,)* })
            }

            fn for_each<'a>(&'a self, f: &mut dyn FnMut(&'a #leaf)) {
                match self { #(#for_each_arms,)* }
            }

            fn for_each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut #leaf)) {
                match self { #(#for_each_mut_arms,)* }
            }
        }
    }
}
