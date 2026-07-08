use proc_macro2::TokenStream;
use quote::quote;
use syn::{Fields, Generics, Ident, Index, ItemEnum, ItemStruct, Type, parse_quote};

pub fn main(item: TokenStream) -> TokenStream {
    let item: syn::Item = match syn::parse(item.into()) {
        Ok(item) => item,
        Err(err) => return err.to_compile_error(),
    };
    match item {
        syn::Item::Struct(s) => derive_struct(s),
        syn::Item::Enum(e) => derive_enum(e),
        _ => syn::Error::new_spanned(item, "#[derive(Tree)] requires struct or enum")
            .to_compile_error(),
    }
}

fn leaf_generics(generics: &Generics, span: proc_macro2::Span) -> Result<(Ident, Generics), TokenStream> {
    let mut params = generics.type_params();
    let leaf = match (params.next(), params.next()) {
        (Some(tp), None) => tp.ident.clone(),
        _ => {
            return Err(syn::Error::new(span, "derive Tree requires exactly one type parameter")
                .to_compile_error());
        }
    };

    let mut generics = generics.clone();
    let tp = generics.type_params_mut().next().expect("one type param");
    if !tp.bounds.iter().any(|b| {
        matches!(b, syn::TypeParamBound::Trait(t) if t.path.is_ident("Clone"))
    }) {
        tp.bounds.push(parse_quote!(Clone));
    }
    Ok((leaf, generics))
}

fn is_leaf(ty: &Type, leaf: &Ident) -> bool {
    matches!(ty, Type::Path(p) if p.path.get_ident() == Some(leaf))
}

fn map_field(value: TokenStream, ty: &Type, leaf: &Ident) -> TokenStream {
    if is_leaf(ty, leaf) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::map(#value, f) }
    }
}

fn visit_field(value: TokenStream, ty: &Type, leaf: &Ident) -> TokenStream {
    if is_leaf(ty, leaf) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::for_each_leaf(#value, &mut f) }
    }
}

fn map_fields(fields: &Fields, leaf: &Ident, from_self: bool) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let mapped = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &self.#ident }
                } else {
                    quote! { #ident }
                };
                let mapped = map_field(value, &field.ty, leaf);
                quote! { #ident: #mapped }
            });
            quote! { #(#mapped,)* }
        }
        Fields::Unnamed(unnamed) => {
            let mapped = unnamed.unnamed.iter().enumerate().map(|(i, field)| {
                let value = if from_self {
                    let index = Index::from(i);
                    quote! { &self.#index }
                } else {
                    let binding = Ident::new(&format!("__f{i}"), proc_macro2::Span::call_site());
                    quote! { #binding }
                };
                map_field(value, &field.ty, leaf)
            });
            quote! { #(#mapped,)* }
        }
        Fields::Unit => TokenStream::new(),
    }
}

fn match_pattern(variant: &Ident, fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let idents = named.named.iter().map(|f| f.ident.as_ref().expect("named field"));
            quote! { Self::#variant { #(#idents),* } }
        }
        Fields::Unnamed(unnamed) => {
            let bindings = (0..unnamed.unnamed.len())
                .map(|i| Ident::new(&format!("__f{i}"), proc_macro2::Span::call_site()));
            quote! { Self::#variant(#(#bindings),*) }
        }
        Fields::Unit => quote! { Self::#variant },
    }
}

fn container(name: &Ident, variant: Option<&Ident>, fields: &Fields, mapped: TokenStream) -> TokenStream {
    let ty = quote! { #name::<U> };
    match (variant, fields) {
        (None, Fields::Named(_)) => quote! { #ty { #mapped } },
        (None, Fields::Unnamed(_)) => quote! { #ty(#mapped) },
        (None, Fields::Unit) => ty,
        (Some(v), Fields::Named(_)) => quote! { #ty::#v { #mapped } },
        (Some(v), Fields::Unnamed(_)) => quote! { #ty::#v(#mapped) },
        (Some(v), Fields::Unit) => quote! { #ty::#v },
    }
}

fn visit_fields(fields: &Fields, leaf: &Ident, from_self: bool) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let visited = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &self.#ident }
                } else {
                    quote! { #ident }
                };
                visit_field(value, &field.ty, leaf)
            });
            quote! { #(#visited;)* }
        }
        Fields::Unnamed(unnamed) => {
            let visited = unnamed.unnamed.iter().enumerate().map(|(i, field)| {
                let value = if from_self {
                    let index = Index::from(i);
                    quote! { &self.#index }
                } else {
                    let binding = Ident::new(&format!("__f{i}"), proc_macro2::Span::call_site());
                    quote! { #binding }
                };
                visit_field(value, &field.ty, leaf)
            });
            quote! { #(#visited;)* }
        }
        Fields::Unit => TokenStream::new(),
    }
}

fn tree_impl(
    name: &Ident,
    leaf: &Ident,
    impl_generics: &syn::ImplGenerics<'_>,
    ty_generics: &syn::TypeGenerics<'_>,
    where_clause: Option<&syn::WhereClause>,
    map_body: TokenStream,
    visit_body: TokenStream,
) -> TokenStream {
    quote! {
        impl #impl_generics resin_core::Tree<#leaf> for #name #ty_generics #where_clause {
            type Mapped<U: Clone> = #name::<U>;

            fn map<U: Clone>(&self, f: impl Fn(&#leaf) -> U) -> #name::<U> {
                #map_body
            }

            fn for_each_leaf(&self, mut f: impl FnMut(&#leaf)) {
                #visit_body
            }
        }
    }
}

fn derive_struct(item: ItemStruct) -> TokenStream {
    let (leaf, generics) = match leaf_generics(&item.generics, item.ident.span()) {
        Ok(g) => g,
        Err(e) => return e,
    };
    let (impl_g, ty_g, where_g) = generics.split_for_impl();
    let name = &item.ident;
    let map_body = container(name, None, &item.fields, map_fields(&item.fields, &leaf, true));
    let visit_body = visit_fields(&item.fields, &leaf, true);
    tree_impl(name, &leaf, &impl_g, &ty_g, where_g, map_body, visit_body)
}

fn derive_enum(item: ItemEnum) -> TokenStream {
    let (leaf, generics) = match leaf_generics(&item.generics, item.ident.span()) {
        Ok(g) => g,
        Err(e) => return e,
    };
    let (impl_g, ty_g, where_g) = generics.split_for_impl();
    let name = &item.ident;

    let map_arms = item.variants.iter().map(|variant| {
        let v = &variant.ident;
        let pattern = match_pattern(v, &variant.fields);
        let mapped = map_fields(&variant.fields, &leaf, false);
        let expr = container(name, Some(v), &variant.fields, mapped);
        quote! { #pattern => #expr }
    });
    let visit_arms = item.variants.iter().map(|variant| {
        let v = &variant.ident;
        let pattern = match_pattern(v, &variant.fields);
        let visited = visit_fields(&variant.fields, &leaf, false);
        quote! { #pattern => { #visited } }
    });

    let map_body = quote! { match self { #(#map_arms,)* } };
    let visit_body = quote! { match self { #(#visit_arms,)* } };
    tree_impl(name, &leaf, &impl_g, &ty_g, where_g, map_body, visit_body)
}
