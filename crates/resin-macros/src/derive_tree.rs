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

fn is_associated_leaf(ty: &Type, leaf: &Ident, associated: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    let segments = &path.path.segments;
    segments.len() == 2
        && segments[0].ident == *leaf
        && segments[1].ident == associated
}

fn uses_associated_leaf(fields: &Fields, leaf: &Ident, associated: &str) -> bool {
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .any(|field| is_associated_leaf(&field.ty, leaf, associated)),
        Fields::Unnamed(unnamed) => unnamed
            .unnamed
            .iter()
            .any(|field| is_associated_leaf(&field.ty, leaf, associated)),
        Fields::Unit => false,
    }
}

fn is_field_leaf(ty: &Type, leaf: &Ident, associated: Option<&str>) -> bool {
    match associated {
        Some(name) if is_associated_leaf(ty, leaf, name) => true,
        Some(_) => false,
        None => is_leaf(ty, leaf),
    }
}

fn map_field(value: TokenStream, ty: &Type, leaf: &Ident, associated: Option<&str>) -> TokenStream {
    if is_field_leaf(ty, leaf, associated) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::map(#value, |leaf| f(leaf)) }
    }
}

fn try_map_field(value: TokenStream, ty: &Type, leaf: &Ident, associated: Option<&str>) -> TokenStream {
    if is_field_leaf(ty, leaf, associated) {
        quote! { f(#value)? }
    } else {
        quote! { resin_core::Tree::try_map(#value, |leaf| f(leaf))?
        }
    }
}

fn visit_field(value: TokenStream, ty: &Type, leaf: &Ident, associated: Option<&str>) -> TokenStream {
    if is_field_leaf(ty, leaf, associated) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::for_each_leaf(#value, |leaf| f(leaf)) }
    }
}

fn visit_mut_field(value: TokenStream, ty: &Type, leaf: &Ident, associated: Option<&str>) -> TokenStream {
    if is_field_leaf(ty, leaf, associated) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::for_each_leaf_mut(#value, |leaf| f(leaf)) }
    }
}

fn try_map_fields(fields: &Fields, leaf: &Ident, from_self: bool, associated: Option<&str>) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let mapped = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &self.#ident }
                } else {
                    quote! { #ident }
                };
                let mapped = try_map_field(value, &field.ty, leaf, associated);
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
                try_map_field(value, &field.ty, leaf, associated)
            });
            quote! { #(#mapped,)* }
        }
        Fields::Unit => TokenStream::new(),
    }
}

fn map_fields(fields: &Fields, leaf: &Ident, from_self: bool, associated: Option<&str>) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let mapped = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &self.#ident }
                } else {
                    quote! { #ident }
                };
                let mapped = map_field(value, &field.ty, leaf, associated);
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
                map_field(value, &field.ty, leaf, associated)
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

fn visit_mut_fields(fields: &Fields, leaf: &Ident, from_self: bool, associated: Option<&str>) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let visited = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &mut self.#ident }
                } else {
                    quote! { #ident }
                };
                visit_mut_field(value, &field.ty, leaf, associated)
            });
            quote! { #(#visited;)* }
        }
        Fields::Unnamed(unnamed) => {
            let visited = unnamed.unnamed.iter().enumerate().map(|(i, field)| {
                let value = if from_self {
                    let index = Index::from(i);
                    quote! { &mut self.#index }
                } else {
                    let binding = Ident::new(&format!("__f{i}"), proc_macro2::Span::call_site());
                    quote! { #binding }
                };
                visit_mut_field(value, &field.ty, leaf, associated)
            });
            quote! { #(#visited;)* }
        }
        Fields::Unit => TokenStream::new(),
    }
}

fn visit_fields(fields: &Fields, leaf: &Ident, from_self: bool, associated: Option<&str>) -> TokenStream {
    match fields {
        Fields::Named(named) => {
            let visited = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = if from_self {
                    quote! { &self.#ident }
                } else {
                    quote! { #ident }
                };
                visit_field(value, &field.ty, leaf, associated)
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
                visit_field(value, &field.ty, leaf, associated)
            });
            quote! { #(#visited;)* }
        }
        Fields::Unit => TokenStream::new(),
    }
}

fn tree_impl(
    name: &Ident,
    leaf_ty: TokenStream,
    mapped_ty: TokenStream,
    impl_generics: &syn::ImplGenerics<'_>,
    ty_generics: &syn::TypeGenerics<'_>,
    where_clause: Option<&syn::WhereClause>,
    map_body: TokenStream,
    try_map_body: TokenStream,
    visit_body: TokenStream,
    visit_mut_body: TokenStream,
) -> TokenStream {
    quote! {
        impl #impl_generics resin_core::Tree<#leaf_ty> for #name #ty_generics #where_clause {
            type Mapped<U: Clone> = #mapped_ty;

            fn map<U: Clone>(&self, f: impl Fn(&#leaf_ty) -> U) -> #mapped_ty {
                #map_body
            }

            fn try_map<U: Clone, E>(
                &self,
                mut f: impl FnMut(&#leaf_ty) -> Result<U, E>,
            ) -> Result<#mapped_ty, E> {
                #try_map_body
            }

            fn for_each_leaf(&self, mut f: impl FnMut(&#leaf_ty)) {
                #visit_body
            }

            fn for_each_leaf_mut(&mut self, mut f: impl FnMut(&mut #leaf_ty)) {
                #visit_mut_body
            }
        }
    }
}

fn mapped_struct_name(name: &Ident) -> Ident {
    Ident::new(&format!("{name}Mapped"), name.span())
}

fn emit_mapped_struct(name: &Ident, fields: &Fields, leaf: &Ident, associated: Option<&str>) -> TokenStream {
    let mapped_name = mapped_struct_name(name);
    match fields {
        Fields::Named(named) => {
            let field_defs = named.named.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let ty = if is_field_leaf(&field.ty, leaf, associated) {
                    quote! { U }
                } else {
                    let ty = &field.ty;
                    quote! { <#ty as resin_core::Tree<#leaf::Tensor>>::Mapped<U> }
                };
                quote! { #ident: #ty }
            });
            quote! {
                #[doc(hidden)]
                #[derive(Clone)]
                struct #mapped_name<U: Clone> {
                    #(#field_defs,)*
                }
            }
        }
        Fields::Unnamed(unnamed) => {
            let field_defs = unnamed.unnamed.iter().enumerate().map(|(i, field)| {
                let index = Index::from(i);
                let ty = if is_field_leaf(&field.ty, leaf, associated) {
                    quote! { U }
                } else {
                    let ty = &field.ty;
                    quote! { <#ty as resin_core::Tree<#leaf::Tensor>>::Mapped<U> }
                };
                quote! { #index: #ty }
            });
            quote! {
                #[doc(hidden)]
                #[derive(Clone)]
                struct #mapped_name<U: Clone>(#(#field_defs,)*);
            }
        }
        Fields::Unit => quote! {
            #[doc(hidden)]
            #[derive(Clone)]
            struct #mapped_name<U: Clone>;
        },
    }
}

fn derive_struct(item: ItemStruct) -> TokenStream {
    let (leaf, generics) = match leaf_generics(&item.generics, item.ident.span()) {
        Ok(g) => g,
        Err(e) => return e,
    };
    let (impl_g, ty_g, where_g) = generics.split_for_impl();
    let name = &item.ident;

    if uses_associated_leaf(&item.fields, &leaf, "Tensor") {
        let mapped_name = mapped_struct_name(name);
        let leaf_ty = quote! { #leaf::Tensor };
        let mapped_ty = quote! { #mapped_name<U> };
        let mapped_struct = emit_mapped_struct(name, &item.fields, &leaf, Some("Tensor"));
        let extra_where = parse_quote! { #leaf::Tensor: Clone };
        let where_clause = match where_g.cloned() {
            Some(mut wc) => {
                wc.predicates.push(extra_where);
                Some(wc)
            }
            None => Some(parse_quote! { where #extra_where }),
        };
        let mapped_fields = map_fields(&item.fields, &leaf, true, Some("Tensor"));
        let try_mapped_fields = try_map_fields(&item.fields, &leaf, true, Some("Tensor"));
        let map_body = quote! {
            #mapped_name {
                #mapped_fields
            }
        };
        let try_map_body = quote! {
            Ok(#mapped_name {
                #try_mapped_fields
            })
        };
        let visit_body = visit_fields(&item.fields, &leaf, true, Some("Tensor"));
        let visit_mut_body = visit_mut_fields(&item.fields, &leaf, true, Some("Tensor"));
        let mapped_visit_body = visit_fields(&item.fields, &leaf, true, Some("Tensor"));
        let mapped_visit_mut_body = visit_mut_fields(&item.fields, &leaf, true, Some("Tensor"));
        let mapped_tree = quote! {
            impl<U: Clone> resin_core::Tree<U> for #mapped_name<U> {
                type Mapped<V: Clone> = #mapped_name<V>;

                fn map<V: Clone>(&self, f: impl Fn(&U) -> V) -> #mapped_name<V> {
                    #map_body
                }

                fn try_map<V: Clone, E>(
                    &self,
                    mut f: impl FnMut(&U) -> Result<V, E>,
                ) -> Result<#mapped_name<V>, E> {
                    #try_map_body
                }

                fn for_each_leaf(&self, mut f: impl FnMut(&U)) {
                    #mapped_visit_body
                }

                fn for_each_leaf_mut(&mut self, mut f: impl FnMut(&mut U)) {
                    #mapped_visit_mut_body
                }
            }
        };
        let tree = tree_impl(
            name,
            leaf_ty,
            mapped_ty,
            &impl_g,
            &ty_g,
            where_clause.as_ref(),
            map_body,
            try_map_body,
            visit_body,
            visit_mut_body,
        );
        return quote! {
            #mapped_struct
            #mapped_tree
            #tree
        };
    }

    let map_body = container(name, None, &item.fields, map_fields(&item.fields, &leaf, true, None));
    let try_map_body = container(
        name,
        None,
        &item.fields,
        try_map_fields(&item.fields, &leaf, true, None),
    );
    let try_map_body = quote! { Ok(#try_map_body) };
    let visit_body = visit_fields(&item.fields, &leaf, true, None);
    let visit_mut_body = visit_mut_fields(&item.fields, &leaf, true, None);
    tree_impl(
        name,
        quote! { #leaf },
        quote! { #name::<U> },
        &impl_g,
        &ty_g,
        where_g,
        map_body,
        try_map_body,
        visit_body,
        visit_mut_body,
    )
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
        let mapped = map_fields(&variant.fields, &leaf, false, None);
        let expr = container(name, Some(v), &variant.fields, mapped);
        quote! { #pattern => #expr }
    });
    let visit_arms = item.variants.iter().map(|variant| {
        let v = &variant.ident;
        let pattern = match_pattern(v, &variant.fields);
        let visited = visit_fields(&variant.fields, &leaf, false, None);
        quote! { #pattern => { #visited } }
    });

    let map_body = quote! { match self { #(#map_arms,)* } };
    let try_map_arms = item.variants.iter().map(|variant| {
        let v = &variant.ident;
        let pattern = match_pattern(v, &variant.fields);
        let mapped = try_map_fields(&variant.fields, &leaf, false, None);
        let expr = container(name, Some(v), &variant.fields, mapped);
        quote! { #pattern => Ok(#expr) }
    });
    let try_map_body = quote! { match self { #(#try_map_arms,)* } };
    let visit_mut_arms = item.variants.iter().map(|variant| {
        let v = &variant.ident;
        let pattern = match_pattern(v, &variant.fields);
        let visited = visit_mut_fields(&variant.fields, &leaf, false, None);
        quote! { #pattern => { #visited } }
    });
    let visit_body = quote! { match self { #(#visit_arms,)* } };
    let visit_mut_body = quote! { match self { #(#visit_mut_arms,)* } };
    tree_impl(
        name,
        quote! { #leaf },
        quote! { #name::<U> },
        &impl_g,
        &ty_g,
        where_g,
        map_body,
        try_map_body,
        visit_body,
        visit_mut_body,
    )
}