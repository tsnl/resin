//! `#[derive(Tree)]` expansion for host module trees.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    parse_quote, Data, DataStruct, DeriveInput, Field, Fields, GenericParam, Generics, Ident, Type,
};

pub fn expand(input: &DeriveInput) -> TokenStream2 {
    let crt = crate_path(input);
    let name = &input.ident;

    let Data::Struct(DataStruct {
        fields: Fields::Named(named),
        ..
    }) = &input.data
    else {
        return syn::Error::new_spanned(
            &input.ident,
            "Tree derive supports only structs with named fields",
        )
        .to_compile_error();
    };

    let (leaf_ty, _leaf_param) = match infer_leaf_type(&input.generics) {
        Ok(pair) => pair,
        Err(e) => return e.to_compile_error(),
    };

    let mut field_specs = Vec::new();
    for field in &named.named {
        match classify_field(field, &leaf_ty, &crt) {
            Ok(spec) => field_specs.push(spec),
            Err(e) => return e.to_compile_error(),
        }
    }

    let mut generics = input.generics.clone();
    for spec in &field_specs {
        if let Some(pred) = &spec.where_predicate {
            generics.make_where_clause().predicates.push(pred.clone());
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let flatten_iters: Vec<_> = field_specs
        .iter()
        .map(|f| {
            let ident = &f.ident;
            let field_name = f.ident.to_string();
            match f.kind {
                FieldKind::Leaf => quote! {
                    std::iter::once((#crt::path_name(#field_name), &self.#ident))
                },
                FieldKind::OptionalLeaf => quote! {
                    self.#ident.as_ref().into_iter().map(|leaf| {
                        (#crt::path_name(#field_name), leaf)
                    })
                },
                FieldKind::Nested => quote! {
                    self.#ident.flatten().map(|(path, leaf)| {
                        (#crt::prepend_name(#field_name, &path), leaf)
                    })
                },
                FieldKind::VecNested => quote! {
                    self.#ident.iter().enumerate().flat_map(|(i, child)| {
                        child.flatten().map(move |(rest, leaf)| {
                            (#crt::prepend_name_index(#field_name, i, &rest), leaf)
                        })
                    })
                },
            }
        })
        .collect();

    let flatten_body = match flatten_iters.as_slice() {
        [] => quote! { std::iter::empty() },
        [one] => quote! { #one },
        [first, rest @ ..] => quote! { #first #(.chain(#rest))* },
    };

    let unflatten_path_fields: Vec<_> = field_specs
        .iter()
        .map(|f| {
            let ident = &f.ident;
            let field_name = f.ident.to_string();
            match f.kind {
                FieldKind::Leaf => quote! {
                    #ident: #crt::take_next_leaf(leaves, #field_name)
                },
                FieldKind::OptionalLeaf => quote! {
                    #ident: #crt::take_optional_leaf(leaves, #field_name)
                },
                FieldKind::Nested => {
                    let ty = &f.ty;
                    quote! {
                        #ident: {
                            let mut child = #crt::take_named_children(leaves, #field_name);
                            let value = <#ty as #crt::Tree>::consume_unflatten(&mut child);
                            child.assert_consumed(concat!(
                                "field `",
                                stringify!(#ident),
                                "` child"
                            ));
                            value
                        }
                    }
                }
                FieldKind::VecNested => {
                    let inner = f.vec_inner_ty.as_ref().expect("vec inner type");
                    quote! {
                        #ident: {
                            let grouped = #crt::take_named_indexed_children(leaves, #field_name);
                            grouped
                                .into_iter()
                                .map(|mut group| {
                                    let item =
                                        <#inner as #crt::Tree>::consume_unflatten(&mut group);
                                    group.assert_consumed(concat!(
                                        "field `",
                                        stringify!(#ident),
                                        "` element"
                                    ));
                                    item
                                })
                                .collect()
                        }
                    }
                }
            }
        })
        .collect();

    quote! {
        impl #impl_generics #crt::Tree for #name #ty_generics #where_clause {
            type Leaf = #leaf_ty;
            type Map<U> = #name<U>;

            fn flatten(
                &self,
            ) -> impl Iterator<Item = (#crt::TreePath, &Self::Leaf)> + '_ {
                #flatten_body
            }

            fn consume_unflatten<I>(
                leaves: &mut #crt::TreeLeaves<Self::Leaf, I>,
            ) -> Self
            where
                I: Iterator<Item = (#crt::TreePath, Self::Leaf)>,
            {
                #name {
                    #(#unflatten_path_fields,)*
                }
            }
        }
    }
}

fn crate_path(input: &DeriveInput) -> TokenStream2 {
    for attr in &input.attrs {
        if !attr.path().is_ident("tree") {
            continue;
        }
        let Ok(syn::Meta::NameValue(nv)) = attr.parse_args() else {
            continue;
        };
        if !nv.path.is_ident("crate") {
            continue;
        }
        let syn::Expr::Path(path) = nv.value else {
            continue;
        };
        return quote!(::#path);
    }
    quote!(::resin_core)
}

struct FieldSpec {
    ident: Ident,
    ty: Type,
    kind: FieldKind,
    vec_inner_ty: Option<Type>,
    where_predicate: Option<syn::WherePredicate>,
}

#[derive(Clone, Copy)]
enum FieldKind {
    Leaf,
    OptionalLeaf,
    Nested,
    VecNested,
}

fn infer_leaf_type(generics: &Generics) -> Result<(Type, Ident), syn::Error> {
    let type_params: Vec<_> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(tp) => Some(tp.ident.clone()),
            _ => None,
        })
        .collect();

    match type_params.as_slice() {
        [] => Err(syn::Error::new(
            generics.span(),
            "Tree derive needs a type parameter for the leaf type (e.g. struct Foo<T> { ... })",
        )),
        [ident] => Ok((parse_quote!(#ident), ident.clone())),
        _ => Err(syn::Error::new(
            generics.span(),
            "Tree derive supports only a single type parameter as the leaf type",
        )),
    }
}

fn classify_field(
    field: &Field,
    leaf_ty: &Type,
    crt: &TokenStream2,
) -> Result<FieldSpec, syn::Error> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?;
    let ty = &field.ty;

    let force_leaf = field.attrs.iter().any(|attr| {
        attr.path().is_ident("tree")
            && attr
                .parse_args::<Ident>()
                .map(|i| i == "leaf")
                .unwrap_or(false)
    });

    let kind = if force_leaf || types_equal(ty, leaf_ty) {
        FieldKind::Leaf
    } else if let Some(inner) = extract_option_inner(ty) {
        if types_equal(inner, leaf_ty) {
            FieldKind::OptionalLeaf
        } else {
            FieldKind::Nested
        }
    } else if extract_vec_inner(ty).is_some() {
        FieldKind::VecNested
    } else {
        FieldKind::Nested
    };

    let vec_inner_ty = match kind {
        FieldKind::VecNested => extract_vec_inner(ty).cloned(),
        _ => None,
    };

    let where_predicate = match kind {
        FieldKind::Nested => {
            let field_ty = ty;
            Some(parse_quote! {
                #field_ty: #crt::Tree<Leaf = #leaf_ty>
            })
        }
        FieldKind::VecNested => {
            let inner = vec_inner_ty.as_ref().expect("vec inner");
            Some(parse_quote! {
                #inner: #crt::Tree<Leaf = #leaf_ty>
            })
        }
        _ => None,
    };

    Ok(FieldSpec {
        ident,
        ty: ty.clone(),
        kind,
        vec_inner_ty,
        where_predicate,
    })
}

fn types_equal(a: &Type, b: &Type) -> bool {
    quote::quote!(#a).to_string() == quote::quote!(#b).to_string()
}

fn extract_option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(tp) = ty else {
        return None;
    };
    let seg = tp.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    }
}

fn extract_vec_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(tp) = ty else {
        return None;
    };
    let seg = tp.path.segments.last()?;
    if seg.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    }
}
