use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    Fields, Generics, Ident, ItemEnum, ItemStruct, Type, TypeParamBound, Variant, parse_quote,
};

pub fn main(item: TokenStream) -> TokenStream {
    let item: syn::Item = match syn::parse(item.into()) {
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

enum FieldBinding {
    Named(Ident),
    Unnamed,
}

struct FieldNode<'a> {
    value: TokenStream,
    other: TokenStream,
    ty: &'a Type,
    binding: FieldBinding,
}

fn bound_is_trait(bound: &TypeParamBound, trait_name: &str) -> bool {
    matches!(
        bound,
        TypeParamBound::Trait(trait_bound) if trait_bound.path.is_ident(trait_name)
    )
}

fn type_param_has_trait_bound(generics: &Generics, param: &Ident, trait_name: &str) -> bool {
    let on_type_param = generics.type_params().any(|tp| {
        tp.ident == *param
            && tp
                .bounds
                .iter()
                .any(|bound| bound_is_trait(bound, trait_name))
    });
    if on_type_param {
        return true;
    }

    generics.where_clause.as_ref().is_some_and(|wc| {
        wc.predicates.iter().any(|pred| {
            let syn::WherePredicate::Type(pred) = pred else {
                return false;
            };
            let Type::Path(ty) = &pred.bounded_ty else {
                return false;
            };
            ty.path.get_ident() == Some(param)
                && pred
                    .bounds
                    .iter()
                    .any(|bound| bound_is_trait(bound, trait_name))
        })
    })
}

fn validate_tree_generics(
    generics: &Generics,
    spanned: &impl Spanned,
) -> Result<(Ident, Generics), TokenStream> {
    let type_params: Vec<_> = generics.type_params().collect();
    if type_params.len() != 1 {
        return Err(syn::Error::new(
            spanned.span(),
            "type must have exactly one type parameter to derive Tree",
        )
        .to_compile_error()
        .into());
    }

    let leaf_type = type_params[0].ident.clone();
    let mut generics = generics.clone();
    if !type_param_has_trait_bound(&generics, &leaf_type, "Clone") {
        generics
            .type_params_mut()
            .next()
            .expect("exactly one type parameter")
            .bounds
            .push(parse_quote!(Clone));
    }

    Ok((leaf_type, generics))
}

fn is_leaf_type(ty: &Type, leaf_type: &Ident) -> bool {
    match ty {
        Type::Path(path) => path
            .path
            .get_ident()
            .is_some_and(|ident| ident == leaf_type),
        _ => false,
    }
}

fn struct_field_nodes(fields: &Fields) -> Vec<FieldNode<'_>> {
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                FieldNode {
                    value: quote! { &self.#ident },
                    other: quote! { &other.#ident },
                    ty: &field.ty,
                    binding: FieldBinding::Named(ident.clone()),
                }
            })
            .collect(),
        Fields::Unnamed(unnamed) => unnamed
            .unnamed
            .iter()
            .enumerate()
            .map(|(index, field)| FieldNode {
                value: quote! { &self.#index },
                other: quote! { &other.#index },
                ty: &field.ty,
                binding: FieldBinding::Unnamed,
            })
            .collect(),
        Fields::Unit => Vec::new(),
    }
}

fn other_ident(ident: &Ident) -> Ident {
    Ident::new(&format!("__o_{ident}"), ident.span())
}

fn enum_variant_nodes(variant: &Variant) -> (TokenStream, Vec<FieldNode<'_>>) {
    let (left, _, nodes) = enum_variant_patterns(variant, None);
    (left, nodes)
}

fn enum_zip_patterns<'a>(
    variant: &'a Variant,
    enum_name: &Ident,
) -> (TokenStream, TokenStream, Vec<FieldNode<'a>>) {
    enum_variant_patterns(variant, Some(enum_name))
}

fn enum_variant_patterns<'a>(
    variant: &'a Variant,
    zip_rhs_type: Option<&Ident>,
) -> (TokenStream, TokenStream, Vec<FieldNode<'a>>) {
    let variant_ident = &variant.ident;
    let rhs = zip_rhs_type
        .map(|name| quote! { #name::<U> })
        .unwrap_or(quote! { Self });
    match &variant.fields {
        Fields::Named(named) => {
            let idents: Vec<_> = named
                .named
                .iter()
                .map(|field| field.ident.as_ref().expect("named field").clone())
                .collect();
            let other_idents: Vec<_> = idents.iter().map(other_ident).collect();
            let nodes = named.named.iter().zip(&idents).zip(&other_idents).map(
                |((field, ident), other)| FieldNode {
                    value: quote! { #ident },
                    other: quote! { #other },
                    ty: &field.ty,
                    binding: FieldBinding::Named(ident.clone()),
                },
            );
            let other_pattern_fields: Vec<_> = idents
                .iter()
                .zip(&other_idents)
                .map(|(name, other)| quote! { #name: #other })
                .collect();
            (
                quote! { Self::#variant_ident { #(#idents),* } },
                quote! { #rhs::#variant_ident { #(#other_pattern_fields),* } },
                nodes.collect(),
            )
        }
        Fields::Unnamed(unnamed) => {
            let bindings: Vec<_> = (0..unnamed.unnamed.len())
                .map(|index| Ident::new(&format!("__f{index}"), variant.ident.span()))
                .collect();
            let other_bindings: Vec<_> = (0..unnamed.unnamed.len())
                .map(|index| Ident::new(&format!("__o{index}"), variant.ident.span()))
                .collect();
            let nodes = unnamed
                .unnamed
                .iter()
                .zip(&bindings)
                .zip(&other_bindings)
                .map(|((field, ident), other)| FieldNode {
                    value: quote! { #ident },
                    other: quote! { #other },
                    ty: &field.ty,
                    binding: FieldBinding::Unnamed,
                });
            (
                quote! { Self::#variant_ident(#(#bindings),*) },
                quote! { #rhs::#variant_ident(#(#other_bindings),*) },
                nodes.collect(),
            )
        }
        Fields::Unit => (
            quote! { Self::#variant_ident },
            quote! { #rhs::#variant_ident },
            Vec::new(),
        ),
    }
}

fn emit_map_field(node: &FieldNode, leaf_type: &Ident) -> TokenStream {
    let value = &node.value;
    if is_leaf_type(node.ty, leaf_type) {
        quote! { f(#value) }
    } else {
        quote! { resin_core::Tree::map(#value, f) }
    }
}

fn emit_zip_field(node: &FieldNode, leaf_type: &Ident) -> TokenStream {
    let value = &node.value;
    let other = &node.other;
    if is_leaf_type(node.ty, leaf_type) {
        quote! { f(#value, #other) }
    } else {
        quote! { resin_core::Tree::zip(#value, #other, f) }
    }
}

fn emit_mapped_fields(nodes: &[FieldNode], leaf_type: &Ident, mode: MapZip) -> TokenStream {
    let fields: Vec<_> = nodes
        .iter()
        .map(|node| {
            let mapped = match mode {
                MapZip::Map => emit_map_field(node, leaf_type),
                MapZip::Zip => emit_zip_field(node, leaf_type),
            };
            match &node.binding {
                FieldBinding::Named(ident) => quote! { #ident: #mapped },
                FieldBinding::Unnamed => mapped,
            }
        })
        .collect();
    quote! { #(#fields,)* }
}

enum MapZip {
    Map,
    Zip,
}

fn emit_container_expr(
    type_ident: &Ident,
    mapped_param: &Ident,
    variant_ident: Option<&Ident>,
    fields: &Fields,
    mapped_fields: TokenStream,
) -> TokenStream {
    let container = quote! { #type_ident::<#mapped_param> };
    match (variant_ident, fields) {
        (None, Fields::Named(_)) => quote! { #container { #mapped_fields } },
        (None, Fields::Unnamed(_)) => quote! { #container(#mapped_fields) },
        (None, Fields::Unit) => container,
        (Some(variant), Fields::Named(_)) => quote! { #container::#variant { #mapped_fields } },
        (Some(variant), Fields::Unnamed(_)) => quote! { #container::#variant(#mapped_fields) },
        (Some(variant), Fields::Unit) => quote! { #container::#variant },
    }
}

fn emit_tree_impl(
    type_ident: &Ident,
    leaf_type: &Ident,
    impl_generics: &syn::ImplGenerics<'_>,
    ty_generics: &syn::TypeGenerics<'_>,
    where_clause: Option<&syn::WhereClause>,
    map_body: TokenStream,
    zip_body: TokenStream,
) -> TokenStream {
    quote! {
        impl #impl_generics resin_core::Tree<#leaf_type> for #type_ident #ty_generics #where_clause {
            type Mapped<U: Clone> = #type_ident::<U>;

            fn map<U: Clone>(
                &self,
                f: impl Fn(&#leaf_type) -> U,
            ) -> #type_ident::<U> {
                #map_body
            }

            fn zip<U: Clone, V: Clone>(
                &self,
                other: &#type_ident::<U>,
                f: impl Fn(&#leaf_type, &U) -> V,
            ) -> #type_ident::<V> {
                #zip_body
            }
        }
    }
}

fn derive_tree_struct(item_struct: ItemStruct) -> TokenStream {
    let (leaf_type, generics) = match validate_tree_generics(&item_struct.generics, &item_struct) {
        Ok(validated) => validated,
        Err(err) => return err,
    };

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let nodes = struct_field_nodes(&item_struct.fields);
    let map_fields = emit_mapped_fields(&nodes, &leaf_type, MapZip::Map);
    let zip_fields = emit_mapped_fields(&nodes, &leaf_type, MapZip::Zip);

    let struct_name = &item_struct.ident;
    let map_body = emit_container_expr(
        struct_name,
        &Ident::new("U", struct_name.span()),
        None,
        &item_struct.fields,
        map_fields,
    );
    let zip_body = emit_container_expr(
        struct_name,
        &Ident::new("V", struct_name.span()),
        None,
        &item_struct.fields,
        zip_fields,
    );

    emit_tree_impl(
        struct_name,
        &leaf_type,
        &impl_generics,
        &ty_generics,
        where_clause,
        map_body,
        zip_body,
    )
    .into()
}

fn derive_tree_enum(item_enum: ItemEnum) -> TokenStream {
    let (leaf_type, generics) = match validate_tree_generics(&item_enum.generics, &item_enum) {
        Ok(validated) => validated,
        Err(err) => return err,
    };

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let enum_name = &item_enum.ident;

    let map_arms: Vec<TokenStream> = item_enum
        .variants
        .iter()
        .map(|variant| {
            let (pattern, nodes) = enum_variant_nodes(variant);
            let variant_ident = &variant.ident;
            let mapped_fields = emit_mapped_fields(&nodes, &leaf_type, MapZip::Map);
            let mapped_variant = emit_container_expr(
                enum_name,
                &Ident::new("U", enum_name.span()),
                Some(variant_ident),
                &variant.fields,
                mapped_fields,
            );
            quote! { #pattern => #mapped_variant }
        })
        .collect();

    let zip_arms: Vec<TokenStream> = item_enum
        .variants
        .iter()
        .map(|variant| {
            let (pattern, other_pattern, nodes) = enum_zip_patterns(variant, enum_name);
            let variant_ident = &variant.ident;
            let zipped_fields = emit_mapped_fields(&nodes, &leaf_type, MapZip::Zip);
            let mapped_variant = emit_container_expr(
                enum_name,
                &Ident::new("V", enum_name.span()),
                Some(variant_ident),
                &variant.fields,
                zipped_fields,
            );
            quote! {
                (#pattern, #other_pattern) => #mapped_variant,
            }
        })
        .collect();

    let map_body = quote! {
        match self {
            #(#map_arms,)*
        }
    };
    let zip_body = quote! {
        match (self, other) {
            #(#zip_arms)*
            _ => panic!("Tree::zip: enum variant mismatch"),
        }
    };

    emit_tree_impl(
        enum_name,
        &leaf_type,
        &impl_generics,
        &ty_generics,
        where_clause,
        map_body,
        zip_body,
    )
    .into()
}
