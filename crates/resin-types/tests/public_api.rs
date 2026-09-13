//! A client defines, checks, and lays out concrete types without a frontend.
use resin_types::prelude::*;

#[test]
fn recursive_records_share_one_table_for_conversions_and_layout() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Node");
    let node = Ty::Defined { definition };
    let body = Ty::Record {
        fields: vec![
            RecordField {
                name: "value".into(),
                ty: Ty::UInt32,
            },
            RecordField {
                name: "next".into(),
                ty: Ty::Pointer {
                    pointee: Box::new(node.clone()),
                },
            },
        ],
    };
    context.define_type(definition, body.clone()).unwrap();
    assert_eq!(
        context.ascribe(&body, &node).unwrap(),
        [Conv::Wrap { definition }]
    );

    let mut table = context.into_definitions().unwrap();
    assert_eq!(table.intern(&node), definition);
    assert_ne!(table.intern(&body), definition);
    let layout = resin_types::layout::layout(&table, &node).unwrap();
    assert_eq!(
        (layout.size, layout.align, layout.offsets),
        (16, 8, vec![0, 8])
    );
    assert_eq!(resin_types::format_type(&node, &table), "Node");
}

#[test]
fn explicit_owner_unwrapping_is_rejected_without_forbidding_field_projection() {
    let definition = TypeId::from_index(0);
    let owner = Ty::Defined { definition };
    let body = Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty: Ty::UInt32,
        }],
    };
    let context = TyperContext::from_definitions(vec![TypeDef::Nominal {
        name: "Owner".into(),
        body: Some(body.clone()),
        drop: Some(FunctionId::from_index(0)),
    }]);
    assert!(
        matches!(context.explicit_conversion(&owner, &body).unwrap_err().kind,
        TypeErrorKind::UnwrapManaged { definition: found } if found == definition)
    );
    assert_eq!(context.type_field(&owner, "value").unwrap().ty, Ty::UInt32);
    assert!(context.explicit_conversion(&body, &owner).is_ok());
}
