(function_definition name: (lid) @name) @item
(source_file (const_declaration "const" @context (const_spec name: (lid) @name)) @item)
(foreign_function "extern" @context name: (lid) @name) @item
(foreign_type "extern" @context "type" @context name: (uid) @name) @item
(source_file (type_definition "type" @context (type_define name: (uid) @name)) @item)
(source_file (struct_definition "struct" @context name: (uid) @name) @item)

(intrinsic_function "intrinsic" @context name: (lid) @name) @item
