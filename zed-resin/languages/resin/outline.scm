(function_definition name: (lid) @name) @item
(foreign_function "extern" @context name: (lid) @name) @item
(foreign_type "extern" @context "type" @context name: (uid) @name) @item
(source_file (type_definition "type" @context (type_define name: (uid) @name)) @item)
(source_file (struct_definition "struct" @context name: (uid) @name) @item)

(impl_definition "impl" @context owner: (uid) @name) @item
