(function_definition name: (lid) @name) @item
(foreign_function "extern" @context name: (lid) @name) @item
(foreign_type "extern" @context "type" @context name: (uid) @name) @item
(source_file (statement (define (type_define name: (uid) @name))) @item)
(source_file (statement (define (term_define name: (lid) @name))) @item)
(source_file (statement (declare name: (lid) @name)) @item)
