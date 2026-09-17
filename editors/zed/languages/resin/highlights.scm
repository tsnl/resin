(lid) @variable
(uid) @type
(builtin_type) @type.builtin
(inferred_type) @type.builtin
["Ptr" "Ref" "Err" "None"] @type.builtin

; Keep declaration and control keywords in sync with the grammar's reserved words.
["export" "import" "extern" "intrinsic" "type" "struct" "fn" "let" "mut" "const"] @keyword
["if" "else" "while" "match" "assert" "return" "break" "continue"] @keyword

(function_definition name: (lid) @function)
(foreign_function name: (lid) @function)
(intrinsic_function name: (lid) @function)
(const_spec name: (lid) @constant)
"sizeof" @function.builtin
((primary_term (lid) @constant.builtin) (#eq? @constant.builtin "iota"))
(function_definition params: (parameter pattern: (binding_pattern name: (lid) @variable.parameter)))
(foreign_function params: (parameter pattern: (binding_pattern name: (lid) @variable.parameter)))
(intrinsic_function params: (parameter pattern: (binding_pattern name: (lid) @variable.parameter)))
(postfix_term prefix: (primary_term (lid) @function) . suffix: (arguments))
((primary_term (lid) @function.builtin)
  (#any-of? @function.builtin "sizeof"))

(field_access name: (lid) @property)
(field_access name: (tuple_index) @property)
(method_call name: (lid) @function)
(record_term fields: (term_define name: (lid) @property))
(struct_definition fields: (declare name: (lid) @property))
(match_arm pattern: (binding_pattern name: (lid) @variable.parameter))

(pointer_deref) @operator
(try_suffix) @operator
(number) @number
(string) @string
(comment) @comment
(doc_comment) @comment.documentation

["=" "->" "=>" "||" "&&" "|" "^" "&" "==" "!=" "<" "<=" ">" ">="
 "<<" ">>" "+" "-" "*" "/" "%" "!" "~"] @operator
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
[";" "," ":" "." "::"] @punctuation.delimiter
(unary_type "<" @punctuation.bracket ">" @punctuation.bracket)


(decorator "@" @attribute name: (lid) @attribute)

(type_parameters "<" @punctuation.bracket ">" @punctuation.bracket)
(type_arguments "<" @punctuation.bracket ">" @punctuation.bracket)
(postfix_term prefix: (primary_term (lid) @function) . suffix: (type_application))

(boolean) @constant.builtin
