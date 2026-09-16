(lid) @variable
(uid) @type
(builtin_type) @type.builtin
(inferred_type) @type.builtin
["Ptr" "Ref" "Result" "None"] @type.builtin

; Keep declaration and control keywords in sync with the grammar's reserved words.
["export" "import" "extern" "intrinsic" "type" "struct" "def" "var" "const"] @keyword
["if" "else" "while" "match" "assert" "return" "break" "continue"] @keyword

(function_definition name: (lid) @function)
(foreign_function name: (lid) @function)
(intrinsic_function name: (lid) @function)
(const_spec name: (lid) @constant)
"sizeof" @function.builtin
((primary_term (lid) @constant.builtin) (#eq? @constant.builtin "iota"))
(function_definition params: (declare name: (lid) @variable.parameter))
(foreign_function params: (declare name: (lid) @variable.parameter))
(intrinsic_function params: (declare name: (lid) @variable.parameter))
(postfix_term prefix: (primary_term (lid) @function) . suffix: (arguments))
((primary_term (lid) @function.builtin)
  (#any-of? @function.builtin "ok" "err" "sizeof"))

(field_access name: (lid) @property)
(field_access name: (tuple_index) @property)
(method_call name: (lid) @function)
(record_term fields: (term_define name: (lid) @property))
(record_type field: (declare name: (lid) @property))
(struct_definition fields: (declare name: (lid) @property))
(match_arm name: (lid) @variable.parameter)
["ok" "err"] @function.builtin

(pointer_deref) @operator
(try_suffix) @operator
(number) @number
(string) @string
(comment) @comment

["=" ":=" "->" "=>" "||" "&&" "|" "^" "&" "==" "!=" "<" "<=" ">" ">="
 "<<" ">>" "+" "-" "*" "/" "%" "!" "~"] @operator
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
[";" "," ":" "." "::"] @punctuation.delimiter
(unary_type "<" @punctuation.bracket ">" @punctuation.bracket)


(decorator "@" @attribute name: (lid) @attribute)

(type_parameters "<" @punctuation.bracket ">" @punctuation.bracket)
(type_arguments "<" @punctuation.bracket ">" @punctuation.bracket)
(postfix_term prefix: (primary_term (lid) @function) . suffix: (type_application))

(boolean) @constant.builtin
