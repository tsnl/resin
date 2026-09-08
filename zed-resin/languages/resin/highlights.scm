(lid) @variable
(uid) @type
(builtin_type) @type.builtin
(inferred_type) @type.builtin
["Ptr" "Span" "Result" "Arc" "Weak" "None"] @type.builtin

; Keep declaration and control keywords in sync with the grammar's reserved words.
["export" "import" "extern" "type" "struct" "impl" "def" "var"] @keyword
["if" "else" "while" "match"] @keyword

(function_definition name: (lid) @function)
(foreign_function name: (lid) @function)
(function_definition params: (declare name: (lid) @variable.parameter))
(foreign_function params: (declare name: (lid) @variable.parameter))
(postfix_term prefix: (primary_term (lid) @function) . suffix: (closed_term))
((primary_term (lid) @function.builtin)
  (#any-of? @function.builtin "print" "ok" "err" "replace"))

(field_access name: (lid) @property)
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
[";" "," ":" "."] @punctuation.delimiter
(unary_type "<" @punctuation.bracket ">" @punctuation.bracket)


(decorator "@" @attribute name: (lid) @attribute)
