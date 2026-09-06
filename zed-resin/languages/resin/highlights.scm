(lid) @variable
(uid) @type
(builtin_type) @type.builtin
(inferred_type) @type.builtin
["Ptr" "Span"] @type.builtin

["export" "import" "extern" "type" "def" "var" "if" "else" "while"] @keyword

(function_definition name: (lid) @function)
(foreign_function name: (lid) @function)
(function_definition params: (declare name: (lid) @variable.parameter))
(foreign_function params: (declare name: (lid) @variable.parameter))
(postfix_term prefix: (primary_term (lid) @function) . suffix: (closed_term))
((primary_term (lid) @function.builtin)
  (#any-of? @function.builtin "print" "shader"))

(field_access name: (lid) @property)
(record_term fields: (term_define name: (lid) @property))
(record_type field: (declare name: (lid) @property))

(pointer_deref) @operator
(number) @number
(string) @string
(comment) @comment

["=" ":=" "->" "||" "&&" "|" "^" "&" "==" "!=" "<" "<=" ">" ">="
 "<<" ">>" "+" "-" "*" "/" "%" "!" "~"] @operator
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
[";" "," ":" "."] @punctuation.delimiter
(unary_type "<" @punctuation.bracket ">" @punctuation.bracket)
