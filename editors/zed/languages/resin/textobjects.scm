(function_definition
  body: (block_body "{" (_)* @function.inside "}")) @function.around
(foreign_function) @function.around
(struct_definition "{" (_)* @class.inside "}") @class.around
(comment)+ @comment.around
