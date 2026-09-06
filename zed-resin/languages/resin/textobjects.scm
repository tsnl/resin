(function_definition
  body: (block_body "{" (_)* @function.inside "}")) @function.around
(foreign_function) @function.around
(comment)+ @comment.around
