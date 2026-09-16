/**
 * @file Resin grammar for tree-sitter
 * @author Nikhil Idiculla <nikhil.idiculla@gmail.com>
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl.d.ts" />
// @ts-check

/**
 * @param {string} field_name - name of the field to accumulate elements into
 * @param {RuleOrLiteral} element - rule for each element in the list
 * @param {RuleOrLiteral} sep - separator between elements
 * @returns {Rule}
 */
function list(field_name, element, sep) {
  return seq(
    repeat(seq(field(field_name, element), sep)),
    optional(field(field_name, element)),
  );
}

/**
 * @param {string} field_name - name of the field to accumulate elements into
 * @param {RuleOrLiteral} element - rule for each element in the list
 * @param {RuleOrLiteral} sep - separator between elements
 * @returns {Rule}
 */
function list1(field_name, element, sep) {
  return seq(
    field(field_name, element),
    optional(seq(sep, list(field_name, element, sep))),
  );
}

// C operator precedence: higher binds tighter.
const BOP_PREC = {
  OR: 1,
  AND: 2,
  BIT_OR: 3,
  BIT_XOR: 4,
  BIT_AND: 5,
  EQ: 6,
  REL: 7,
  SHIFT: 8,
  ADD: 9,
  MUL: 10,
};

const BUILTIN_TYPES = [
  "str",
  "bool",
  "sbyte",
  "short",
  "int",
  "long",
  "ubyte",
  "ushort",
  "uint",
  "ulong",
  "float32",
  "float64",
  "Never",
  "None",
  "GpuArguments",
  "StrongOwner",
  "WeakOwner",
  "GpuView",
  "GpuPipelineContract",
];

const TYPE_FORMERS = ["Ptr", "Ref", "Result"];

/**
 * Tree-sitter reserves words for only one token; uppercase names need an exclusion too.
 * @param {string} initial - character class for the first identifier character
 * @param {readonly string[]} words - reserved words to exclude
 * @param {boolean} [allowEmpty=false] - allow the empty suffix during recursion
 * @returns {string} Rust regex matching identifiers except the reserved words
 */
function identifierExcept(initial, words, allowEmpty = false) {
  const heads = [
    ...new Set(words.filter(Boolean).map((word) => word.charAt(0))),
  ];
  const branches = heads.map(
    (head) =>
      head +
      identifierExcept(
        "a-zA-Z0-9_",
        words
          .filter((word) => word.startsWith(head))
          .map((word) => word.slice(1)),
        true,
      ),
  );
  const other = heads.length
    ? `[${initial}&&[^${heads.join("")}]]`
    : `[${initial}]`;
  branches.push(`${other}[a-zA-Z0-9_]*`);
  if (allowEmpty && !words.includes("")) branches.push("");
  return `(?:${branches.join("|")})`;
}

export default grammar({
  name: "resin",

  word: ($) => $.lid,
  reserved: {
    global: () => [
      "export",
      "import",
      "extern",
      "intrinsic",
      "type",
      "struct",
      "def",
      "var",
      "const",
      "sizeof",
      "if",
      "else",
      "while",
      "match",
      "true",
      "false",
      ...BUILTIN_TYPES,
      ...TYPE_FORMERS,
    ],
  },

  conflicts: ($) => [
    [$.primary_term, $.union_type],
    [$.unit_term, $.unit_type],
    [$.parameter_types, $.tuple_type],
  ],

  rules: {
    //
    // Source file
    //

    source_file: ($) =>
      seq(
        optional(field("exports", $.export_clause)),
        optional(field("externs", $.extern_clause)),
        optional(field("imports", $.import_clause)),
        repeat(
          field(
            "stmt",
            choice(
              $.function_definition,
              $.intrinsic_function,
              $.foreign_type,
              $.type_definition,
              $.struct_definition,
              $.const_declaration,
            ),
          ),
        ),
      ),

    export_clause: ($) =>
      seq("export", "{", list("name", choice($.lid, $.uid), ","), "}", ";"),
    import_clause: ($) =>
      seq("import", "{", list("path", $.string, ","), "}", ";"),
    extern_clause: ($) =>
      seq("extern", "{", list("groups", $.foreign_group, ","), "}", ";"),
    foreign_group: ($) =>
      seq(
        field("header", $.string),
        ":",
        "{",
        repeat(field("functions", $.foreign_function)),
        "}",
      ),

    foreign_function: ($) =>
      seq(
        "def",
        field("name", $.lid),
        "(",
        list("params", $.declare, ","),
        ")",
        optional(seq("->", field("result", $.type))),
        ";",
      ),
    intrinsic_function: ($) =>
      seq(
        "intrinsic",
        field("operation", $.string),
        "def",
        field("name", $.lid),
        optional(field("type_params", $.type_parameters)),
        "(",
        list("params", $.declare, ","),
        ")",
        "->",
        field("result", $.type),
        ";",
      ),
    foreign_type: ($) => seq("extern", "type", field("name", $.uid), ";"),

    type_definition: ($) =>
      seq("type", field("definition", $.type_define), ";"),
    struct_definition: ($) =>
      seq(
        "struct",
        field("name", $.uid),
        optional(field("type_params", $.type_parameters)),
        "{",
        list("fields", $.declare, ","),
        repeat(field("method", $.function_definition)),
        "}",
        ";",
      ),

    decorator: ($) => seq("@", field("name", $.lid)),

    function_definition: ($) =>
      seq(
        repeat(field("decorator", $.decorator)),
        "def",
        field("name", $.lid),
        optional(field("type_params", $.type_parameters)),
        "(",
        list("params", $.declare, ","),
        ")",
        optional(seq("->", field("result", $.type))),
        "=",
        field("body", $.block_body),
        ";",
      ),
    block_body: ($) =>
      seq(
        "{",
        repeat(field("stmt", $.statement)),
        optional(field("tail", $.term)),
        "}",
      ),

    //
    // Statement
    //

    statement: ($) =>
      choice(
        field("constant", $.const_declaration),
        field("struct", $.struct_definition),
        seq(field("define", $.define), ";"),
        seq("var", field("declare", $.declare), ";"),
        seq(field("expr", $.term), ";"),
      ),

    const_declaration: ($) =>
      seq(
        "const",
        choice(
          field("spec", $.const_spec),
          seq("(", repeat(seq(field("spec", $.const_spec), ";")), ")"),
        ),
        ";",
      ),
    const_spec: ($) =>
      seq(
        list1("name", choice($.lid, $.discard), ","),
        optional(seq(":", field("ann", $.type))),
        "=",
        list1("init", $.term, ","),
      ),
    discard: () => "_",

    define: ($) =>
      choice(
        seq("var", field("term", $.local_define)),
        seq("type", field("type", $.type_define)),
      ),
    local_define: ($) =>
      seq(
        field("name", $.lid),
        optional(seq(":", field("ann", $.type))),
        "=",
        field("init", $.term),
      ),
    term_define: ($) => seq(field("name", $.lid), "=", field("init", $.term)),
    type_define: ($) =>
      seq(
        field("name", $.uid),
        optional(field("type_params", $.type_parameters)),
        "=",
        field("init", $.type),
      ),
    type_parameters: ($) => seq("<", list1("params", $.uid, ","), ">"),
    type_arguments: ($) => seq("<", list1("args", $.type, ","), ">"),
    declare: ($) => seq(field("name", $.lid), ":", field("ann", $.type)),

    //
    // Term
    //

    term: ($) => $.assignment_term,

    assignment_term: ($) =>
      choice(
        prec.right(
          seq(
            field("place", $.binary_term),
            ":=",
            field("value", $.assignment_term),
          ),
        ),
        $.binary_term,
      ),

    binary_term: ($) =>
      choice(
        ...[
          { tok: "||", prec: BOP_PREC.OR },
          { tok: "&&", prec: BOP_PREC.AND },
          { tok: "|", prec: BOP_PREC.BIT_OR },
          { tok: "^", prec: BOP_PREC.BIT_XOR },
          { tok: "&", prec: BOP_PREC.BIT_AND },
          { tok: "==", prec: BOP_PREC.EQ },
          { tok: "!=", prec: BOP_PREC.EQ },
          { tok: "<", prec: BOP_PREC.REL },
          { tok: "<=", prec: BOP_PREC.REL },
          { tok: ">", prec: BOP_PREC.REL },
          { tok: ">=", prec: BOP_PREC.REL },
          { tok: "<<", prec: BOP_PREC.SHIFT },
          { tok: ">>", prec: BOP_PREC.SHIFT },
          { tok: "+", prec: BOP_PREC.ADD },
          { tok: "-", prec: BOP_PREC.ADD },
          { tok: "*", prec: BOP_PREC.MUL },
          { tok: "/", prec: BOP_PREC.MUL },
          { tok: "%", prec: BOP_PREC.MUL },
        ].map((it) =>
          prec.left(
            it.prec,
            seq(
              field("left", $.binary_term),
              field("operator", it.tok),
              field("right", $.binary_term),
            ),
          ),
        ),
        $.unary_term,
      ),

    unary_term: ($) =>
      choice(
        ...["!", "-", "+", "~", "&"].map((op) =>
          seq(field("operator", op), field("operand", $.unary_term)),
        ),
        $.postfix_term,
      ),

    postfix_term: ($) =>
      prec.right(
        seq(
          field("prefix", $.primary_term),
          repeat(
            field(
              "suffix",
              choice(
                $.arguments,
                $.field_access,
                $.method_call,
                $.type_application,
                $.pointer_deref,
                $.try_suffix,
                $.unwrap_suffix,
              ),
            ),
          ),
        ),
      ),
    field_access: ($) =>
      seq(
        ".",
        choice(
          field("name", $.tuple_index),
          seq(
            field("name", $.lid),
            optional(field("type_args", $.type_application)),
          ),
        ),
      ),
    tuple_index: () => token(/0|[1-9][0-9]*/),
    method_call: ($) =>
      prec(
        1,
        seq(
          ".",
          field("name", $.lid),
          optional(field("type_args", $.type_application)),
          field("args", $.arguments),
        ),
      ),
    type_application: ($) => seq("::", field("types", $.type_arguments)),
    pointer_deref: () => ".*",
    try_suffix: () => "?",
    unwrap_suffix: () => "!",

    arguments: ($) => seq("(", list("args", $.term, ","), ")"),
    constructor_term: ($) =>
      seq(
        field("type", $.unary_type),
        field(
          "value",
          choice($.record_term, alias($._empty_record, $.unit_term)),
        ),
      ),
    _empty_record: () => seq("{", "}"),

    closed_term: ($) =>
      choice(
        $.paren_term,
        $.tuple_term,
        $.array_term,
        $.record_term,
        $.chain_term,
        $.unit_term,
      ),
    paren_term: ($) => seq("(", field("inner", $.term), ")"),
    tuple_term: ($) =>
      seq(
        "(",
        field("elems", $.term),
        ",",
        repeat(seq(field("elems", $.term), ",")),
        optional(field("elems", $.term)),
        ")",
      ),
    array_term: ($) => seq("[", list("elems", $.term, ","), "]"),
    record_term: ($) => seq("{", list1("fields", $.term_define, ","), "}"),
    chain_term: ($) =>
      seq(
        "{",
        choice(
          seq(
            field("prefix", repeat1($.statement)),
            optional(field("tail", $.term)),
          ),
          field("tail", $.term),
        ),
        "}",
      ),
    unit_term: () => prec.dynamic(1, choice(seq("{", "}"), seq("(", ")"))),

    boolean: () => choice("true", "false"),

    primary_term: ($) =>
      choice(
        $.sizeof_term,
        $.constructor_term,
        $.closed_term,
        $.lid,
        $.number,
        $.boolean,
        $.string,
        $.if_term,
        $.while_term,
        $.match_term,
        $.unary_type,
      ),
    sizeof_term: ($) => seq("sizeof", "(", field("type", $.type), ")"),
    match_term: ($) =>
      seq(
        "match",
        "(",
        field("value", $.term),
        ")",
        "{",
        list1("arms", $.match_arm, ","),
        "}",
      ),
    match_arm: ($) =>
      seq(
        choice(
          seq(
            field("variant", choice($.type, "ok", "err")),
            "(",
            field("name", $.lid),
            ")",
          ),
          field("variant", "None"),
        ),
        "=>",
        field("body", $.block_body),
      ),
    while_term: ($) =>
      seq(
        "while",
        "(",
        field("cond", $.term),
        ")",
        field("body", $.block_body),
      ),
    if_term: ($) =>
      seq(
        "if",
        "(",
        field("cond", $.term),
        ")",
        field("then", $.closed_term),
        optional(seq("else", field("else", choice($.closed_term, $.if_term)))),
      ),

    //
    // Types
    //

    type: ($) => $.infix_type,

    infix_type: ($) =>
      choice(
        seq(
          field("params", $.parameter_types),
          "->",
          field("ret_ty", $.infix_type),
        ),
        $.union_type,
      ),

    parameter_types: ($) => seq("(", list("params", $.type, ","), ")"),

    union_type: ($) =>
      choice(
        prec.left(
          seq(field("left", $.union_type), "|", field("right", $.unary_type)),
        ),
        $.unary_type,
      ),

    unary_type: ($) =>
      choice(
        prec(1, seq(field("former", $.uid), field("args", $.type_arguments))),
        seq(
          field("former", choice("Ptr", "Ref")),
          "<",
          field("arg", $.type),
          ">",
        ),
        seq(
          "Result",
          "<",
          field("value", $.type),
          ",",
          field("error", $.type),
          ">",
        ),
        $.primary_type,
      ),

    primary_type: ($) =>
      choice(
        $.inferred_type,
        field("var", $.uid),
        field("builtin", $.builtin_type),
        $.closed_type,
      ),
    builtin_type: () => choice(...BUILTIN_TYPES),
    inferred_type: () => "_",

    closed_type: ($) =>
      choice($.paren_type, $.tuple_type, $.unit_type, $.record_type),
    paren_type: ($) => seq("(", field("inner", $.type), ")"),
    tuple_type: ($) =>
      seq(
        "(",
        field("elems", $.type),
        ",",
        repeat(seq(field("elems", $.type), ",")),
        optional(field("elems", $.type)),
        ")",
      ),
    unit_type: () => seq("(", ")"),
    record_type: ($) => seq("{", list1("field", $.declare, ","), "}"),

    // Tokens
    //

    lid: () => token(new RustRegex("[_]*[a-z][a-zA-Z0-9_]*")),

    uid: () =>
      token(
        new RustRegex(
          `[_]+[A-Z][a-zA-Z0-9_]*|${identifierExcept("A-Z", [...TYPE_FORMERS, ...BUILTIN_TYPES.filter((name) => /^[A-Z]/.test(name))])}`,
        ),
      ),

    number: () =>
      token(
        choice(
          new RustRegex(
            "[0-9][0-9_]*(\\.[0-9_]+)?([eE][+-]?[0-9_]+)?([uU]?[bBhHiIlL]|[fFdD])?",
          ),
          new RustRegex("0[xX][0-9a-fA-F_]+([hHiIlL]|[uU][bBhHiIlL])?"),
        ),
      ),

    string: () =>
      token(
        seq(
          '"',
          repeat(
            choice(
              new RustRegex('[^"\\\\\\r\\n]'),
              seq("\\", choice('"', "\\", "n", "r", "t", "0")),
            ),
          ),
          '"',
        ),
      ),

    comment: () =>
      token(
        choice(
          seq("//", new RustRegex(".*")),
          seq("/*", new RustRegex("[^*]*\\*+([^/*][^*]*\\*+)*"), "/"),
        ),
      ),
  },

  extras: ($) => [new RustRegex("\\s"), $.comment],
});
