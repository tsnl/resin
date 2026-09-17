// Resin's declaration and literal syntax; hljs is supplied by mdBook.
if (typeof hljs !== "undefined") {
  hljs.registerLanguage("resin", function (hljs) {
    return {
      name: "Resin",
      keywords: {
        keyword: "export import extern intrinsic fn struct type let mut const if else while match return break continue assert",
        literal: "true false None",
        type: "Ref Ptr Err ArcPtr ArcSpan Span GpuPtr GpuSpan WeakPtr WeakSpan bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 str String Never",
      },
      contains: [
        hljs.C_LINE_COMMENT_MODE,
        hljs.C_BLOCK_COMMENT_MODE,
        hljs.QUOTE_STRING_MODE,
        { className: "number", begin: "\\b(?:0[xX][0-9a-fA-F_]+|[0-9][0-9_]*(?:\\.[0-9_]+)?(?:[eE][+-]?[0-9_]+)?)\\b" },
        { className: "meta", begin: "@[a-z_]+" },
      ],
    };
  });
  document.querySelectorAll("code.language-resin").forEach(function (block) {
    block.removeAttribute("data-highlighted");
    hljs.highlightBlock(block);
  });
}
