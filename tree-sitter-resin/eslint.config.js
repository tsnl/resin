import js from "@eslint/js";
import globals from "globals";

export default [
  {
    files: ["grammar.js"],
    ...js.configs.recommended,
    languageOptions: {
      globals: {
        ...globals.es2022,
        grammar: "readonly",
        seq: "readonly",
        repeat: "readonly",
        repeat1: "readonly",
        field: "readonly",
        optional: "readonly",
        choice: "readonly",
        prec: "readonly",
        token: "readonly",
        RustRegex: "readonly",
      },
    },
  },
];
