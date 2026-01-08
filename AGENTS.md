Platform
- Use `./invoke <task>+` targets to do anything if possible. 
  See `tasks.py`.
- To run a Python shell, use `uv run python`.
- To run specific tests, use `./invoke tests --filter "filter-args"`.
- Run `./invoke tests` often. It's really fast and will catch many issues early. No need to ask for permission.
- Export `ZFW_TEST_LOG_LEVEL=DEBUG` to change the log level when running tests.

Python
- Our code-base is statically typed and must type-check successfully.
  - Use the configured LSP aggressively: it should work well.
  - Run `make check` and `make format` often to automatically and quickly sort imports, `__all__` lists, etc.
  - NEVER use `# type: ignore` or `# noqa` comments to silence type-checker or linter warnings. However, if you find these already in the codebase, leave them as is. 
  - NEVER use `Any` type. NEVER use `cast()`.
- We target Python 3.14+, so use the most modern syntax and features available.
  - Always use Python3.12+ `type Name = Aliased` statements instead of type aliases.
  - Always use `|` for union types instead of `Union[]`, and `T | None` instead of `Optional[T]`. If `"T"` is specified as a string, note that you can write `"T | None"`. Always use `list[T]`, `dict[K, V]`, etc. instead of `List[T]`, `Dict[K, V]`, etc.
  - Prefer generic type syntax over `Generic[T]` subclasses.
  - Never `from __future__ import annotations`.

Code style
- Always re-export public symbols under `zfw` directly. E.g. instead of `from zfw.basic import logger`, use `from zfw import logger`.
