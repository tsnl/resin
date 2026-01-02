Platform
- Use `make` targets to do anything if possible
  - `make sync` runs codgen via `make build` and then runs `uv sync`.
  - `make tests` runs all tests and writes test outputs (e.g. image renders).
  - `make sandbox` launches a windowed interactive application.
- Ensure you use `uv` to activate a Python environment with the right dependencies.

Python
- Our code-base is statically typed and must type-check successfully.
  - Use the configured LSP aggressively: it should work well.
  - Run `make check` and `make format` often to automatically and quickly sort imports, `__all__` lists, etc.
  - NEVER use `# type: ignore` or `# noqa` comments to silence type-checker or linter warnings.
  - NEVER use `Any` type. NEVER use `cast()`.
- We target Python 3.14+, so use the most modern syntax and features available.
  - Always use Python3.12+ `type Name = Aliased` statements instead of type aliases.
  - Always use `|` for union types instead of `Union[]`, and `T | None` instead of `Optional[T]`. If `"T"` is specified as a string, note that you can write `"T | None"`. Always use `list[T]`, `dict[K, V]`, etc. instead of `List[T]`, `Dict[K, V]`, etc.
  - Prefer generic type syntax over `Generic[T]` subclasses.
  - Never `from __future__ import annotations`.

Test-driven development
- Run `make tests` often. It's really fast and will catch many issues early. No need to ask for permission.
