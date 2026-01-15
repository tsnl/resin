Worktree workflow
- This repository uses a bare git repository with worktrees. Each branch lives in its own folder.
- The repository root (containing `.git`) is bare—do not create files directly there.
- Each worktree folder (e.g. `main/`, `feature-branch/`) is a complete checkout of that branch.
- When working on a new feature or task:
  1. Create a new worktree/branch: `git worktree add <branch-name> -b <branch-name> main`
  2. Work exclusively within that worktree folder.
  3. Do not modify files in other worktree folders.
- Always respect worktree boundaries: changes for one branch should only be made in its corresponding folder.
- To list all worktrees: `git worktree list`
- To remove a worktree after merging: `git worktree remove <folder-name>`
- Each worktree should have its own virtual environment. Run `uv sync` in each worktree folder to create a separate `.venv` for that branch.

Platform
- Use `./invoke <task>+` targets to do anything if possible.
  See `tasks.py`.
- To run a Python shell, use `uv run python`.
- To run specific tests, use `./invoke tests --filter "filter-args"`.
- Run `./invoke tests` often. It's really fast and will catch many issues early. No need to ask for permission.
- Export `RESIN_TEST_LOG_LEVEL=DEBUG` to change the log level when running tests.

Python
- Our code-base is statically typed and must type-check successfully.
  - Use the configured LSP aggressively: it should work well.
  - Run `./invoke check` and `./invoke format` often to automatically and quickly sort imports, `__all__` lists, etc.
  - NEVER use `# type: ignore` or `# noqa` comments to silence type-checker or linter warnings. However, if you find these already in the codebase, leave them as is. 
  - NEVER use `Any` type. NEVER use `cast()`.
- We target Python 3.14+, so use the most modern syntax and features available.
  - Always use Python3.12+ `type Name = Aliased` statements instead of type aliases.
  - Always use `|` for union types instead of `Union[]`, and `T | None` instead of `Optional[T]`. If `"T"` is specified as a string, note that you can write `"T | None"`. Always use `list[T]`, `dict[K, V]`, etc. instead of `List[T]`, `Dict[K, V]`, etc.
  - Prefer generic type syntax over `Generic[T]` subclasses.
  - Never `from __future__ import annotations`.

Code style
- Always re-export public symbols under `resin` directly. E.g. instead of `from resin.basic import logger`, use `from resin import logger`.
