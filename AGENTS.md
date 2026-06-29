# `AGENTS.md` aka `CLAUDE.md`

Python style
- Always use type annotations, static typechecking and `ruff` lints must pass.
- Targeting Python >=3.14 (PEP 649 deferred annotations): do not use
  `from __future__ import annotations`, and do not quote forward references
  (write `PyTensor | View` and `-> View`, not `"View"`), except for names that
  are not bound at runtime (e.g. imports only under `TYPE_CHECKING`).
- All imports must be at the top-level.

## PR hygiene

PRs are reviewed commit-by-commit (tabbing through each diff in order). Treat
that sequence as the review unit: each commit should tell one coherent story and
be understandable on its own, like a small stacked PR.

- Factor work into a few focused commits — not one giant dump, not a trail of
  fixups (`address review`, `inline X`, `remove __future__`, etc.) that belong
  in earlier steps.
- When you revise a PR, rewrite the commit stack for the whole branch: soft-reset
  onto the base, then re-stage and commit cleanly. Fold follow-up fixes into the
  commits where that logic first appears; do not append drive-by cleanup at the
  tip.
- Before pushing, skim the commit list as a reviewer would. If a later commit
  only corrects an earlier one, squash that correction backward.
- Do not include prefixes like `[resin]`, `feat:`, etc in PR titles or commit messages
  unless explicitly requested.
- If the user asks to reflow commits, it means they want you to rewrite the commit stack
  potentially including more changes made by the user.
