# `AGENTS.md` aka `CLAUDE.md`

Rust style
- Prefer clear types and `cargo clippy` / `cargo fmt` clean trees.
- Graph identity is allocation identity via `NodeRef` (pointer `Eq`/`Hash`), not structural equality on `Node`.

## PR hygiene

PRs are reviewed commit-by-commit (tabbing through each diff in order). Treat
that sequence as the review unit: each commit should tell one coherent story and
be understandable on its own, like a small stacked PR.

- Factor work into a few focused commits — not one giant dump, not a trail of
  fixups that belong in earlier steps.
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
