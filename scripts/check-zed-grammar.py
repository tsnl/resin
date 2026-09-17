"""Require Zed's pinned grammar to match the grammar tested in this checkout."""

from pathlib import Path
import subprocess
import tomllib


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = tomllib.loads((root / "editors/zed/extension.toml").read_text())
    grammar = manifest["grammars"]["resin"]
    revision = grammar["rev"]
    directory = grammar["path"]
    # Check source and generated artifacts: query tests compile the local parser,
    # while Zed builds the parser from this revision, even for dev extensions.
    paths = [f"{directory}/grammar.js"]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / directory / "src").rglob("*"))
        if path.is_file()
    )
    mismatches = []
    for path in paths:
        pinned = subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=root, capture_output=True
        )
        if pinned.returncode or pinned.stdout != (root / path).read_bytes():
            mismatches.append(path)
    if mismatches:
        raise SystemExit(
            f"Zed grammar pin {revision} does not match the checkout:\n"
            + "\n".join(mismatches)
            + "\nCommit and push the regenerated grammar, then update "
            "editors/zed/extension.toml to that commit. "
            "The pinned commit must be available in local Git history."
        )
    print(f"Zed grammar pin {revision} matches the current grammar.")


if __name__ == "__main__":
    main()
