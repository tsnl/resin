#!/usr/bin/env python3
"""Build API chapters and link repository files to this source revision."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == "supports":
    sys.exit(0 if sys.argv[2] == "html" else 1)

context, book = json.load(sys.stdin)
root = Path(context["root"]).resolve()
source = root / context["config"]["book"]["src"]
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def library_chapters(items):
    for item in items:
        chapter = item.get("Chapter")
        if chapter is None:
            continue
        if chapter.get("source_path") == "library.md":
            tool = os.environ.get("RESIN_DOC_TOOL")
            if tool is None:
                subprocess.run(["cargo", "build", "--quiet", "--locked", "-p", "resin"], cwd=root, check=True)
                target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
                if not target.is_absolute():
                    target = root / target
                tool = str(target / "debug" / ("resin.exe" if os.name == "nt" else "resin"))
            for index, module in enumerate(sorted((root / "resin").glob("*.resin")), 1):
                chapter["content"] = chapter["content"].replace(
                    f"(../resin/{module.name})", f"(api/{module.stem}.md)")
                content = subprocess.check_output([tool, "--doc", str(module)], cwd=root, text=True)
                content = content.replace("# " + module.name, "# `$/" + module.name + "`", 1)
                url = f"https://github.com/tsnl/resin/blob/{revision}/resin/{module.name}"
                content += f"\n[Module source]({url}). This page uses exported source signatures; inferred result holes remain visible.\n"
                chapter["sub_items"].append({"Chapter": {
                    "name": "$/" + module.name, "content": content,
                    "number": chapter["number"] + [index], "sub_items": [],
                    "path": "api/" + module.stem + ".md", "source_path": None,
                    "parent_names": chapter["parent_names"] + [chapter["name"]],
                }})
        else:
            library_chapters(chapter["sub_items"])


def chapters(sections):
    for section in sections:
        chapter = section.get("Chapter")
        if chapter is None:
            continue
        path = (source / chapter["source_path"] if chapter["source_path"] else
                root / "resin" / chapter["name"].removeprefix("$/"))

        def link(match):
            target = match.group(2)
            if ":" in target or target.startswith(("#", "/")):
                return match.group(0)
            relative, separator, anchor = target.partition("#")
            destination = (path.parent / relative).resolve()
            if destination.is_relative_to(source):
                return match.group(0)
            if not destination.is_relative_to(root) or not destination.exists():
                raise ValueError(f"{path}: missing repository link: {target}")
            url = f"https://github.com/tsnl/resin/blob/{revision}/{destination.relative_to(root)}"
            if separator:
                url += "#" + anchor
            return f"[{match.group(1)}]({url})"

        chapter["content"] = re.sub(r"(?<!!)\[([^\]]+)\]\(([^\s)]+)\)", link, chapter["content"])
        chapters(chapter["sub_items"])


library_chapters(book["items"])
chapters(book["items"])
json.dump(book, sys.stdout)
