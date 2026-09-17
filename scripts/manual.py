#!/usr/bin/env python3
"""mdBook preprocessor: link repository files outside doc/ to this source revision."""
import json
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


def chapters(sections):
    for section in sections:
        chapter = section.get("Chapter")
        if chapter is None:
            continue
        path = source / chapter["source_path"]

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


chapters(book["items"])
json.dump(book, sys.stdout)
