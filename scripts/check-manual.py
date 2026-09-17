"""Check local links, anchors, and assets in an mdBook HTML build (no network)."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in ("a", "link") and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("img", "script", "iframe") and attrs.get("src"):
            self.links.append(attrs["src"])


root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/manual").resolve()
pages = {path: Page(path) for path in root.rglob("*.html")}
if not pages:
    sys.exit(f"No HTML pages in {root}; run mdbook build first")
errors = []
for path, page in pages.items():
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        if url.path.startswith("/"):
            target = root / unquote(url.path).lstrip("/")
        if target.is_dir():
            target /= "index.html"
        if not target.is_relative_to(root) or not target.exists():
            errors.append(f"{path.relative_to(root)}: missing target {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f"{path.relative_to(root)}: missing anchor {link}")
if errors:
    sys.exit("\n".join(sorted(set(errors))))
print(f"Checked local links and assets in {len(pages)} HTML pages")
