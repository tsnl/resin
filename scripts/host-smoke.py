"""Exercise explicit service startup and the real CLI on each supported CI host."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

root = Path(__file__).resolve().parent.parent
target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
suffix = ".exe" if os.name == "nt" else ""
with tempfile.TemporaryDirectory() as directory:
    server = subprocess.Popen(
        [target / "debug" / f"resin-server{suffix}", "--listen", "127.0.0.1:7412",
         "--library-root", root / "resin", "--temporary", directory], cwd=directory)
    try:
        url = "http://127.0.0.1:7412"
        deadline = time.monotonic() + 30
        while True:
            try:
                with urllib.request.urlopen(f"{url}/v1/capabilities", timeout=1):
                    break
            except OSError:
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("compiler service did not become ready")
                time.sleep(0.1)
        subprocess.run([target / "debug" / f"resin{suffix}", root / "examples/eg001.resin"],
                       env={**os.environ, "RESIN_SERVER": url}, check=True)
    finally:
        server.terminate()
        server.wait(timeout=15)
