#!/usr/bin/env python3
"""Render the book illustration with the explorer's headless CPU mode."""
import base64
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def render(root, client, server):
    with tempfile.TemporaryDirectory(prefix="resin-manual-") as temporary:
        work = Path(temporary)
        log_path = work / "service.log"
        with log_path.open("w") as log:
            service = subprocess.Popen([
                str(server), "--listen", "127.0.0.1:0",
                "--library-root", str(root / "resin"),
                "--storage", str(work / "dependencies"),
                "--temporary", str(work / "native"),
            ], cwd=root, stdout=log, stderr=log)
        try:
            endpoint = wait_for_service(service, log_path)
            executable = work / ("mandelbrot.exe" if os.name == "nt" else "mandelbrot")
            subprocess.run([
                str(client), "examples/eg011_mandelbrot.resin", "-o", str(executable),
            ], cwd=root, env={**os.environ, "RESIN_SERVER": endpoint},
                stdout=sys.stderr, check=True, timeout=180)
            image = work / "mandelbrot.png"
            subprocess.run([
                str(executable), "--cpu", "--width", "640", "--height", "480",
                "--iterations", "256", "--output", str(image),
            ], cwd=work, stdout=sys.stderr, check=True, timeout=60)
            return base64.b64encode(image.read_bytes()).decode("ascii")
        except Exception:
            print(log_path.read_text(), file=sys.stderr)
            raise
        finally:
            service.terminate()
            try:
                service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()


def wait_for_service(service, log_path):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        match = re.search(r"listening on (http://127\.0\.0\.1:\d+)", log_path.read_text())
        if match:
            return match[1]
        if service.poll() is not None:
            raise RuntimeError("the manual's compiler service exited during startup")
        time.sleep(0.05)
    raise TimeoutError("the manual's compiler service did not start within 30 seconds")


if __name__ == "__main__":
    root, client, server = map(Path, sys.argv[1:])
    print(render(root, client, server))
