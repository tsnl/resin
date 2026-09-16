#!/usr/bin/env python3
"""Measure HTTP analysis, LSP latency, and complete CLI builds using local binaries.

Run inside shell.nix after building resin and resin-server in release mode. Results
include raw samples and observational statistics; no timing thresholds are asserted.
"""

import argparse
import concurrent.futures
import contextlib
import datetime
import http.client
import json
import math
import os
from pathlib import Path
import platform
import queue
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments():
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=target / "release")
    parser.add_argument("--output", type=Path, default=ROOT / "build/service-benchmark")
    parser.add_argument("--trials", type=positive, default=3)
    parser.add_argument("--samples", type=positive, default=30)
    parser.add_argument("--edit-samples", type=positive, default=20)
    parser.add_argument("--build-samples", type=positive, default=8)
    parser.add_argument("--build-edit-samples", type=positive, default=5)
    parser.add_argument("--suite", choices=["http", "lsp", "build", "all"], default="all")
    parser.add_argument("--label", help="Optional description of this run")
    args = parser.parse_args()
    args.bin_dir = args.bin_dir.resolve()
    args.output = args.output.resolve()
    return args


def timed(function):
    started = time.perf_counter()
    value = function()
    return (time.perf_counter() - started) * 1000, value


def fixture(value=42):
    sources = []
    for index in range(16):
        text = (
            f"export {{ value_{index} }}; def value_{index}() -> int = "
            f"{{ {value if index == 0 else 42} }};\n"
        )
        text += "".join(
            f"def helper_{helper}(value: int) -> int = {{ value + {helper} }};\n"
            for helper in range(64)
        )
        sources.append(dict(name=f"file{index}.resin", text=text))
    imports = ", ".join(f'"file{index}.resin"' for index in range(16))
    calls = " + ".join(f"value_{index}()" for index in range(16))
    sources.append(dict(
        name="main.resin",
        text=f"export {{ main }}; import {{ {imports} }}; def main() -> int = {{ {calls} }};\n",
    ))
    return dict(
        entry="main.resin", sources=sources,
        imports=[dict(importer="main.resin", reference=f"file{i}.resin", target=f"file{i}.resin")
                 for i in range(16)],
        headers=dict(bundles=[], bindings=[], include_roots=[]), acquisition_diagnostics=[],
    )


def write_fixture(directory, data):
    directory.mkdir(parents=True, exist_ok=True)
    for source in data["sources"]:
        (directory / source["name"]).write_text(source["text"], encoding="utf-8")


def selection(inputs, base=None, replacements=None):
    if base is None:
        return dict(kind="full", inputs=inputs)
    return dict(
        kind="delta", base=base, entry=inputs["entry"], replacements=replacements or [],
        deleted=[], imports=inputs["imports"], headers=inputs["headers"],
        acquisition_diagnostics=[], managed_snapshot=inputs["managed_snapshot"],
    )


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def rss(pid):
    if platform.system() != "Linux":
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_name():
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None


class Http:
    def __init__(self, port, timeout=60):
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def request(self, method, route, data=None):
        payload = None if data is None else json.dumps(data, separators=(",", ":")).encode()
        self.connection.request(method, "/v1/" + route, payload, {"Content-Type": "application/json"})
        response = self.connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw.decode(errors='replace')}")
        if not response.getheader("Content-Type", "").startswith("application/json"):
            raise RuntimeError("expected JSON HTTP response")
        return json.loads(raw)

    def close(self):
        self.connection.close()


def analyze(connection, inputs, base=None, replacements=None):
    result = connection.request("POST", "analyze", dict(
        request=str(uuid.uuid4()), revision=1, inputs=selection(inputs, base, replacements), queries=[],
    ))
    if result.get("revision") != 1 or result.get("diagnostics") != [] or result.get("results") != []:
        raise RuntimeError(f"invalid analysis response: {result}")
    if not result.get("input", {}).get("id") or not result["input"].get("instance"):
        raise RuntimeError(f"missing analysis input handle: {result}")
    return result


class Lsp:
    def __init__(self, binary, port, directory, log_path):
        self.process = None
        self.reader = None
        self.log = open(log_path, "ab")
        self.messages = queue.Queue()
        self.counter = 0
        try:
            self.process = subprocess.Popen(
                [binary, "--lsp", directory], cwd=directory,
                env={**os.environ, "RESIN_SERVER": f"http://127.0.0.1:{port}"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            )
            self.reader = threading.Thread(target=self.read, daemon=True)
            self.reader.start()
            initialized = self.request("initialize", dict(
                processId=None, rootUri=directory.as_uri(), capabilities={},
            ))
            if not isinstance(initialized, dict) or "capabilities" not in initialized:
                raise RuntimeError(f"invalid initialize response: {initialized}")
            self.send(dict(jsonrpc="2.0", method="initialized", params={}))
        except BaseException:
            self.close(graceful=False)
            raise

    def read(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError("LSP exited before a response")
                    if line == b"\r\n":
                        break
                    name, value = line.decode("ascii").split(":", 1)
                    headers[name.lower()] = value.strip()
                length = int(headers["content-length"])
                if length < 0 or length > 64 * 1024 * 1024:
                    raise RuntimeError("invalid LSP message length")
                raw = self.process.stdout.read(length)
                if len(raw) != length:
                    raise RuntimeError("truncated LSP message")
                value = json.loads(raw)
                if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
                    raise RuntimeError(f"invalid LSP message: {value}")
                self.messages.put(value)
        except Exception as error:
            self.messages.put(error)

    def send(self, value):
        raw = json.dumps(value).encode()
        self.process.stdin.write(f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        self.process.stdin.flush()

    def wait(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LSP response timeout")
            try:
                value = self.messages.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("LSP response timeout") from error
            if isinstance(value, Exception):
                raise value
            if "method" in value and "id" in value:
                self.send(dict(jsonrpc="2.0", id=value["id"], result=None))
            elif predicate(value):
                return value

    def request(self, method, params, timeout=30):
        self.counter += 1
        identifier = self.counter
        self.send(dict(jsonrpc="2.0", id=identifier, method=method, params=params))
        result = self.wait(lambda message: message.get("id") == identifier, timeout)
        if "error" in result or "result" not in result:
            raise RuntimeError(f"invalid LSP {method} response: {result}")
        return result["result"]

    def change(self, uri, text, version, opening=False):
        if opening:
            params = dict(textDocument=dict(uri=uri, languageId="resin", version=version, text=text))
        else:
            params = dict(textDocument=dict(uri=uri, version=version), contentChanges=[dict(text=text)])
        self.send(dict(jsonrpc="2.0", method="textDocument/didOpen" if opening else "textDocument/didChange",
                       params=params))
        result = self.wait(lambda message: message.get("method") == "textDocument/publishDiagnostics"
                           and message["params"]["uri"] == uri
                           and message["params"].get("version") == version)
        if result["params"]["diagnostics"] != []:
            raise RuntimeError(f"unexpected LSP diagnostics: {result}")

    def hover(self, uri, column):
        result = self.request("textDocument/hover", dict(
            textDocument=dict(uri=uri), position=dict(line=0, character=column),
        ))
        if not isinstance(result, dict) or not result.get("contents"):
            raise RuntimeError(f"missing hover result: {result}")
        return result

    def close(self, graceful=True):
        try:
            if graceful and self.process is not None and self.process.poll() is None:
                try:
                    self.request("shutdown", None, timeout=5)
                    self.send(dict(jsonrpc="2.0", method="exit"))
                    self.process.wait(timeout=5)
                except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError):
                    pass
        finally:
            stop(self.process)
            if self.reader is not None:
                self.reader.join(timeout=2)
            if self.process is not None:
                with contextlib.suppress(OSError):
                    self.process.stdin.close()
                self.process.stdout.close()
            self.log.close()


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.binary = args.bin_dir / ("resin" + EXE)
        self.server_binary = args.bin_dir / ("resin-server" + EXE)
        for binary in (self.binary, self.server_binary):
            if not binary.is_file():
                raise RuntimeError(f"missing binary: {binary}; build resin and resin-server first")
        args.output.mkdir(parents=True, exist_ok=True)
        write_fixture(args.output / "fixture", fixture())
        self.server_count = 0
        self.results = {"environment": dict(
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            label=args.label, platform=platform.platform(), machine=platform.machine(), cpu=cpu_name(),
            logical_cpus=os.cpu_count(), python=platform.python_version(), revision=git("rev-parse", "HEAD"),
            checkout_changes=git("status", "--porcelain", "--untracked-files=no"),
            binaries={binary.name: dict(path=str(binary), bytes=binary.stat().st_size,
                                       modified_ns=binary.stat().st_mtime_ns)
                      for binary in (self.binary, self.server_binary)},
            transport="HTTP loopback", fixture_sources=17, fixture_declarations=1041,
            fixture_bytes=sum(len(source["text"].encode()) for source in fixture()["sources"]),
            options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            cold="fresh service process and empty native cache; OS page cache is not cleared (warm pages)",
            warmup="one unmeasured request before each repeated measurement group",
            build_mode="CLI -o native release; startup excluded; client launch/acquisition/upload/download included; executable validation excluded",
            rss="Linux /proc instantaneous RSS in KiB; null when unavailable; not a peak or exact cache size",
        )}
        self.save()

    def save(self):
        temporary = self.args.output / "results.json.tmp"
        temporary.write_text(json.dumps(self.results, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.args.output / "results.json")

    def record(self, name, values, **extra):
        ordered = sorted(values)
        self.results[name] = dict(
            samples_ms=values, sample_count=len(values), median_ms=statistics.median(values),
            p95_ms=ordered[math.ceil(len(ordered) * .95) - 1], **extra,
        )
        print(name, json.dumps({key: value for key, value in self.results[name].items()
                               if key != "samples_ms"}), flush=True)
        self.save()

    @contextlib.contextmanager
    def server(self):
        self.server_count += 1
        log_path = self.args.output / f"server-{self.server_count}.log"
        with tempfile.TemporaryDirectory(prefix="server-cache-", dir=self.args.output) as directory:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            with log_path.open("ab") as log:
                process = None
                connection = Http(port, timeout=2)
                started = time.perf_counter()
                try:
                    process = subprocess.Popen(
                        [self.server_binary, "--listen", f"127.0.0.1:{port}", "--library-root", ROOT / "resin",
                         "--temporary", directory],
                        cwd=directory, stdout=log, stderr=log,
                    )
                    deadline = started + 30
                    while True:
                        try:
                            capabilities = connection.request("GET", "capabilities")
                            break
                        except (OSError, http.client.HTTPException):
                            connection.close()
                            if process.poll() is not None or time.perf_counter() > deadline:
                                raise RuntimeError(f"server startup failed; see {log_path}")
                            time.sleep(.005)
                    if not capabilities.get("instance") or not capabilities.get("managed_snapshot"):
                        raise RuntimeError(f"invalid capabilities response: {capabilities}")
                    startup = (time.perf_counter() - started) * 1000
                    connection.close()
                    connection = Http(port)
                    yield process, port, connection, capabilities, startup
                finally:
                    connection.close()
                    stop(process)

    def http_analysis(self):
        cold, full, delta, edits, starts, retained = [], [], [], [], [], []
        for _ in range(self.args.trials):
            with self.server() as (process, port, connection, capabilities, startup):
                starts.append(startup)
                data = fixture()
                data["managed_snapshot"] = capabilities["managed_snapshot"]
                elapsed, previous = timed(lambda: analyze(connection, data))
                cold.append(elapsed)
                previous = analyze(connection, data)
                for _ in range(self.args.samples):
                    elapsed, previous = timed(lambda: analyze(connection, data))
                    full.append(elapsed)
                previous = analyze(connection, data, previous["input"])
                for _ in range(self.args.samples):
                    elapsed, previous = timed(lambda: analyze(connection, data, previous["input"]))
                    delta.append(elapsed)
                data["sources"][0]["text"] = fixture(99)["sources"][0]["text"]
                previous = analyze(connection, data, previous["input"], [data["sources"][0]])
                for index in range(self.args.edit_samples):
                    data["sources"][0]["text"] = fixture(100 + index)["sources"][0]["text"]
                    elapsed, previous = timed(lambda: analyze(
                        connection, data, previous["input"], [data["sources"][0]],
                    ))
                    edits.append(elapsed)
                retained.append(rss(process.pid))
                if len(starts) == self.args.trials:
                    self.http_concurrency(port, data)
        self.record("service_startup", starts)
        self.record("http_analysis_cold_17_files", cold)
        self.record("http_analysis_warm_full_17_files", full)
        self.record("http_analysis_warm_delta_17_files", delta)
        self.record("http_analysis_edit_dependency_17_files", edits, server_rss_kib=retained)

    def http_concurrency(self, port, data):
        workers, per_worker = 8, 8
        started = []
        barrier = threading.Barrier(workers + 1, action=lambda: started.append(time.perf_counter()))

        def batch(_):
            with contextlib.closing(Http(port)) as connection:
                try:
                    analyze(connection, data)
                    barrier.wait(timeout=60)
                    return [timed(lambda: analyze(connection, data))[0] for _ in range(per_worker)]
                except BaseException:
                    barrier.abort()
                    raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [pool.submit(batch, index) for index in range(workers)]
            barrier.wait(timeout=60)
            samples = [sample for task in tasks for sample in task.result()]
            elapsed = time.perf_counter() - started[0]
        self.record("http_warm_concurrency_8", samples, workers=workers, requests_per_worker=per_worker,
                    requests_per_second=len(samples) / elapsed, batch_elapsed_ms=elapsed * 1000)

    def lsp_analysis(self):
        opened, hovers, edits, server_memory, client_memory = [], [], [], [], []
        for trial in range(self.args.trials):
            directory = self.args.output / f"lsp-fixture-{trial}"
            data = fixture()
            write_fixture(directory, data)
            with self.server() as (process, port, _, _, _):
                client = Lsp(self.binary, port, directory, self.args.output / f"lsp-{trial}.log")
                try:
                    uri = (directory / "main.resin").as_uri()
                    original = next(source["text"] for source in data["sources"] if source["name"] == "main.resin")
                    elapsed, _ = timed(lambda: client.change(uri, original, 1, opening=True))
                    opened.append(elapsed)
                    column = original.index("value_0()") + 2
                    client.hover(uri, column)
                    for _ in range(self.args.samples):
                        elapsed, _ = timed(lambda: client.hover(uri, column))
                        hovers.append(elapsed)
                    client.change(uri, original.replace("value_0()", "(value_0() + 1000)"), 2)
                    for index in range(self.args.edit_samples):
                        text = original.replace("value_0()", f"(value_0() + {index + 1})")
                        elapsed, _ = timed(lambda: client.change(uri, text, index + 3))
                        edits.append(elapsed)
                    server_memory.append(rss(process.pid))
                    client_memory.append(rss(client.process.pid))
                finally:
                    client.close()
        self.record("lsp_open_to_diagnostics_17_files", opened)
        self.record("lsp_warm_hover_17_files", hovers)
        self.record("lsp_edit_to_diagnostics_17_files", edits,
                    server_rss_kib=server_memory, client_rss_kib=client_memory)

    def cli_builds(self):
        for kind in ("tiny", "17_files"):
            cold, warm, edits = [], [], []
            cold_bytes, warm_bytes, edit_bytes = [], [], []
            for trial in range(self.args.trials):
                directory = self.args.output / f"build-{kind}-{trial}"
                data = fixture() if kind == "17_files" else dict(sources=[dict(
                    name="main.resin", text="export { main }; def main() -> int = { 42 };\n",
                )])
                write_fixture(directory, data)
                artifact = directory / ("program" + EXE)
                with self.server() as (_, port, _, _, _):
                    def build():
                        result = subprocess.run(
                            [self.binary, directory / "main.resin", "-o", artifact], cwd=directory,
                            env={**os.environ, "RESIN_SERVER": f"http://127.0.0.1:{port}"},
                            capture_output=True, timeout=180,
                        )
                        if result.returncode:
                            raise RuntimeError(result.stderr.decode(errors="replace"))
                        size = artifact.stat().st_size
                        if not size:
                            raise RuntimeError("built artifact is empty")
                        return size

                    def validate(value):
                        expected = (15 * 42 if kind == "17_files" else 0) + value
                        expected = expected & 0xffffffff if os.name == "nt" else expected % 256
                        result = subprocess.run([artifact], capture_output=True, timeout=30)
                        if result.returncode != expected:
                            raise RuntimeError(f"program returned {result.returncode}, expected {expected}")

                    elapsed, size = timed(build)
                    validate(42)
                    cold.append(elapsed)
                    cold_bytes.append(size)
                    build()
                    validate(42)
                    for _ in range(self.args.build_samples):
                        elapsed, size = timed(build)
                        validate(42)
                        warm.append(elapsed)
                        warm_bytes.append(size)
                    name = "file0.resin" if kind == "17_files" else "main.resin"
                    original = next(source["text"] for source in data["sources"] if source["name"] == name)
                    (directory / name).write_text(original.replace("{ 42 }", "{ 99 }"), encoding="utf-8")
                    build()
                    validate(99)
                    for index in range(self.args.build_edit_samples):
                        value = 100 + index
                        (directory / name).write_text(original.replace("{ 42 }", f"{{ {value} }}"), encoding="utf-8")
                        elapsed, size = timed(build)
                        validate(value)
                        edits.append(elapsed)
                        edit_bytes.append(size)
            self.record(f"cli_build_cold_{kind}", cold, artifact_bytes=cold_bytes)
            self.record(f"cli_build_warm_{kind}", warm, artifact_bytes=warm_bytes)
            self.record(f"cli_build_edit_{kind}", edits, artifact_bytes=edit_bytes)


def main():
    args = arguments()
    benchmark = Benchmark(args)
    try:
        if args.suite in ("http", "all"):
            benchmark.http_analysis()
        if args.suite in ("lsp", "all"):
            benchmark.lsp_analysis()
        if args.suite in ("build", "all"):
            benchmark.cli_builds()
    except BaseException as error:
        benchmark.results["failure"] = f"{type(error).__name__}: {error}"
        benchmark.save()
        raise


if __name__ == "__main__":
    main()
