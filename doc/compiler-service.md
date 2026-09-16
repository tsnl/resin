# Compiler service

`resin` is a thin client. Builds, runs, and LSP sessions require an explicit HTTP(S)
URL in `RESIN_SERVER`. The service owns semantic compilation, shared caches, managed
libraries, and native tools. The client uploads user files, downloads one executable,
and runs it locally when no `-o` is supplied. Formatting and `--embed` remain local.

## Local development

In the repository development shell, build both applications and the native archive:

```sh
nix-shell
cargo build -p resin -p resin-server -p resin-runtime
./target/debug/resin-server --listen 127.0.0.1:7412
```

In another development shell:

```sh
export RESIN_SERVER=http://127.0.0.1:7412
cargo run -- examples/eg001.resin
cargo run -- examples/eg001.resin -o program
cargo run -- --lsp /absolute/path/to/project
```

There is no automatic startup, workspace address file, discovery, `resin.toml`, or
local compiler fallback. Invalid URLs, unreachable services, protocol mismatches,
and unsupported targets fail explicitly. Nested URL prefixes are preserved, allowing
an operator to mount the API behind a reverse proxy. Editors inherit `RESIN_SERVER`;
LSP initialization succeeds only after capability negotiation.

The service needs Ninja, a C compiler (`CC`/server `--cc`), the runtime archive and
headers, and SPIR-V Tools (`SPIRV_OPT`/server `--spirv-opt`) when compiling shaders.
`NINJA` selects Ninja. Git is needed for configured dependencies. Host-only compilation
needs no GPU or Vulkan SDK. Programs using GPU/window functionality need the required
runtime libraries/device on the **client** that runs the downloaded program.
The service uses its own executable for `--embed`; keep its runtime archive and
library/header installation from the same build. The client has no native tool flags.

## Inputs and identity

The client fully parses each reachable user file with the existing CST parser and
queries its preamble. It follows relative/parent imports, freezes one version per
physical source, deduplicates aliases, terminates cycles, and uploads exact text plus
explicit import bindings. Invalid bodies still permit preamble discovery and editor
recovery. Acquisition is a captured view, not an atomic filesystem transaction.
LSP buffers take precedence over disk; stale results are rejected by document epoch,
version, dependency, and disk-event checks. LSP never saves edits to disk.

Names are logical paths anchored at the entry's parent, independent of a checkout's
absolute location. Non-UTF-8 local path components use lossless percent encoding.
Leading `../` names are permitted source identifiers; they are never extraction paths.
User names cannot use the reserved `$/` namespace. The server validates parsed edges
and never repairs missing user uploads by opening a matching server path.

`$/` imports select the service's frozen standard library or pinned packages. Library
and runtime-header bytes are read once during startup; restart to adopt a new snapshot.
`RESIN_LIBRARY_ROOT`/`--library-root`, `RESIN_RUNTIME_INCLUDE`, and `RESIN_RUNTIME_LIB`
are **server** configuration. Managed definitions returned to an editor become
immutable local mirror files; their compiler identities remain managed names.

Every request selects an entry and complete current metadata. A returned opaque
`InputHandle { instance, id }` can be used with source replacements/deletions for the
next request. A delta replaces entry, imports, header bundles, and acquisition
diagnostics completely; handles retain complete inputs without predecessor chains.
Clients keep full captures and retry once when a base was evicted or the server
restarted. A changed managed snapshot refreshes capabilities and requires recapture.
Request IDs, revision numbers, handles, local paths, and caller identity are absent
from compiler cache keys. Equivalent CLI and LSP graphs therefore share results.

## C header directories

An extern group's header is resolved through ordered explicit client roots, its
importer's directory, advertised managed roots, and finally target system headers.
Use `resin -I include -I vendor/include main.resin` (also `--include-root`). LSP
accepts the same flags or `initializationOptions: { "includeRoots": ["include"] }`;
relative initialization roots use the selected editor project directory.

Upload complete selected directory bundles, including nested and non-`.h` files.
Overlapping roots are coalesced while preserving search order. Place headers in
separate directories: generated outputs or changing logs inside a selected bundle
also change its identity. Bundles are binary-safe base64 file lists with a framed
BLAKE3 digest, checked independently on the server. Names must be portable relative
UTF-8 paths. Contained symlinks are flattened; cycles, escaping links, unsupported
names, and path/file conflicts diagnose before extraction. Explicitly selecting a
wider root can include a dependency while preserving its directory layout.

Bindings retain the declaring source and original header spelling, so two modules
can each declare `"api.h"` with different contents. Empty extern groups still select
headers. Absolute direct header names are resolved/uploaded on the client. All
bundle files and ordered roots enter native cache keys, including currently unused
files. The compiler-injected runtime ABI header uses an explicit protected binding.
Native preprocessing records and validates actual dependency paths before compiling
captured `.i` bytes; missing or escaping transitive headers cannot produce artifacts
by borrowing unrelated server files. This check is not an operating-system filesystem
sandbox: preprocessing runs with service-account permissions. Operators choose the
container/service isolation and access policy for their callers.

## Cache and concurrency configuration

Compiler libraries return completed immutable values and know nothing about HTTP.
`Cache<K,V>::update()` builds missing keys with bounded parallelism and publishes a
new immutable generation. Requested values are retained even when their count exceeds
capacity, with a warning. Unrequested entries are evicted by recency while fitting
capacity. There is no TTL, prune operation, workspace table, or background GC.
Server handlers explicitly sequence passes and publish each cache through shared-pointer
compare-and-swap; a failed publication rebases selected results without recomputing them.

Default capacities are **entries**, not bytes:

| Cache | Capacity |
| --- | ---: |
| Input handles | 64 |
| Source / CST / AST, each | 4096 |
| HIR / verified LIR, each | 64 |
| Generated projects | 32 |
| Simultaneously admitted HTTP requests | 64 |

`resin-server --cache-capacity N` overrides every cache; `--requests N` sets positive
request admission. Embedding applications may set individual `Config.capacities`.
Active requests and streamed downloads retain selected outputs after head eviction;
final owners release temporary directories. Cancellation propagates to bounded CPU
work and native process trees. SIGINT/SIGTERM drain service work. Native Ninja caches
live under the service working directory's `build/`; `--temporary` owns transient
projects/artifacts and `--storage` owns pinned dependency checkouts.

## Wire contract

The full strict serde contract is [resin-protocol/lib.rs](../crates/resin-protocol/src/lib.rs).
Protocol version is 1; unknown fields and malformed inputs fail explicitly.
Request bodies and client JSON responses are limited to 64 MiB. Artifact files stream
separately. Connection/capability negotiation has a 10-second timeout; long builds
remain cancellable rather than using a fixed build timeout. Once cancelled, clients
allow up to five seconds for cancellation retries and response cleanup, then close
the HTTP exchange so a stalled service cannot block editor shutdown indefinitely.

| Route | Contract |
| --- | --- |
| `GET /v1/capabilities` | Version, instance, managed snapshot, targets, entry profiles, header catalogues |
| `POST /v1/analyze` | Request token, revision, full/delta inputs, ordered queries; diagnostics and ordered query results |
| `POST /v1/build` | Request token, revision, inputs, explicit build contract; one binary file |
| `DELETE /v1/requests/{request}` | 204 for cancellation accepted, 404 when not registered |

Build contracts select OS, architecture, ABI, runtime identity, exported entry,
Host/Shader profile, Debug/Release, and per-function specialization allowance. Only
native Host executable targets matching the server installation are advertised today;
embedded shaders still compile. Direct shader artifacts/cross-compilation are rejected.
Destination paths, run arguments, local environment, and working directory stay local.

Success uses `application/octet-stream`, exact `Content-Length`, and a bounded 4096-byte
`Resin-Metadata` header containing base64url JSON `BuildMetadata`. The client verifies
revision, target, artifact kind, safe name, length, and BLAKE3 before atomically replacing
an adjacent destination file. Failed/cancelled downloads preserve existing outputs.
There is no archive extraction or detached download lease in version 1.

Failures are JSON `{ code, message, diagnostics, managed_sources }`: invalid requests
400; incompatible protocol, unavailable inputs/snapshot, unsupported target 409;
compilation 422; busy 429; cancellation 499; internal failure 500. Clients use the
structured code for retry decisions. Cancellation retains the original request future
and retries an early DELETE 404 until admission or original completion, covering the
DELETE-before-POST race. The API currently has no authentication layer; restrict access
or put an authenticated TLS proxy in front of it for a shared deployment.

## Pinned dependencies

`--dependencies /etc/resin/dependencies.json` reads an operator-owned JSON array:

```json
[{"name":"math","repository":"https://example.org/team/math.git",
  "commit":"0123456789012345678901234567890123456789",
  "source_subdirectory":"resin","header_subdirectories":["include"]}]
```

Replace the example with a real full Git commit. Startup fetches that commit, verifies
it, and freezes source/header bytes before serving. Sources appear at
`$/deps/math/...`; managed header root IDs include package/root identity. Moving
branches/tags are not accepted pins. This is service configuration, not a project
manifest. Existing source/dependency snapshots remain immutable throughout requests.

## Manual Docker deployment

Build the supplied [Dockerfile](../deploy/Dockerfile) from the repository root:

```sh
docker build -f deploy/Dockerfile -t resin-server .
docker run --rm --name resin-server -p 127.0.0.1:7412:7412 resin-server
export RESIN_SERVER=http://127.0.0.1:7412
```

The container runs as UID 10001 and contains server libraries, headers, archive, and
native tools. It requires no client workspace mount. Persistent volumes for
`/var/lib/resin` may retain dependency/native build state; provision them writable by
UID 10001. Cache heads are in memory and restart cold. See Docker's
[container run documentation](https://docs.docker.com/engine/containers/run/).

## Manual systemd deployment

There is no installer. Build release server/runtime artifacts, copy the binary,
`libresin_runtime.a`, `resin/` libraries, and `crates/resin-runtime/include/` to the
paths in the chosen unit. Install native tools separately. Match all files to the
same compiler build; a copied binary alone is insufficient.

For a service account, provision user/group `resin`, place the installation under
`/opt/resin`, copy [resin-server.service](../deploy/systemd/resin-server.service) into
`/etc/systemd/system/`, then run `systemctl daemon-reload` and
`systemctl enable --now resin-server`. The unit provisions server-owned state/cache
directories; these contain build data, not connection-discovery files.

For a rootless per-user service, install under `~/.local/` using the paths in
[resin-server-user.service](../deploy/systemd/resin-server-user.service), create
`~/.local/state/resin` and `~/.cache/resin`, and copy the unit to
`~/.config/systemd/user/resin-server.service`. Run:

```sh
systemctl --user daemon-reload
systemctl --user enable --now resin-server
export RESIN_SERVER=http://127.0.0.1:7412
```

Optional `loginctl enable-linger` keeps the user manager alive after logout and starts
it at boot; local policy may require administrator authorization. See systemd's
[loginctl documentation](https://github.com/systemd/systemd/blob/main/man/loginctl.xml).
Each client/editor still receives its explicit URL. Configure service native-tool
paths and runtime libraries for the service environment; an interactive Nix shell's
environment is not automatically inherited by systemd.
