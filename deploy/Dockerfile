FROM docker.io/library/rust:1.96.0-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build pkg-config libx11-dev libxcursor-dev libxi-dev \
    libxinerama-dev libxrandr-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
RUN cargo build --release --locked -p resin-server -p resin-runtime

FROM docker.io/library/debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev ninja-build spirv-tools git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home resin \
    && mkdir -p /var/lib/resin /var/tmp/resin \
    && chown resin:resin /var/lib/resin /var/tmp/resin
COPY --from=build /build/target/release/resin-server /usr/local/bin/resin-server
COPY --from=build /build/target/release/libresin_runtime.a /usr/local/lib/resin/libresin_runtime.a
COPY resin/ /usr/local/share/resin/
COPY crates/resin-runtime/include/ /usr/local/include/resin/
ENV RESIN_LIBRARY_ROOT=/usr/local/share/resin \
    RESIN_RUNTIME_INCLUDE=/usr/local/include/resin \
    RESIN_RUNTIME_LIB=/usr/local/lib/resin/libresin_runtime.a
USER resin
WORKDIR /var/lib/resin
EXPOSE 7412
ENTRYPOINT ["/usr/local/bin/resin-server"]
CMD ["--listen", "0.0.0.0:7412", "--storage", "/var/lib/resin/dependencies", "--temporary", "/var/tmp/resin"]
