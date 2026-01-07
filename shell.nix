let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  # Add necessary system libraries to the build inputs
  buildInputs = with pkgs; [
    # nix-ld is required for UV
    nix-ld

    # Python
    uv

    # zlib (for NumPy)
    zlib

    # GLFW:
    glfw # The GLFW library
    # GLFW for Wayland
    wayland
    wayland-protocols
    # GLFW for X11
    xorg.libX11
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi

    # Vulkan:
    vulkan-loader
    vulkan-tools
    vulkan-headers
    vulkan-validation-layers

    # Slang compiler
    shader-slang

    # RenderDoc
    renderdoc

    # # Radeon GPU Profiler (non-free)
    # rgp

    # amdgpu_top
    amdgpu_top

    # astcenc: ASTC texture compressor/decompressor
    astc-encoder

    # npx (for Playwright MCP)
    nodejs_24

    # Basic
    unzip
    zstd
    pv
  ];

  # Crucially, set LD_LIBRARY_PATH so the dynamic linker can find the libraries at runtime
  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      stdenv.cc.cc.lib
      libGL
      glfw
      xorg.libX11
      xorg.libXrandr
      xorg.libXinerama
      xorg.libXcursor
      xorg.libXi
      vulkan-loader
      # Wayland support
      wayland
      libxkbcommon
      libdecor
    ];
}
