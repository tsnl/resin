{
  pkgs ? import <nixpkgs> { },
}:
let
  libPath =
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
    ];
in
{
  devShell =
    with pkgs;
    mkShell {
      buildInputs = [
        cargo

        # nix-ld is required for UV
        nix-ld

        # Python
        uv

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

        # amdgpu_top
        amdgpu_top

        # astcenc: ASTC texture compressor/decompressor
        astc-encoder

        # Basic
        unzip
        zstd
        pv
      ];

      RUST_LOG = "debug";
      RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
      LD_LIBRARY_PATH = libPath;
    };
}
