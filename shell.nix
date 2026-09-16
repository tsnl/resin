let
  pkgs = import <nixpkgs> { };
  lib = pkgs.lib;
  linux = pkgs.stdenv.hostPlatform.isLinux;
  windowLibraries =
    with pkgs;
    lib.optionals linux [
      libx11
      libxcursor
      libxi
      libxinerama
      libxrandr
      libxkbcommon
      wayland
    ];
in
pkgs.mkShell (
  {
    # Handwritten native test fixtures include unoptimized C builds.
    hardeningDisable = [ "fortify" ];

    packages =
      with pkgs;
      [
        rustup
        helix
        clang
        libclang
        shaderc # Handwritten GLSL fixtures in the runtime tests.
        spirv-tools
        nodejs_26
      ]
      ++ lib.optionals linux (
        with pkgs;
        [
          vulkan-tools
          vulkan-validation-layers
          renderdoc
        ]
      );

    nativeBuildInputs =
      with pkgs;
      [
        cmake
        ninja
        pkg-config
      ]
      ++ lib.optional linux wayland-scanner;
    buildInputs = windowLibraries;
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";

  }
  // lib.optionalAttrs linux {
    LD_LIBRARY_PATH = lib.makeLibraryPath (
      [
        pkgs.vulkan-loader
        pkgs.stdenv.cc.cc.lib
      ]
      ++ windowLibraries
    );
  }
)
