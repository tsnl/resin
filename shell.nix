let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  # Add necessary system libraries to the build inputs
  buildInputs = with pkgs; [
    rustc
    cargo
    glfw # The GLFW library
    libGL # The OpenGL library
    # Add X11 dependencies if targeting X11 (default for many setups)
    xorg.libX11
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi
    cmake # often required for the glfw-rs build script
    pkg-config
    wayland
    wayland-protocols
  ];

  # Crucially, set LD_LIBRARY_PATH so the dynamic linker can find the libraries at runtime
  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      libGL
      glfw
      xorg.libX11
      xorg.libXrandr
      xorg.libXinerama
      xorg.libXcursor
      xorg.libXi
    ];

  # Optional: set RUST_SRC_PATH for rust-analyzer/IDE support
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
}
