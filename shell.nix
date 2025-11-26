let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  # Add necessary system libraries to the build inputs
  buildInputs = with pkgs; [
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

    # RenderDoc
    renderdoc
  ];

  # Crucially, set LD_LIBRARY_PATH so the dynamic linker can find the libraries at runtime
  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      libGL
      glfw
      xorg.libX11
      xorg.libXrandr
      xorg.libXinerama
      xorg.libXcursor
      xorg.libXi
    ];
}
