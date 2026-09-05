let
  pkgs = import <nixpkgs> { };
  lib = pkgs.lib;
  linux = pkgs.stdenv.hostPlatform.isLinux;
  windowLibraries = lib.optionals linux (with pkgs; [
    libx11
    libxcursor
    libxi
    libxinerama
    libxrandr
    libxkbcommon
    wayland
  ]);
in
pkgs.mkShell {
  packages = with pkgs; [
    rustup
    shaderc
    vulkan-tools
    vulkan-validation-layers
  ] ++ lib.optionals linux [ pkgs.renderdoc ];

  nativeBuildInputs = with pkgs; [ cmake pkg-config ] ++ lib.optionals linux [ wayland-scanner ];
  buildInputs = windowLibraries;

  ${if pkgs.stdenv.hostPlatform.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"} =
    lib.makeLibraryPath ([ pkgs.vulkan-loader ] ++ windowLibraries
      ++ lib.optionals linux [ pkgs.stdenv.cc.cc.lib ]);
}
