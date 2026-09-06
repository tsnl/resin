let
  pkgs = import <nixpkgs> { };
  lib = pkgs.lib;
  windowLibraries = with pkgs; [
    libx11
    libxcursor
    libxi
    libxinerama
    libxrandr
    libxkbcommon
    wayland
  ];
in
assert lib.assertMsg pkgs.stdenv.hostPlatform.isLinux "Resin currently supports Linux only.";
pkgs.mkShell {
  packages = with pkgs; [
    rustup
    shaderc
    vulkan-tools
    vulkan-validation-layers
    renderdoc
  ];

  nativeBuildInputs = with pkgs; [ cmake pkg-config wayland-scanner ];
  buildInputs = windowLibraries;

  LD_LIBRARY_PATH = lib.makeLibraryPath ([ pkgs.vulkan-loader pkgs.stdenv.cc.cc.lib ] ++ windowLibraries);
}
