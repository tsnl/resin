let
  pkgs = import <nixpkgs> { };
  lib = pkgs.lib;
  linux = pkgs.stdenv.hostPlatform.isLinux;
  windowLibraries = with pkgs; lib.optionals linux [
    libx11
    libxcursor
    libxi
    libxinerama
    libxrandr
    libxkbcommon
    wayland
  ];
in
pkgs.mkShell ({
  # Fortify requires optimization, but Resin's default C builds use -O0.
  hardeningDisable = [ "fortify" ];

  packages = with pkgs; [
    rustup
    shaderc
  ] ++ lib.optionals linux (with pkgs; [
    vulkan-tools
    vulkan-validation-layers
    renderdoc
  ]);

  nativeBuildInputs = with pkgs; [ cmake pkg-config ] ++ lib.optional linux wayland-scanner;
  buildInputs = windowLibraries;

} // lib.optionalAttrs linux {
  LD_LIBRARY_PATH = lib.makeLibraryPath ([ pkgs.vulkan-loader pkgs.stdenv.cc.cc.lib ] ++ windowLibraries);
})
