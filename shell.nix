# shell.nix

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [ 
    #pkgs.cmake
    #pkgs.stdenv
    # pkgs.cplex
    pkgs.python3
    pkgs.gcc
    #pkgs.osi
  ];  
  nativeBuildInputs = [ 
    pkgs.cmake
    pkgs.gcc
    pkgs.gcc_multi
    pkgs.python3.pkgs.wrapPython
    pkgs.gcc
    pkgs.libtorch-bin
    pkgs.cudaPackages.cudatoolkit
  ]; 
  shellHook = ''
    # nix-shell runs shellHook under POSIX sh/bash.
    # Help CMake find libtorch's TorchConfig.cmake.
    export Torch_DIR="${pkgs.libtorch-bin}/libtorch/share/cmake/Torch"
    if [ -n "''${CMAKE_PREFIX_PATH-}" ]; then
      export CMAKE_PREFIX_PATH="${pkgs.libtorch-bin}/libtorch:$CMAKE_PREFIX_PATH"
    else
      export CMAKE_PREFIX_PATH="${pkgs.libtorch-bin}/libtorch"
    fi
  '';
}
