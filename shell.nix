{
  pkgs ? import <nixpkgs> { },
}:

with pkgs;
mkShell {
  buildInputs = [
    gcc
    llvmPackages.openmp
    clang-tools
  ];
}
