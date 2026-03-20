{
  pkgs ? import <nixpkgs> { },
}:

with pkgs;
mkShell {
  NIX_ENFORCE_NO_NATIVE = 0;
  buildInputs = [
    gcc
    llvmPackages.openmp
    clang-tools
  ];
}
