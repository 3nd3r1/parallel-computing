{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

with pkgs;
mkShell {
  NIX_ENFORCE_NO_NATIVE = 0;
  buildInputs = [
    gcc
    llvmPackages.openmp
    clang-tools
    cudaPackages.cudatoolkit
  ];
  CUDA_PATH = "${cudaPackages.cudatoolkit}";
}
