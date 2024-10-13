{
  description = "An FHS shell for speechnoodle";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            allowBroken = true;
            cudaSupport = true;
            allowUnfreePredicate = pkg: true;
            acceptLicense = true;
          };
        };

        kernelPkgs = pkgs.linuxPackagesFor (pkgs.linux_6_10.override {
          argsOverride = rec {
            src = pkgs.fetchurl {
              url = "mirror://kernel/linux/kernel/v6.x/linux-${version}.tar.xz";
              sha256 = "sha256-UkhYhS9YaanvF96LHm5/rwW8ssRivJazwk2/gu3jc88=";
            };
            version = "6.10.12";
            modDirVersion = "6.10.12";
          };
        });

        nvidiaLatest = kernelPkgs.nvidiaPackages.latest;

        defaultDeps = [
          pkgs.ruff
          pkgs.nodejs
          pkgs.pyright
          pkgs.jq
          pkgs.uv
          pkgs.trufflehog
        ];
        cudaDeps = with pkgs; [
          autoconf
          curl
          freeglut
          gcc11
          git
          gitRepo
          gnumake
          gnupg
          gperf
          libGLU
          libGL
          libselinux
          m4
          ncurses5
          procps
          pkg-config # for triton build from src
          cmake # for triton build from src
          llvm_18 # for triton build from src
          stdenv.cc
          unzip
          util-linux
          wget
          xorg.libICE
          xorg.libSM
          xorg.libX11
          xorg.libXext
          xorg.libXi
          xorg.libXmu
          xorg.libXrandr
          xorg.libXrender
          xorg.libXv
          libxml2
          zlib
          nvidiaLatest
          cudaPackages_12_1.cudatoolkit
          file
          libaio
          file
          pkgs.cudaPackages_12_4.cudnn_9_3
          pkgs.cudaPackages_12_4.libcublas
          pkgs.cudaPackages_12_4.cuda_cudart
          pkgs.cudaPackages_12_4.cuda_cudart.static
          pkgs.pythonManylinuxPackages.manylinux2014Package
          pkgs.cudaPackages_12_4.nccl
          pkgs.cudaPackages_12_4.nvidia_fs
          pkgs.cudaPackages_12_4.nccl-tests
          pkgs.cudaPackages_12_4.tensorrt
        ];
        pyAndPkgs = pkgs.python311.withPackages (
          ps:
            with ps; [
              #triton
              #torch
              #torchvision
              #torchaudio
              onnx
            ]
        );
        fixedPython = pkgs.writeShellScriptBin "python" ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
          exec ${pyAndPkgs}/bin/python "$@"
        '';
        fixedPip = pkgs.writeShellScriptBin "pip" ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
          exec ${pyAndPkgs}/bin/pip "$@"
        '';
        fixedUv = pkgs.writeShellScriptBin "uv" ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
          exec ${pkgs.uv}/bin/uv "$@"
        '';
      in {
        _module.args = {inherit pkgs;};
        legacyPackages = pkgs;

        devShells = {
          default = (pkgs.mkShell.override {stdenv = pkgs.gcc11Stdenv;}) {
            name = "speechnoodle-env";
            venvDir = "./.venv";
            buildInputs = [
              fixedPython
              fixedPip
              fixedUv
              pyAndPkgs
              pkgs.python311Packages.virtualenv
              pkgs.python311Packages.venvShellHook
            ];
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';

            packages = defaultDeps ++ cudaDeps;
            NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.cmake
              pkgs.llvm_18
              pkgs.stdenv.cc.cc
              pkgs.zlib
              pkgs.zstd
              pkgs.fuse3
              pkgs.icu
              pkgs.nss
              pkgs.openssl
              pkgs.curl
              pkgs.attr
              pkgs.libssh
              pkgs.bzip2
              pkgs.libaio
              pkgs.file
              pkgs.libxml2
              pkgs.acl
              pkgs.libsodium
              pkgs.util-linux
              pkgs.xz
              pkgs.systemd
              pkgs.glibc_multi
              pkgs.expat
              pkgs.xorg.libX11
              pkgs.vulkan-headers
              pkgs.vulkan-loader
              pkgs.vulkan-tools
              pkgs.pkg-config
              pkgs.glibc
              pkgs.python311Packages.triton
              nvidiaLatest
              pkgs.cudaPackages_12_4.cudatoolkit
              pkgs.cudaPackages_12_4.libcublas
              pkgs.cudaPackages_12_4.cudnn_9_3
              pkgs.cudaPackages_12_4.cuda_cudart
              pkgs.cudaPackages_12_4.cuda_cudart.static
              pkgs.pythonManylinuxPackages.manylinux2014Package
              pkgs.cudaPackages_12_4.nccl
              pkgs.cudaPackages_12_4.nvidia_fs
              pkgs.cudaPackages_12_4.nccl-tests
              pkgs.cudaPackages_12_4.tensorrt
            ];
            # https://arestless.rest/blog/llamafile-on-nixos-23-11/
            # save this technique into awesome list
            # NVCC_APPEND_FLAGS="-L$(nix eval --impure --raw 'nixpkgs#cudaPackages_12.1.cuda_cudart.static')/lib";
            PYTHONPATH = "${pyAndPkgs}/${pyAndPkgs.sitePackages}";
            NVCC_APPEND_FLAGS = "-L${pkgs.cudaPackages_12_4.cuda_cudart.static}/lib"; # to make it available for triton and pytorch compilation
            TORCH_CUDA_ARCH_LIST = "8.9"; # support for 4090 not to compile useless compatibilities
            TRITON_LIBCUDA_PATH = "${nvidiaLatest}/lib/libcuda.so";
            NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
            CUDA_PATH = "${pkgs.cudaPackages_12_4.cudatoolkit}";
            TORCH_USE_CUDA_DSA = "1";
            CUDA_VISIBLE_DEVICES = "0,1,2";
            TORCH_DEVICE = "cuda";
            HF_HOME = "/shelf/hf_home";
            #TRANSFORMERS_CACHE = "/shelf/cache";
            PHONEMIZER_ESPEAK_PATH = "/nix/store/17m2mkgsrl4591696ym29693052sjc9v-espeak-ng-1.51.1/bin";
            PHONEMIZER_LANGUAGE = "fi";
            PHONEMIZER_ESPEAK_LIBRARY = "/nix/store/17m2mkgsrl4591696ym29693052sjc9v-espeak-ng-1.51.1/lib/libespeak-ng.so.1.1.51";
            shellHook = ''
              set -eu
              export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
              export UID_DOCKER=$(id -u)
              export GID_DOCKER=$(id -g)
              export TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
              source .venv/bin/activate
              export CGO_ENABLED=0
              export CUDA_PATH="${pkgs.cudaPackages_12_4.cudatoolkit}"
              export HF_HOME="/shelf/hf_home"
              export OMP_NUM_THREADS=32
              # Add CUDA and TensorRT to PATH and LD_LIBRARY_PATH
              export PATH=$PATH:${pkgs.cudaPackages_12_4.cudatoolkit}/bin
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath [pkgs.cudaPackages_12_4.cudatoolkit pkgs.cudaPackages_12_4.libcublas pkgs.cudaPackages_12_4.cudnn_9_3 pkgs.cudaPackages_12_4.tensorrt]}
              # Ensure Python can find the ONNX Runtime GPU libraries
              export PYTHONPATH=${pkgs.python311Packages.onnx}/lib/python3.11/site-packages:$PYTHONPATH
              export PHONEMIZER_ESPEAK_PATH=/nix/store/17m2mkgsrl4591696ym29693052sjc9v-espeak-ng-1.51.1/bin
              export PHONEMIZER_LANGUAGE="fi"
              export PHONEMIZER_ESPEAK_LIBRARY=/nix/store/17m2mkgsrl4591696ym29693052sjc9v-espeak-ng-1.51.1/lib/libespeak-ng.so.1.1.51

              # Load environment variables from .env file if it exists
              if [ -f .env ]; then
                set -a
                source .env
                set +a
              fi
            '';
          };
        };
      };
    };
}
