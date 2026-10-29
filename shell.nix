{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      problems.handlers = {
        torch.unsupported-cuda-version = "warn";
      };
    };

    overlays = [
      (final: prev: {
        cudaPackages = prev.cudaPackages_13_0;
      })
    ];
  }
}:

pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (ps: with ps; [
      # Use the pre-compiled binary versions which include CUDA out-of-the-box
 #     torchvision-bin
#      torch-bin
    ]))
    libxcb
    zlib
    libx11
    libxext
    libsm
    libice
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.xorg.libxcb
          pkgs.libGL
          pkgs.glib
          pkgs.zlib
          pkgs.libx11
          pkgs.libxext
          pkgs.libsm
          pkgs.libice
          # Add cudatoolkit/cuda libraries to LD_LIBRARY_PATH if torch-bin needs them
  #        pkgs.cudaPackages.cudatoolkit
        ]}:$LD_LIBRARY_PATH

    # opencv-python bundles Qt with only the xcb plugin; force it so Qt does not
    # probe for the (absent) wayland plugin on Wayland sessions.
    export QT_QPA_PLATFORM=xcb
  '';
}
