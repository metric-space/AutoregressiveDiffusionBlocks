{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.torch
      ps.lightning
      ps.torchvision
      ps.datasets
      ps.hydra-core
      ps.wandb
      ps.regex
    ]))
  ];
  shellHook = ''
    echo "Entered Python development environment"
  '';
}
