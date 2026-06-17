param(
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$TrainManifest,
  [Parameter(Mandatory=$true)][string]$ValManifest,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$Device = "cuda",
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  python .\audio_app.py audio-train `
    --config $Config `
    --train-manifest $TrainManifest `
    --val-manifest $ValManifest `
    --output-dir $OutputDir `
    --device $Device `
    --seed $Seed
}
finally {
  Pop-Location
}
