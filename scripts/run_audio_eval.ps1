param(
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [Parameter(Mandatory=$true)][string]$Manifest,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  python .\audio_app.py audio-eval `
    --config $Config `
    --checkpoint $Checkpoint `
    --manifest $Manifest `
    --output-dir $OutputDir `
    --device $Device
}
finally {
  Pop-Location
}
