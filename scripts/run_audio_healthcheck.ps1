param(
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$Manifest,
  [string]$OutputDir = "",
  [int]$MaxSamples = 3,
  [switch]$FailOnMissing
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $args = @(
    ".\audio_app.py", "audio-healthcheck",
    "--config", $Config,
    "--manifest", $Manifest,
    "--max-samples", "$MaxSamples"
  )
  if ($OutputDir) {
    $args += @("--output-dir", $OutputDir)
  }
  if ($FailOnMissing) {
    $args += "--fail-on-missing"
  }
  python @args
}
finally {
  Pop-Location
}
